//===--- Tracing.h - A layer for tracing and reproducibility ----*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines API for performance analysis and building reproducers.
//
//===----------------------------------------------------------------------===//

#ifndef CPPINTEROP_TRACING_H
#define CPPINTEROP_TRACING_H

// Visibility for tracing symbols that must be exported from the shared
// library so that tests (and the crash handler) can access them.
#if defined(_WIN32) || defined(__CYGWIN__)
#define CPPINTEROP_TRACE_API __declspec(dllexport)
#elif defined(__GNUC__)
#define CPPINTEROP_TRACE_API __attribute__((__visibility__("default")))
#else
#define CPPINTEROP_TRACE_API
#endif

#include "CppInterOp/CppInterOpTypes.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace CppInterOp {
namespace Tracing {

class TraceInfo;

/// Process-global tracer pointer. Exported because TracingTests and
/// the crash handler (linked-mode consumers) read it directly; cppyy
/// and Dispatch.h consumers go through the DispatchRaw trace slots
/// declared in CppInterOpTypes.h instead.
extern CPPINTEROP_TRACE_API TraceInfo* TheTraceInfo;

class TraceInfo {
  llvm::TimerGroup m_TG;
  llvm::StringMap<std::unique_ptr<llvm::Timer>> m_Timers;
  std::vector<llvm::Timer*> m_TimerStack;

  std::unordered_map<const void*, std::string> m_HandleMap;
  unsigned m_VarCount = 0;
  /// Monotonic suffix for `_retN` placeholders -- one slot per call
  /// that returns std::vector<P*>. Separate from m_VarCount so the
  /// vN handle namespace stays dense.
  unsigned m_RetCount = 0;
  /// Monotonic suffix for `_outN` -- one slot per distinct OUT
  /// pointer-container source. Multiple calls passing the same
  /// container alias the same slot (see m_OutAliases).
  unsigned m_OutCount = 0;
  /// Source address of an OUT container -> its `_outN` index. Lets
  /// the reproducer faithfully replay the API's append-on-call
  /// contract: a vector reused across calls keeps its single buffer.
  std::unordered_map<const void*, unsigned> m_OutAliases;

  std::vector<std::string> m_Log;
  size_t m_RegionStart = 0; ///< Log index where current region began.
  bool m_InRegion = false;  ///< True between StartTracing/StopTracing.
  /// Set while the reproducer is being emitted. Gates appendToLog and
  /// setLogEntry so the dumper's own INTEROP_TRACE-wrapped calls
  /// (GetVersion, GetBuildInfo) don't recurse into m_Log.
  bool m_Dumping = false;

public:
  TraceInfo() : m_TG("CppInterOp", "CppInterOp Timing Report") {}
  ~TraceInfo() { TheTraceInfo = nullptr; }
  TraceInfo(const TraceInfo&) = delete;
  TraceInfo& operator=(const TraceInfo&) = delete;
  TraceInfo(TraceInfo&&) = delete;
  TraceInfo& operator=(TraceInfo&&) = delete;

  static bool isEnabled() { return TheTraceInfo; }

  llvm::Timer& getTimer(llvm::StringRef Name) {
    auto& T = m_Timers[Name];
    if (!T)
      T = std::make_unique<llvm::Timer>(Name, Name, m_TG);
    return *T;
  }

  /// True when at least one TraceRegion is currently active.
  bool insideTracedRegion() const { return !m_TimerStack.empty(); }

  void pushTimer(llvm::Timer* T) {
    if (!m_TimerStack.empty())
      m_TimerStack.back()->stopTimer();
    m_TimerStack.push_back(T);
    T->startTimer();
  }

  void popTimer() {
    if (m_TimerStack.empty())
      return;
    m_TimerStack.back()->stopTimer();
    m_TimerStack.pop_back();
    if (!m_TimerStack.empty())
      m_TimerStack.back()->startTimer();
  }

  std::string getOrRegisterHandle(const void* p) {
    if (!p)
      return "";
    auto it = m_HandleMap.find(p);
    if (it != m_HandleMap.end())
      return it->second;
    return m_HandleMap[p] = "v" + std::to_string(++m_VarCount);
  }

  /// Resolve a pointer to a printable form.
  /// - "nullptr" for an actual null pointer.
  /// - "vN" for a pointer registered via getOrRegisterHandle.
  /// - "" (empty) for a non-null pointer never seen by the tracer --
  ///   the caller decides how to render the unknown case (e.g.
  ///   `nullptr /*unknown*/` in argument lists, "is this new?" in the
  ///   producer-side `auto vN = ...` gating logic).
  std::string lookupHandle(const void* p) {
    if (!p)
      return "nullptr";
    auto it = m_HandleMap.find(p);
    return (it != m_HandleMap.end()) ? it->second : "";
  }

  /// Allocate the next `_retN` index for a vector-return placeholder.
  unsigned nextRetIndex() { return m_RetCount++; }

  /// Resolve an OUT-container source address to its `_outN` index.
  /// First call with a given address allocates a fresh slot; later
  /// calls return the same slot so the reproducer reuses the buffer.
  /// \pre \p Addr is non-null (MakeOutParam captures `&C`).
  /// \returns {idx, true} on first use, {idx, false} on alias.
  std::pair<unsigned, bool> outIndexFor(const void* Addr) {
    assert(Addr && "OutParam without a source address");
    auto it = m_OutAliases.find(Addr);
    if (it != m_OutAliases.end())
      return {it->second, false};
    unsigned idx = m_OutCount++;
    m_OutAliases.emplace(Addr, idx);
    return {idx, true};
  }

  /// Append a line; the returned index pairs with setLogEntry to
  /// rewrite the same slot later (TraceRegion's placeholder pattern).
  size_t appendToLog(const std::string& line) {
    if (m_Dumping)
      return 0;
    size_t idx = m_Log.size();
    m_Log.push_back(line);
    if (m_InRegion && m_WriteOnStdErr)
      llvm::errs() << line << "\n";
    return idx;
  }
  void setLogEntry(size_t idx, const std::string& line) {
    if (m_Dumping)
      return;
    m_Log[idx] = line;
  }
  void setDumping(bool v) { m_Dumping = v; }
  const std::vector<std::string>& getLog() const { return m_Log; }
  std::string getLastLogEntry() const {
    return m_Log.empty() ? "" : m_Log.back();
  }

  /// Write the accumulated reproducer log to a file.
  /// \param Version optional version string embedded as comments.
  CPPINTEROP_TRACE_API std::string writeToFile(const std::string& Version = "");

  /// Begin a traced region. Returns the path where StopTracing will write.
  /// \param WriteOnStdErr if true, also emit the reproducer to stderr.
  CPPINTEROP_TRACE_API std::string StartRegion(bool WriteOnStdErr = true);

  /// End the traced region and write only the region's entries to the file.
  CPPINTEROP_TRACE_API void StopRegion(const std::string& Version = "");

private:
  std::string m_RegionPath;
  bool m_WriteOnStdErr = false;

public:
  void clear() {
    // Stop any running timers before clearing to avoid triggering
    // TimerGroup's destructor report.
    while (!m_TimerStack.empty()) {
      m_TimerStack.back()->stopTimer();
      m_TimerStack.pop_back();
    }
    // Clear timers after clearing the group to suppress the report.
    m_TG.clear();
    m_Timers.clear();
    m_HandleMap.clear();
    m_Log.clear();
    m_VarCount = 0;
    m_RetCount = 0;
    m_OutCount = 0;
    m_OutAliases.clear();
  }
};

/// Activate tracing. Called once during process initialization.
/// After this, TheTraceInfo is non-null and all INTEROP_TRACE calls record.
CPPINTEROP_TRACE_API void InitTracing();

/// Begin recording a traced region. If tracing is not yet active, activates
/// it. Returns the path where StopTracing() will write the reproducer.
/// \param WriteOnStdErr if true, also emit the reproducer to stderr on stop.
inline std::string StartTracing(bool WriteOnStdErr = true) {
  if (!TheTraceInfo)
    InitTracing();
  return TheTraceInfo->StartRegion(WriteOnStdErr);
}

/// End the traced region and write the reproducer file containing only the
/// calls made between StartTracing() and this call.
inline void StopTracing(const std::string& Version = "") {
  if (TheTraceInfo)
    TheTraceInfo->StopRegion(Version);
}

/// Marks a function parameter as an output container (e.g. std::vector<T>&
/// that the function fills). Constructed via the INTEROP_OUT(var) macro.
///
/// Purpose:
///  - Excluded from the reproducer's argument list (it's an output, not input).
///  - If the container holds pointers, its elements are registered as handles
///    at trace-region exit so later calls can reference them by name.
///
/// The container type is erased at construction via MakeOutParam() so that
/// all downstream code (ReproBuffer, TraceRegion) works with a single
/// concrete type — no template specializations or detection traits needed.
struct OutParam {
  /// Register the container's pointer elements as handles and emit
  /// one `void* vN = _outN[i] : nullptr;` decl per newly-registered
  /// element so a later call referencing the registered name finds a
  /// real binding.
  std::function<void(TraceInfo&, unsigned OutIdx)> RegisterHandles;
  /// Drives the `_outN` preamble + alias path. Non-pointer containers
  /// stay on the legacy "skip in arg list" rendering -- their type is
  /// not erasable to void*, so the preamble decl would not type-check.
  bool IsPointerContainer = false;
  /// Scalar pointer OUT (e.g. `bool*`); rendered as `nullptr`.
  bool IsScalarPointer = false;
  /// Address of the source container object (not its data buffer);
  /// multiple calls with the same container alias the same `_outN`
  /// buffer in the reproducer so accumulation across calls replays
  /// faithfully. Stable across capacity growth -- a `push_back` that
  /// reallocates the heap buffer leaves the object's address put.
  const void* SourceAddr = nullptr;
};

/// Create an OutParam for any container. Only sets up handle registration
/// when the container's value_type is a pointer.
template <typename Container> OutParam MakeOutParam(const Container& C) {
  OutParam OP;
  OP.SourceAddr = static_cast<const void*>(&C);
  using Value = typename Container::value_type;
  if constexpr (std::is_pointer_v<Value>) {
    OP.IsPointerContainer = true;
    OP.RegisterHandles = [&C](TraceInfo& TI, unsigned OutIdx) {
      size_t I = 0;
      for (const auto& Elem : C) {
        const void* P = static_cast<const void*>(Elem);
        bool isNew = TI.lookupHandle(P).empty();
        std::string Name = TI.getOrRegisterHandle(P);
        // Bounds-guard with `.size() > i` for replays where the
        // function fills fewer elements than the original call did.
        if (isNew && P)
          TI.appendToLog(llvm::formatv("  void* {0} = _out{1}.size() > {2} ? "
                                       "_out{1}[{2}] : nullptr;",
                                       Name, OutIdx, I)
                             .str());
        ++I;
      }
    };
  }
  return OP;
}

/// Scalar pointer overload (e.g. `bool* HadError`). Rendered as
/// `nullptr` at the call site -- replay only needs the call sequence.
template <typename T> OutParam MakeOutParam(T*) {
  OutParam OP;
  OP.IsScalarPointer = true;
  return OP;
}

/// Internal helper to stringify arguments into a C++ call format.
struct ReproBuffer {
  llvm::SmallString<128> Buffer;
  llvm::raw_svector_ostream OS;

  ReproBuffer() : OS(Buffer) {}

  // Opaque handle structs — unwrap .data and resolve to registered name.
  void append(Cpp::DeclRef h) { append(h.data); }
  void append(Cpp::TypeRef h) { append(h.data); }
  void append(Cpp::FuncRef h) { append(h.data); }
  void append(Cpp::ObjectRef h) { append(h.data); }
  void append(Cpp::InterpRef h) { append(h.data); }
  void append(Cpp::ConstDeclRef h) { append(h.data); }
  void append(Cpp::ConstTypeRef h) { append(h.data); }
  void append(Cpp::ConstFuncRef h) { append(h.data); }

  // Raw void* pointers — resolved to their registered name.
  void append(const void* p) {
    if (!p) {
      OS << "nullptr";
      return;
    }
    auto h = TheTraceInfo->lookupHandle(p);
    if (h.empty())
      OS << "nullptr /*unknown*/";
    else
      OS << h;
  }

  // Strings -- emit a plain `"..."` literal when the content is
  // printable ASCII without `"`, `\`, or control chars; otherwise
  // fall back to `R"CPPI(...)CPPI"`, which passes the bytes through
  // verbatim. The raw form is reserved for the cases that actually
  // need it (quotes, backslashes, newlines from Cpp::Process /
  // Cpp::Declare source blocks); plain literals keep the trace lines
  // short and human-readable for the common case (identifiers,
  // paths, simple expressions).
  void appendRaw(std::string_view s) {
    bool NeedsRaw = false;
    for (char c : s) {
      auto u = static_cast<unsigned char>(c);
      // 0x20 is space: anything below it is a C0 control char (incl. \n,
      // \t, \r) that would break a plain quoted literal; 0x7f is DEL.
      if (c == '"' || c == '\\' || u < 0x20 || u == 0x7f) {
        NeedsRaw = true;
        break;
      }
    }
    if (NeedsRaw)
      OS << "R\"CPPI(" << s << ")CPPI\"";
    else
      OS << '"' << s << '"';
  }
  void append(const char* s) {
    appendRaw(s ? std::string_view(s) : std::string_view());
  }
  void append(const std::string& s) { appendRaw(s); }

  // Numeric types — printed directly.
  void append(bool v) { OS << (v ? "true" : "false"); }
  void append(int v) { OS << v; }
  void append(unsigned v) { OS << v; }
  void append(long v) { OS << v; }
  void append(unsigned long v) { OS << v; }
  void append(long long v) { OS << v; }
  void append(unsigned long long v) { OS << v; }
  void append(double d) { OS << llvm::formatv("{0:f}", d); }
  void append(float f) { OS << llvm::formatv("{0:f}", f); }

  // Vector input args -- emit a braced init list of recursively-formatted
  // elements so the reproducer's call-site literal compiles.
  template <typename T> void append(const std::vector<T>& V) {
    OS << "{";
    bool First = true;
    for (const auto& E : V) {
      if (!First)
        OS << ", ";
      First = false;
      append(E);
    }
    OS << "}";
  }

  // TemplateArgInfo: brace-init through the (DeclRef, const char*)
  // ctor so the reproducer compiles. m_Type takes the void* path
  // (renders as vN); nullptr m_IntegralValue must render as `nullptr`
  // (the ctor's default), not the empty string the const char* path
  // would produce.
  void append(const Cpp::TemplateArgInfo& tai) {
    OS << "Cpp::TemplateArgInfo{";
    append(static_cast<const void*>(tai.m_Type));
    OS << ", ";
    if (tai.m_IntegralValue == nullptr)
      OS << "nullptr";
    else
      appendRaw(tai.m_IntegralValue);
    OS << "}";
  }

  // Enums: emit "static_cast<EnumName>(N)" so the reproducer compiles.
  // EnumName comes from the compiler's pretty signature -- no RTTI.
  // Compute outside any lambda: gcc/clang substitute T into a lambda's
  // signature, which drops the "T = " marker we parse from.
  static std::string parseEnumName(const char* sig) {
    llvm::StringRef s(sig);
#ifdef _MSC_VER
    // MSVC: "... ReproBuffer::append<enum QualKind>(enum QualKind)"
    auto pos = s.find("append<");
    if (pos == llvm::StringRef::npos)
      return {};
    s = s.drop_front(pos + 7);
    for (auto kw : {"enum ", "class ", "struct "})
      if (s.consume_front(kw))
        break;
    return s.take_until([](char c) { return c == '>'; }).str();
#else
    // gcc:   "... [with T = QualKind; ...]"
    // clang: "... [T = QualKind]"
    auto pos = s.find("T = ");
    if (pos == llvm::StringRef::npos)
      return {};
    return s.drop_front(pos + 4)
        .take_until([](char c) { return c == ',' || c == ';' || c == ']'; })
        .str();
#endif
  }
  template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
  void append(T v) {
#ifdef _MSC_VER
    static const std::string TN = parseEnumName(__FUNCSIG__);
#else
    static const std::string TN = parseEnumName(__PRETTY_FUNCTION__);
#endif
    OS << "static_cast<" << TN << ">("
       << +static_cast<std::underlying_type_t<T>>(v) << ")";
  }
  // Catch-all for non-enum, non-pointer types.
  template <
      typename T,
      std::enable_if_t<!std::is_enum_v<T> && !std::is_pointer_v<T>, int> = 0>
  void append(const T&) {
    OS << "?";
  }

  /// Format a comma-separated argument list. Pointer-container OUTs
  /// emit `_outN` (next index from \p OutIndices); scalar pointer OUTs
  /// emit `nullptr` (the replay does not consume the value); non-pointer
  /// containers are skipped, matching the legacy rendering.
  template <typename... Args>
  void format(llvm::ArrayRef<unsigned> OutIndices, Args&&... args) {
    bool first = true;
    size_t nextOut = 0;
    auto appendOne = [&](auto&& val) {
      using V = std::decay_t<decltype(val)>;
      if constexpr (std::is_same_v<V, OutParam>) {
        if (val.IsPointerContainer) {
          if (!first)
            OS << ", ";
          first = false;
          OS << "_out" << OutIndices[nextOut++];
        } else if (val.IsScalarPointer) {
          if (!first)
            OS << ", ";
          first = false;
          OS << "nullptr";
        }
      } else {
        if (!first)
          OS << ", ";
        first = false;
        append(std::forward<decltype(val)>(val));
      }
    };
    (appendOne(std::forward<Args>(args)), ...);
  }
};

/// Matches std::vector<T*> for any pointer element type. Used by
/// TraceRegion::record to recognise functions whose return value is a
/// vector of opaque handles (e.g. GetFunctionsUsingName), so the
/// reproducer can name the vector and read its elements out by index.
template <typename T> struct is_pointer_vector : std::false_type {};
template <typename T>
struct is_pointer_vector<std::vector<T*>> : std::true_type {};
template <typename T>
inline constexpr bool is_pointer_vector_v = is_pointer_vector<T>::value;

/// Detect opaque handle structs (Cpp::DeclRef, Cpp::TypeRef, etc.)
template <typename T>
inline constexpr bool is_handle_v =
    std::is_same_v<T, Cpp::DeclRef> || std::is_same_v<T, Cpp::TypeRef> ||
    std::is_same_v<T, Cpp::FuncRef> || std::is_same_v<T, Cpp::ObjectRef> ||
    std::is_same_v<T, Cpp::InterpRef> || std::is_same_v<T, Cpp::ConstDeclRef> ||
    std::is_same_v<T, Cpp::ConstTypeRef> ||
    std::is_same_v<T, Cpp::ConstFuncRef>;

/// Detect vector-of-handle types.
template <typename T> struct is_handle_vector : std::false_type {};
template <>
struct is_handle_vector<std::vector<Cpp::DeclRef>> : std::true_type {};
template <>
struct is_handle_vector<std::vector<Cpp::FuncRef>> : std::true_type {};
template <typename T>
inline constexpr bool is_handle_vector_v = is_handle_vector<T>::value;

/// Holds all the data that is only needed when tracing is active.
/// Heap-allocated only when TheTraceInfo != nullptr, so disabled tracing
/// pays zero cost beyond a single pointer + bool on the stack.
struct TraceData {
  const char* Name;
  llvm::SmallString<128> ArgStr;
  void* Result = nullptr;
  bool HasPtrResult = false;
  /// Set when the call returned std::vector<P*>; ~TraceRegion uses this
  /// to spell `auto _retN = Cpp::Foo(...)` so element decls can read
  /// `_retN[i]` and the replay sees the original pointers.
  bool HasRetVec = false;
  /// Snapshot of the returned vector's pointer elements, taken at
  /// record() time -- the source vector goes out of scope before
  /// ~TraceRegion runs.
  llvm::SmallVector<const void*, 8> RetVecPtrs;
  double StartTime = 0;
  bool Returned = false;
  llvm::SmallVector<std::function<void(TraceInfo&, unsigned)>, 2> OutCallbacks;
  /// `_outN` indices for this call's OUT pointer-containers, in
  /// left-to-right order. Consumed by format() at the call site and
  /// by ~TraceRegion to emit the preamble decls.
  llvm::SmallVector<unsigned, 2> OutIndices;
  /// Per-OUT argument flag: true on the first call with this source
  /// container (preamble decl needed), false on later calls reusing
  /// the same slot.
  llvm::SmallVector<bool, 2> OutFirstUse;
  /// Set when another TraceRegion was already active at construction.
  /// Nested calls skip log emission but still register handles.
  bool Nested = false;
  /// Slot for the ctor-emitted placeholder; the dtor rewrites it.
  size_t LogIndex = 0;
};

class TraceRegion {
  std::unique_ptr<TraceData> m_Data;

  /// Resolve the OUT pointer-container's `_outN` index up-front so
  /// format() can render the call site's alias and the dtor knows
  /// whether to emit the preamble decl (only on first use of a given
  /// source container). Skipped for nested calls (no emission); the
  /// handle callback still queues so the outer call can resolve names.
  void captureArg(OutParam&& op) {
    if (!m_Data->Nested && op.IsPointerContainer) {
      auto [idx, firstUse] = TheTraceInfo->outIndexFor(op.SourceAddr);
      m_Data->OutIndices.push_back(idx);
      m_Data->OutFirstUse.push_back(firstUse);
    }
    if (op.RegisterHandles)
      m_Data->OutCallbacks.push_back(std::move(op.RegisterHandles));
  }
  template <typename T> void captureArg(T&&) {}

public:
  template <typename... Args> TraceRegion(const char* Name, Args&&... args) {
    if (!TheTraceInfo)
      return;
    m_Data = std::make_unique<TraceData>();
    m_Data->Name = Name;
    TraceInfo& TI = *TheTraceInfo;
    // Detect nesting before pushing this call's frame.
    m_Data->Nested = TI.insideTracedRegion();
    // captureArg before format(): it fills OutIndices that format()
    // consumes.
    (captureArg(std::forward<Args>(args)), ...);
    if (!m_Data->Nested) {
      if constexpr (sizeof...(args) > 0) {
        ReproBuffer RB;
        RB.format(m_Data->OutIndices, std::forward<Args>(args)...);
        m_Data->ArgStr = RB.Buffer;
      }
      // `_outN` preamble + placeholder so an abort before the dtor
      // still leaves a record of the failing call. Emit the preamble
      // only on first use of each source container; reused ones alias
      // the existing slot.
      for (size_t i = 0; i < m_Data->OutIndices.size(); ++i)
        if (m_Data->OutFirstUse[i])
          TI.appendToLog(llvm::formatv("  std::vector<void*> _out{0};",
                                       m_Data->OutIndices[i])
                             .str());
      m_Data->LogIndex = TI.appendToLog(
          llvm::formatv("  Cpp::{0}({1}); // [aborted before return]",
                        m_Data->Name, m_Data->ArgStr)
              .str());
    }
    TI.pushTimer(&TI.getTimer(Name));
    m_Data->StartTime = llvm::TimeRecord::getCurrentTime(false).getWallTime();
  }

  ~TraceRegion() {
    if (!m_Data)
      return;

    if (!m_Data->Returned) {
      llvm::errs() << "ERROR: Function '" << m_Data->Name
                   << "' exited without calling INTEROP_RETURN!\n";
      assert(
          m_Data->Returned &&
          "Unannotated exit branch detected: use `return INTEROP_RETURN(...)`");
    }

    auto EndTime = llvm::TimeRecord::getCurrentTime(false).getWallTime();
    auto Dur = static_cast<long long>((EndTime - m_Data->StartTime) * 1e9);
    TraceInfo& TI = *TheTraceInfo;
    TI.popTimer();

    // Nested calls don't reach the log -- their args reference
    // outer-scope locals the reproducer doesn't have.
    if (m_Data->Nested) {
      m_Data.reset();
      return;
    }

    // Allocate a `_retN` slot up-front for vector-of-pointer returns so
    // the call line spells `auto _retN = ...` and the per-element decls
    // below can read `_retN[i]` (the replay sees the original
    // pointers, not literal nulls).
    int RetIdx = -1;
    llvm::SmallVector<std::pair<size_t, std::string>, 8> RetDecls;
    if (m_Data->HasRetVec) {
      RetIdx = static_cast<int>(TI.nextRetIndex());
      for (size_t i = 0; i < m_Data->RetVecPtrs.size(); ++i) {
        const void* p = m_Data->RetVecPtrs[i];
        if (!p)
          continue;
        bool isNew = TI.lookupHandle(p).empty();
        std::string Name = TI.getOrRegisterHandle(p);
        if (isNew)
          RetDecls.emplace_back(i, std::move(Name));
      }
    }

    std::string VarPart;
    if (m_Data->Result) {
      bool isNew = TI.lookupHandle(m_Data->Result).empty();
      std::string HandleName = TI.getOrRegisterHandle(m_Data->Result);
      VarPart =
          llvm::formatv(isNew ? "auto {0} = " : "/*{0}*/ ", HandleName).str();
    } else if (m_Data->HasPtrResult) {
      VarPart = "/*nullptr*/ ";
    } else if (RetIdx >= 0) {
      VarPart = llvm::formatv("auto _ret{0} = ", RetIdx).str();
    }

    // Rewrite the placeholder with the completed call line.
    std::string Call = llvm::formatv("  {0}Cpp::{1}({2}); // [{3} ns]", VarPart,
                                     m_Data->Name, m_Data->ArgStr, Dur);
    TI.setLogEntry(m_Data->LogIndex, Call);

    // After the call: register OUT-element handles and emit one
    // `void* vN = _outN[i] : nullptr;` decl per newly-seen element so a
    // later call referencing the registered name finds a real binding.
    for (size_t i = 0; i < m_Data->OutCallbacks.size(); ++i)
      m_Data->OutCallbacks[i](TI, m_Data->OutIndices[i]);

    // Per-element extractions for vector returns. Bounds-guard so the
    // line is safe even if the replay produces a shorter vector than
    // the original.
    for (auto& [Idx, Name] : RetDecls)
      TI.appendToLog(
          llvm::formatv(
              "  void* {0} = _ret{1}.size() > {2} ? _ret{1}[{2}] : nullptr;",
              Name, RetIdx, Idx)
              .str());

    m_Data.reset();
  }

  TraceRegion(const TraceRegion&) = delete;
  TraceRegion& operator=(const TraceRegion&) = delete;
  TraceRegion(TraceRegion&&) noexcept = default;
  TraceRegion& operator=(TraceRegion&&) = delete;

  [[nodiscard]] bool isActive() const { return m_Data != nullptr; }

  /// Record a non-void return value. Tracks pointer results for the
  /// reproducer's handle chain (e.g. auto v1 = Cpp::GetScope(...)).
  template <typename T> T record(T val) {
    if (!m_Data)
      return val;
    m_Data->Returned = true;
    if constexpr (is_handle_v<T>) {
      m_Data->HasPtrResult = true;
      m_Data->Result = const_cast<void*>(static_cast<const void*>(val.data));
    } else if constexpr (std::is_pointer_v<T>) {
      m_Data->HasPtrResult = true;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      m_Data->Result = const_cast<void*>(static_cast<const void*>(val));
    } else if constexpr (std::is_null_pointer_v<T>) {
      m_Data->HasPtrResult = true;
    } else if constexpr (is_handle_vector_v<T>) {
      m_Data->HasRetVec = true;
      for (const auto& h : val)
        m_Data->RetVecPtrs.push_back(h.data);
    } else if constexpr (is_pointer_vector_v<T>) {
      // Snapshot the element pointers; the source vector is owned by
      // the API call site and goes out of scope before ~TraceRegion.
      m_Data->HasRetVec = true;
      for (auto* p : val)
        m_Data->RetVecPtrs.push_back(static_cast<const void*>(p));
    }
    return val;
  }

  void recordVoid() {
    if (m_Data)
      m_Data->Returned = true;
  }

  /// Count parameters from a __PRETTY_FUNCTION__ / __FUNCSIG__ string.
  /// Counts top-level commas between the outermost '(' and ')' of the
  /// function signature, handling nested <>, (), and [] correctly.
  static unsigned countParams(const char* sig) {
    // Find the first '(' — that's the start of the parameter list.
    const char* p = sig;
    while (*p && *p != '(')
      ++p;
    if (!*p)
      return 0;
    ++p; // skip '('

    // Skip whitespace.
    while (*p == ' ')
      ++p;
    if (*p == ')')
      return 0; // empty parameter list

    // MSVC's __FUNCSIG__ uses (void) for no-parameter functions.
    if (p[0] == 'v' && p[1] == 'o' && p[2] == 'i' && p[3] == 'd' &&
        (p[4] == ')' || (p[4] == ' ' && p[5] == ')')))
      return 0;

    unsigned count = 1; // at least one parameter
    int depth = 0;      // nesting depth for <>, (), []
    for (; *p; ++p) {
      switch (*p) {
      case '<':
      case '(':
      case '[':
        ++depth;
        break;
      case '>':
      case ')':
      case ']':
        if (depth > 0)
          --depth;
        else
          return count; // closing ')' of the parameter list
        break;
      case ',':
        if (depth == 0)
          ++count;
        break;
      default:
        break;
      }
    }
    return count;
  }

  /// Bit i set iff Args[i] is OutParam (i.e. wrapped in INTEROP_OUT).
  /// The trailing `false` keeps the array non-empty for sizeof...(Args)==0.
  template <typename... Args> static constexpr uint64_t computeOutMask() {
    constexpr bool isOut[] = {std::is_same_v<std::decay_t<Args>, OutParam>...,
                              false};
    uint64_t mask = 0;
    for (std::size_t i = 0; i < sizeof...(Args); ++i)
      if (isOut[i])
        mask |= 1ULL << i;
    return mask;
  }

  /// Helper to allow INTEROP_TRACE() to work with zero or more arguments
  /// without relying on non-standard ##__VA_ARGS__ comma elision.
  struct Proxy {
    const char* Name;
    const char* Sig;
    template <typename... Args> TraceRegion operator()(Args&&... args) {
#ifndef NDEBUG
      unsigned expected = countParams(Sig);
      unsigned actual = sizeof...(Args);
      if (expected != actual) {
        llvm::errs() << "ERROR: INTEROP_TRACE argument count mismatch in '"
                     << Name << "': function has " << expected
                     << " parameter(s) but INTEROP_TRACE received " << actual
                     << " argument(s).\n"
                     << "  Signature: " << Sig << "\n";
        assert(expected == actual &&
               "INTEROP_TRACE argument count does not match function "
               "parameters. Update the INTEROP_TRACE call.");
      }
#endif
      // OUT-mask check runs in any build but only when tracing is active,
      // so the per-call cost off-trace is one load + branch. The diagnostic
      // is unconditional; assert() is a no-op in NDEBUG, so Release reports
      // the drift via stderr without aborting.
      if (TheTraceInfo) {
        if (auto Expected = lookupOutMask(Name)) {
          uint64_t actual_out = computeOutMask<Args...>();
          if (*Expected != actual_out) {
            llvm::errs() << formatOutMaskMismatchMessage(Name, *Expected,
                                                         actual_out);
            assert(
                *Expected == actual_out &&
                "INTEROP_TRACE OUT-arg coverage does not match the .td "
                "OutArg<...> declarations. Wrap or unwrap with INTEROP_OUT.");
          }
        }
      }
      return TraceRegion(Name, std::forward<Args>(args)...);
    }
  };

  /// .td-declared OUT-arg bitmask for \p Name (matches CppName /
  /// __func__), or std::nullopt for non-public-API tracepoints.
  CPPINTEROP_TRACE_API static std::optional<uint64_t>
  lookupOutMask(llvm::StringRef Name);

  /// Pure-function diagnostic for the OUT-mask mismatch. Extracted so a
  /// non-death test can cover the format path -- the assert path itself
  /// runs in a forked child whose coverage data is not merged back.
  static std::string formatOutMaskMismatchMessage(llvm::StringRef Name,
                                                  uint64_t expected,
                                                  uint64_t actual) {
    return llvm::formatv("ERROR: INTEROP_OUT coverage mismatch in '{0}': .td "
                         "OutArg mask {1:X} vs INTEROP_OUT-wrapped args "
                         "{2:X}.\n",
                         Name, expected, actual)
        .str();
  }
};

} // namespace Tracing
} // namespace CppInterOp

#ifdef _MSC_VER
#define INTEROP_FUNC_SIG __FUNCSIG__
#else
#define INTEROP_FUNC_SIG __PRETTY_FUNCTION__
#endif

#define INTEROP_TRACE(...)                                                     \
  CppInterOp::Tracing::TraceRegion _TR =                                       \
      CppInterOp::Tracing::TraceRegion::Proxy{__func__,                        \
                                              INTEROP_FUNC_SIG}(__VA_ARGS__)

#define INTEROP_RETURN(Val) _TR.record(Val)
#define INTEROP_VOID_RETURN() (_TR.recordVoid())

#define INTEROP_OUT(Var) CppInterOp::Tracing::MakeOutParam(Var)

#endif // CPPINTEROP_TRACING_H
