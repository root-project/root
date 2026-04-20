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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace CppInterOp {
namespace Tracing {

class TraceInfo {
  llvm::TimerGroup m_TG;
  llvm::StringMap<std::unique_ptr<llvm::Timer>> m_Timers;
  std::vector<llvm::Timer*> m_TimerStack;

  std::unordered_map<void*, std::string> m_HandleMap;
  unsigned m_VarCount = 0;

  std::vector<std::string> m_Log;
  size_t m_RegionStart = 0; ///< Log index where current region began.
  bool m_InRegion = false;  ///< True between StartTracing/StopTracing.

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

  std::string getOrRegisterHandle(void* p) {
    if (!p)
      return "";
    auto it = m_HandleMap.find(p);
    if (it != m_HandleMap.end())
      return it->second;
    return m_HandleMap[p] = "v" + std::to_string(++m_VarCount);
  }

  std::string lookupHandle(void* p) {
    if (!p)
      return "nullptr";
    auto it = m_HandleMap.find(p);
    return (it != m_HandleMap.end()) ? it->second : "nullptr";
  }

  void appendToLog(const std::string& line) {
    m_Log.push_back(line);
    if (m_InRegion && m_WriteOnStdErr)
      llvm::errs() << line << "\n";
  }
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
  }

  CPPINTEROP_TRACE_API static TraceInfo* TheTraceInfo;
};

/// Activate tracing. Called once during process initialization.
/// After this, TheTraceInfo is non-null and all INTEROP_TRACE calls record.
CPPINTEROP_TRACE_API void InitTracing();

/// Begin recording a traced region. If tracing is not yet active, activates
/// it. Returns the path where StopTracing() will write the reproducer.
/// \param WriteOnStdErr if true, also emit the reproducer to stderr on stop.
inline std::string StartTracing(bool WriteOnStdErr = true) {
  if (!TraceInfo::TheTraceInfo)
    InitTracing();
  return TraceInfo::TheTraceInfo->StartRegion(WriteOnStdErr);
}

/// End the traced region and write the reproducer file containing only the
/// calls made between StartTracing() and this call.
inline void StopTracing(const std::string& Version = "") {
  if (TraceInfo::TheTraceInfo)
    TraceInfo::TheTraceInfo->StopRegion(Version);
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
  /// Callback that registers the container's pointer elements as handles.
  /// Null when the container doesn't hold pointers (e.g. vector<string>).
  std::function<void(TraceInfo&)> RegisterHandles;
};

/// Create an OutParam for any container. Only sets up handle registration
/// when the container's value_type is a pointer.
template <typename Container> OutParam MakeOutParam(const Container& C) {
  OutParam OP;
  using Value = typename Container::value_type;
  if constexpr (std::is_pointer_v<Value>) {
    OP.RegisterHandles = [&C](TraceInfo& TI) {
      for (const auto& Elem : C)
        TI.getOrRegisterHandle(reinterpret_cast<void*>(Elem));
    };
  }
  return OP;
}

/// Internal helper to stringify arguments into a C++ call format.
struct ReproBuffer {
  llvm::SmallString<128> Buffer;
  llvm::raw_svector_ostream OS;

  ReproBuffer() : OS(Buffer) {}

  // Opaque handle pointers — resolved to their registered name.
  void append(void* p) { OS << TraceInfo::TheTraceInfo->lookupHandle(p); }

  // Strings — quoted.
  void append(const char* s) { OS << "\"" << s << "\""; }
  void append(const std::string& s) { OS << "\"" << s << "\""; }

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

  // Containers — not meaningfully printable; emit a placeholder.
  template <typename T> void append(const std::vector<T>&) { OS << "{...}"; }

  // Anything else we haven't accounted for.
  template <typename T> void append(const T&) { OS << "?"; }

  /// Format a comma-separated argument list, skipping OutParam entries.
  template <typename... Args> void format(Args&&... args) {
    bool first = true;
    auto appendOne = [&](auto&& val) {
      if constexpr (!std::is_same_v<std::decay_t<decltype(val)>, OutParam>) {
        if (!first)
          OS << ", ";
        first = false;
        append(std::forward<decltype(val)>(val));
      }
    };
    (appendOne(std::forward<Args>(args)), ...);
  }
};

/// Holds all the data that is only needed when tracing is active.
/// Heap-allocated only when TheTraceInfo != nullptr, so disabled tracing
/// pays zero cost beyond a single pointer + bool on the stack.
struct TraceData {
  const char* Name;
  llvm::SmallString<128> ArgStr;
  void* Result = nullptr;
  bool HasPtrResult = false;
  double StartTime = 0;
  bool Returned = false;
  llvm::SmallVector<std::function<void(TraceInfo&)>, 2> OutCallbacks;
};

class TraceRegion {
  std::unique_ptr<TraceData> m_Data;

  // Capture an OutParam's handle-registration callback; ignore everything else.
  void captureArg(OutParam&& op) {
    if (op.RegisterHandles)
      m_Data->OutCallbacks.push_back(std::move(op.RegisterHandles));
  }
  template <typename T> void captureArg(T&&) {}

public:
  template <typename... Args> TraceRegion(const char* Name, Args&&... args) {
    if (!TraceInfo::TheTraceInfo)
      return;
    m_Data = std::make_unique<TraceData>();
    m_Data->Name = Name;
    (captureArg(std::forward<Args>(args)), ...);
    if constexpr (sizeof...(args) > 0) {
      ReproBuffer RB;
      RB.format(std::forward<Args>(args)...);
      m_Data->ArgStr = RB.Buffer;
    }
    TraceInfo& TI = *TraceInfo::TheTraceInfo;
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
    TraceInfo& TI = *TraceInfo::TheTraceInfo;
    TI.popTimer();

    // Register out-param handles now that the function has filled them.
    for (auto& cb : m_Data->OutCallbacks)
      cb(TI);

    std::string VarPart;
    if (m_Data->Result) {
      bool isNew = TI.lookupHandle(m_Data->Result) == "nullptr";
      std::string HandleName = TI.getOrRegisterHandle(m_Data->Result);
      VarPart =
          llvm::formatv(isNew ? "auto {0} = " : "/*{0}*/ ", HandleName).str();
    } else if (m_Data->HasPtrResult) {
      VarPart = "/*nullptr*/ ";
    }

    std::string Call = llvm::formatv("  {0}Cpp::{1}({2}); // [{3} ns]", VarPart,
                                     m_Data->Name, m_Data->ArgStr, Dur);

    // Store in log for the reproducer file.
    TI.appendToLog(Call);

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
    if constexpr (std::is_pointer_v<T>) {
      m_Data->HasPtrResult = true;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      m_Data->Result = const_cast<void*>(static_cast<const void*>(val));
    } else if constexpr (std::is_null_pointer_v<T>) {
      m_Data->HasPtrResult = true;
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
      return TraceRegion(Name, std::forward<Args>(args)...);
    }
  };
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
