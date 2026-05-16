//===--- Tracing.cpp - Tracing implementation -------------------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tracing.h"

#include "CppInterOp/CppInterOp.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <system_error>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

// libclangCppInterOp's own trace-hook slot storage; Dispatch.h
// consumers carry their per-DSO copies, populated by LoadDispatchAPI.
namespace CppInternal {
namespace DispatchRaw {
CPPINTEROP_API void (*CppInterOpTraceJitCallInvokeImpl)(
    const CppImpl::JitCall*, void*, void**, std::size_t, void*) = nullptr;
CPPINTEROP_API void (*CppInterOpTraceJitCallInvokeDestructorImpl)(
    const CppImpl::JitCall*, void*, unsigned long, int) = nullptr;
CPPINTEROP_API void (*CppInterOpTraceJitCallInvokeReturnImpl)(
    const CppImpl::JitCall*, void*) = nullptr;
} // namespace DispatchRaw
} // namespace CppInternal

namespace CppInterOp {
namespace Tracing {

CPPINTEROP_TRACE_API TraceInfo* TheTraceInfo = nullptr;

void InitTracing() {
  assert(!TheTraceInfo);
  static llvm::ManagedStatic<TraceInfo> TI;
  TheTraceInfo = &*TI;
  // Wire libclangCppInterOp's own slots; Dispatch.h consumers do
  // the equivalent via the X-macro in LoadDispatchAPI.
  ::CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeImpl =
      &CppImpl::CppInterOpTraceJitCallInvokeImpl;
  ::CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeDestructorImpl =
      &CppImpl::CppInterOpTraceJitCallInvokeDestructorImpl;
  ::CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeReturnImpl =
      &CppImpl::CppInterOpTraceJitCallInvokeReturnImpl;
}

std::optional<uint64_t> TraceRegion::lookupOutMask(llvm::StringRef Name) {
  // Built once from CppInterOpAPI.inc via the OUT-only X-macro;
  // CPPINTEROP_API_FUNC stays a no-op.
  static const llvm::StringMap<uint64_t> Map = {
#define CPPINTEROP_API_FUNC(DN, CN, Ret, DA, CA, RT)
#define CPPINTEROP_API_OUT(CN, OutMask) {#CN, OutMask},
#include "CppInterOp/CppInterOpAPI.inc"
  };
  auto It = Map.find(Name);
  if (It == Map.end())
    return std::nullopt;
  return It->second;
}

/// RAII: hold m_Dumping while the reproducer is being emitted.
namespace {
class DumpScope {
  TraceInfo& TI;

public:
  explicit DumpScope(TraceInfo& T) : TI(T) { TI.setDumping(true); }
  ~DumpScope() { TI.setDumping(false); }
};
} // namespace

/// Helper: emit version info as comment lines.
static void WriteVersionComment(llvm::raw_ostream& OS,
                                const std::string& Version) {
  llvm::StringRef Ver(Version);
  while (!Ver.empty()) {
    auto [Line, Rest] = Ver.split('\n');
    if (!Line.empty())
      OS << "// " << Line << "\n";
    Ver = Rest;
  }
}

/// dladdr-based path to libclangCppInterOp, used as the dispatch-mode
/// fallback. Empty for statically-linked images (where dladdr returns
/// the executable, not a real shared object) so the reproducer falls
/// through to CPPINTEROP_LIBRARY_PATH.
static std::string GetCppInterOpLibPath() {
#if defined(__linux__) || defined(__APPLE__)
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&CppImpl::GetVersion), &info) &&
      info.dli_fname) {
    llvm::StringRef Name(info.dli_fname);
    if (Name.contains(".so") || Name.ends_with(".dylib") ||
        Name.ends_with(".dll"))
      return info.dli_fname;
  }
#endif
  return {};
}

/// Stream Cpp::GetBuildInfo() into the prologue as `//` comments.
static void WriteBuildContext(llvm::raw_ostream& OS) {
  OS << "// Build context (CppInterOp configure-time snapshot):\n";
  // Bind to a local: StringRef of the GetBuildInfo() rvalue would
  // dangle past the next `;`.
  std::string Snapshot = CppImpl::GetBuildInfo();
  llvm::StringRef Info(Snapshot);
  while (!Info.empty()) {
    auto [Line, Rest] = Info.split('\n');
    if (!Line.empty())
      OS << "//" << Line << "\n";
    Info = Rest;
  }
}

/// Emit a prologue compilable in two modes selected by -D:
///   default                       <CppInterOp/CppInterOp.h>, static-link
///   -DCPPINTEROP_USE_DISPATCH     <CppInterOp/Dispatch.h>, dlopen-based
/// The dispatch path reads CPPINTEROP_LIBRARY_PATH and falls back to
/// the dladdr-captured libclangCppInterOp path (when available).
static void WriteReproducerPrologue(llvm::raw_ostream& OS,
                                    llvm::StringRef Title,
                                    const std::string& Version,
                                    llvm::StringRef Footnote) {
  OS << "// CppInterOp " << Title << "\n";
  WriteVersionComment(OS, Version);
  if (!Footnote.empty())
    OS << "// " << Footnote << "\n";
  OS << "//\n";
  OS << "// Replay scope: re-runs recorded CppInterOp API calls. JIT\n";
  OS << "// wrappers (Cpp::JitCall::Invoke) are not replayed -- their\n";
  OS << "// pointer args reference the original process's memory. The\n";
  OS << "// trailing `// JitCall::Invoke ...` comment, if present,\n";
  OS << "// names the call active at the crash site.\n";
  OS << "//\n";
  WriteBuildContext(OS);
  OS << "//\n";
  OS << "// Build (default, static link):\n";
  OS << "//   c++ -std=c++17 -I<CppInterOp-include-dir>"
        " reproducer.cpp -lclangCppInterOp -o reproducer\n";
  OS << "// Build (dlopen via Dispatch.h):\n";
  OS << "//   c++ -std=c++17 -DCPPINTEROP_USE_DISPATCH"
        " -I<CppInterOp-include-dir>"
        " reproducer.cpp -ldl -o reproducer\n";
  std::string LibPath = GetCppInterOpLibPath();
  OS << "// Run: ./reproducer";
  if (LibPath.empty())
    OS << "  (dispatch build: set "
          "CPPINTEROP_LIBRARY_PATH=<libclangCppInterOp.so>)";
  else
    OS << "  (dispatch build: CPPINTEROP_LIBRARY_PATH overrides the captured "
          "path)";
  OS << "\n\n";

  OS << "#ifdef CPPINTEROP_USE_DISPATCH\n";
  OS << "#include <CppInterOp/Dispatch.h>\n\n";
  // X-macro defines one DispatchRaw slot per public API entry.
  OS << "using namespace CppImpl;\n";
  OS << "#define CPPINTEROP_API_FUNC(DN, CN, Ret, DeclArgs, CallArgs, "
        "RawTypes) \\\n"
        "  Ret(*CppInternal::DispatchRaw::DN) RawTypes = nullptr;\n"
        "#include \"CppInterOp/CppInterOpAPI.inc\"\n\n";
  // Lazy: lets a cling-driven harness call reproducer() before main().
  OS << "static void initCppInterOpReproducer() {\n";
  OS << "  static bool inited = false;\n";
  OS << "  if (inited) return;\n";
  OS << "  const char* path = std::getenv(\"CPPINTEROP_LIBRARY_PATH\");\n";
  if (!LibPath.empty())
    OS << "  if (!path) path = R\"PATH(" << LibPath << ")PATH\";\n";
  OS << "  Cpp::LoadDispatchAPI(path);\n";
  OS << "  inited = true;\n";
  OS << "}\n";
  OS << "#else\n";
  OS << "#include <CppInterOp/CppInterOp.h>\n";
  // Static-link path: no-op so reproducer()'s init call is harmless.
  OS << "static void initCppInterOpReproducer() {}\n";
  OS << "#endif\n\n";
}

std::string TraceInfo::writeToFile(const std::string& Version) {
  llvm::SmallString<128> TmpDir;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, TmpDir);
  llvm::SmallString<128> Path;
  llvm::sys::path::append(Path, TmpDir, "cppinterop-reproducer-%%%%%%.cpp");

  int FD;
  std::error_code EC = llvm::sys::fs::createUniqueFile(Path, FD, Path);
  if (EC)
    return "";

  llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
  DumpScope Guard(*this);

  std::string Ver = Version.empty() ? CppImpl::GetVersion() : Version;
  WriteReproducerPrologue(OS, "crash reproducer", Ver,
                          "Generated automatically — re-run to reproduce.");
  OS << "void reproducer() {\n";
  OS << "  initCppInterOpReproducer();\n";
  for (const auto& Line : m_Log)
    OS << Line << "\n";
  OS << "}\n\n";
  OS << "int main() { reproducer(); return 0; }\n";
  OS.flush();
  return std::string(Path);
}

std::string TraceInfo::StartRegion(bool WriteOnStdErr) {
  m_RegionStart = m_Log.size();
  m_InRegion = true;
  m_WriteOnStdErr = WriteOnStdErr;

  // When streaming to stderr, skip creating a file.
  if (WriteOnStdErr) {
    m_RegionPath.clear();
    return "";
  }

  llvm::SmallString<128> TmpDir;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, TmpDir);
  llvm::SmallString<128> Path;
  llvm::sys::path::append(Path, TmpDir, "cppinterop-reproducer-%%%%%%.cpp");
  int FD;
  std::error_code EC = llvm::sys::fs::createUniqueFile(Path, FD, Path);
  if (EC)
    return "";
  llvm::sys::Process::SafelyCloseFileDescriptor(FD);
  m_RegionPath = std::string(Path);
  return m_RegionPath;
}

void TraceInfo::StopRegion(const std::string& Version) {
  if (!m_InRegion)
    return;
  m_InRegion = false;

  // When streaming to stderr, there is no file to write.
  if (m_WriteOnStdErr)
    return;

  std::error_code EC;
  llvm::raw_fd_ostream OS(m_RegionPath, EC);
  if (EC)
    return;

  DumpScope Guard(*this);
  std::string Ver = Version.empty() ? CppImpl::GetVersion() : Version;
  WriteReproducerPrologue(OS, "trace region", Ver, "Generated automatically.");
  OS << "void reproducer() {\n";
  OS << "  initCppInterOpReproducer();\n";
  for (size_t i = m_RegionStart; i < m_Log.size(); ++i)
    OS << m_Log[i] << "\n";
  OS << "}\n\n";
  OS << "int main() { reproducer(); return 0; }\n";
  OS.flush();
}

} // namespace Tracing
} // namespace CppInterOp
