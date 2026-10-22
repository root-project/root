//--------------------------------------------------------------------*- C++ -*-
// CppInterOp Compatibility
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------
#ifndef CPPINTEROP_COMPATIBILITY_H
#define CPPINTEROP_COMPATIBILITY_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#if CLANG_VERSION_MAJOR < 21
#include "clang/Basic/Cuda.h"
#else
#include "clang/Basic/OffloadArch.h"
#endif
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#if CLANG_VERSION_MAJOR < 22
#include "clang/Driver/Options.h"
#else
#include "clang/Options/Options.h"
#endif
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Triple.h"

#include "CppInterOp/Box.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#if CLANG_VERSION_MAJOR < 22
#define clang_driver_options clang::driver::options
#else
#define clang_driver_options clang::options
#endif

#if CLANG_VERSION_MAJOR < 22
#define Suppress_Elab SuppressElaboration
#else
#define Suppress_Elab FullyQualifiedName
#endif

#ifdef _MSC_VER
#define dup _dup
#define dup2 _dup2
#define close _close
#define fileno _fileno
#endif

static inline char* GetEnv(const char* Var_Name) {
#ifdef _MSC_VER
  char* Env = nullptr;
  size_t sz = 0;
  getenv_s(&sz, Env, sz, Var_Name);
  return Env;
#else
  return getenv(Var_Name);
#endif
}

#if CLANG_VERSION_MAJOR < 21
#define Print_Canonical_Types PrintCanonicalTypes
#else
#define Print_Canonical_Types PrintAsCanonical
#endif

#if CLANG_VERSION_MAJOR < 21
#define clang_LookupResult_Found clang::LookupResult::Found
#define clang_LookupResult_Not_Found clang::LookupResult::NotFound
#define clang_LookupResult_Found_Overloaded clang::LookupResult::FoundOverloaded
#else
#define clang_LookupResult_Found clang::LookupResultKind::Found
#define clang_LookupResult_Not_Found clang::LookupResultKind::NotFound
#define clang_LookupResult_Found_Overloaded                                    \
  clang::LookupResultKind::FoundOverloaded
#endif

#define STRINGIFY(s) STRINGIFY_X(s)
#define STRINGIFY_X(...) #__VA_ARGS__

#include "clang/Interpreter/CodeCompletion.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

// std::regex breaks pytorch's jit: pytorch/pytorch#49460
#include "llvm/Support/Regex.h"

#ifdef CPPINTEROP_USE_CLING

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"

#include "cling/Utils/AST.h"

#include <regex>
#include <vector>

namespace CppInternal {
namespace utils = cling::utils;
}

namespace compat {

using Interpreter = cling::Interpreter;

class SynthesizingCodeRAII : public Interpreter::PushTransactionRAII {
public:
  SynthesizingCodeRAII(Interpreter* i) : Interpreter::PushTransactionRAII(i) {}
};

inline void maybeMangleDeclName(const clang::GlobalDecl& GD,
                                std::string& mangledName) {
  cling::utils::Analyze::maybeMangleDeclName(GD, mangledName);
}

/// The getExecutionEngine() interface was been added for Cling based on LLVM
/// >=18. For previous versions, the LLJIT was obtained by computing the object
/// offsets in the cling::Interpreter instance(IncrementalExecutor):
/// sizeof (m_Opts) + sizeof(m_LLVMContext). The IncrementalJIT and JIT itself
/// have an offset of 0 as the first datamember.
inline llvm::orc::LLJIT* getExecutionEngine(cling::Interpreter& I) {
  return I.getExecutionEngine();
}

inline llvm::Expected<llvm::JITTargetAddress>
getSymbolAddress(cling::Interpreter& I, llvm::StringRef IRName) {
  if (void* Addr = I.getAddressOfGlobal(IRName))
    return (llvm::JITTargetAddress)Addr;

  llvm::orc::LLJIT& Jit = *compat::getExecutionEngine(I);
  llvm::orc::SymbolNameVector Names;
  llvm::orc::ExecutionSession& ES = Jit.getExecutionSession();
  Names.push_back(ES.intern(IRName));
  return llvm::make_error<llvm::orc::SymbolsNotFound>(ES.getSymbolStringPool(),
                                                      std::move(Names));
}

inline void codeComplete(std::vector<std::string>& Results,
                         const cling::Interpreter& I, const char* code,
                         unsigned complete_line = 1U,
                         unsigned complete_column = 1U) {
  std::vector<std::string> results;
  size_t column = complete_column;
  I.codeComplete(code, column, results);
  std::string error;
  llvm::Error Err = llvm::Error::success();
  // Regex patterns
  llvm::Regex removeDefinition("\\[\\#.*\\#\\]");
  llvm::Regex removeVariableName("(\\ |\\*)+(\\w+)(\\#\\>)");
  llvm::Regex removeTrailingSpace("\\ *(\\#\\>)");
  llvm::Regex removeTags("\\<\\#([^#>]*)\\#\\>");

  // append cleaned results
  for (auto& r : results) {
    // remove the definition at the beginning (e.g., [#int#])
    r = removeDefinition.sub("", r, &error);
    if (!error.empty()) {
      Err = llvm::make_error<llvm::StringError>(error,
                                                llvm::inconvertibleErrorCode());
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Invalid substitution in CodeComplete");
      return;
    }
    // remove the variable name in <#type name#>
    r = removeVariableName.sub("$1$3", r, &error);
    if (!error.empty()) {
      Err = llvm::make_error<llvm::StringError>(error,
                                                llvm::inconvertibleErrorCode());
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Invalid substitution in CodeComplete");
      return;
    }
    // remove unnecessary space at the end of <#type   #>
    r = removeTrailingSpace.sub("$1", r, &error);
    if (!error.empty()) {
      Err = llvm::make_error<llvm::StringError>(error,
                                                llvm::inconvertibleErrorCode());
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Invalid substitution in CodeComplete");
      return;
    }
    // remove <# #> to keep only the type
    r = removeTags.sub("$1", r, &error);
    if (!error.empty()) {
      Err = llvm::make_error<llvm::StringError>(error,
                                                llvm::inconvertibleErrorCode());
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Invalid substitution in CodeComplete");
      return;
    }

    if (r.find(code) == 0)
      Results.push_back(r);
  }
  llvm::consumeError(std::move(Err));
}

} // namespace compat

#endif // CPPINTEROP_USE_CLING

#ifndef CPPINTEROP_USE_CLING

#include "DynamicLibraryManager.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"

#if LLVM_VERSION_MAJOR > 21
#include "clang/Basic/Version.h"
#include "clang/Interpreter/IncrementalExecutor.h"

#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#endif

#if LLVM_VERSION_MAJOR > 21 && !defined(_WIN32)
#include <unistd.h>
#endif

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <algorithm>

namespace compat {

/// Detect the CUDA installation path using clang::Driver
/// \param args user-provided interpreter arguments (may contain --cuda-path).
/// \param[out] CudaPath the detected CUDA installation path.
/// \returns true on success, false if not found.
inline bool detectCudaInstallPath(const std::vector<const char*>& args,
                                  std::string& CudaPath) {
  // minimal driver that runs CudaInstallationDetector internally
  std::string TT = llvm::sys::getProcessTriple();
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  auto* DiagsBuffer = new clang::TextDiagnosticBuffer;
#if CLANG_VERSION_MAJOR < 21
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new clang::DiagnosticOptions());
  clang::DiagnosticsEngine Diags(DiagID, DiagOpts, DiagsBuffer);
#else
  clang::DiagnosticOptions DiagOpts;
  clang::DiagnosticsEngine Diags(DiagID, DiagOpts, DiagsBuffer);
#endif

  clang::driver::Driver D("clang", TT, Diags);
  D.setCheckInputsExist(false);

  // construct args: clang -x cuda -c <<< inputs >>> [args]
  llvm::SmallVector<const char*, 16> Argv;
  Argv.push_back("clang");
  Argv.push_back("-xcuda");
  Argv.push_back("-c");
  Argv.push_back("<<< inputs >>>");
  for (const auto* arg : args)
    Argv.push_back(arg);

  // build a compilation object, which runs the driver's CUDA installation
  // detection logic and stores the paths
  std::unique_ptr<clang::driver::Compilation> C(D.BuildCompilation(Argv));
  if (!C)
    return false;

  // --cuda-path was explicitly provided in user args
  if (auto* A =
          C->getArgs().getLastArg(clang_driver_options::OPT_cuda_path_EQ)) {
    std::string Candidate = A->getValue();
    if (llvm::sys::fs::is_directory(Candidate + "/include")) {
      CudaPath = Candidate;
      return true;
    }
  }

  // fallback: clang tries to auto-detect the install, CudaInstallationDetector
  // stores the path internally but doesn't expose it, so we look for
  // "-internal-isystem <cuda-path>/include" that the driver adds for CUDA
  // headers.
  for (const auto& Job : C->getJobs()) {
    if (const auto* Cmd = llvm::dyn_cast<clang::driver::Command>(&Job)) {
      const auto& Args = Cmd->getArguments();
      for (size_t i = 0; i + 1 < Args.size(); ++i) {
        if (llvm::StringRef(Args[i]) == "-internal-isystem") {
          llvm::StringRef IncDir(Args[i + 1]);
          if (IncDir.ends_with("/include") &&
              llvm::sys::fs::exists(IncDir.str() + "/cuda.h")) {
            CudaPath = IncDir.drop_back(strlen("/include")).str();
            return true;
          }
        }
      }
    }
  }
  return false;
}

/// Detect GPU architecture via the CUDA Driver API, tweaked from clang's
/// nvptx-arch tool (NVPTXArch.cpp) \param[out] Arch Set to "sm_XX" on success,
/// or clang's default fallback. \returns true on success, false on error (no
/// CUDA driver available).
inline bool detectNVPTXArch(std::string& Arch) {
  std::string Err;
  // FIXME: Use ToolChain::getSystemGPUArchs() from a minimal driver compilation
  // instead, and unify this function with detectCudaInstallPath. Ideally we
  // should rely on the offload-arch/nvptx-arch tool in clang, but there is no
  // public API or library to link against.
  auto Lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
#ifdef _WIN32
      "nvcuda.dll",
#else
      "libcuda.so.1",
#endif
      &Err);
  if (!Lib.isValid())
    return false;

  using cuInit_t = int (*)(unsigned);
  using cuDeviceGet_t = int (*)(uint32_t*, int);
  using cuDeviceGetAttribute_t = int (*)(int*, int, uint32_t);

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto cuInit = reinterpret_cast<cuInit_t>(Lib.getAddressOfSymbol("cuInit"));
  auto cuDeviceGet =
      reinterpret_cast<cuDeviceGet_t>(Lib.getAddressOfSymbol("cuDeviceGet"));
  auto cuDeviceGetAttribute = reinterpret_cast<cuDeviceGetAttribute_t>(
      Lib.getAddressOfSymbol("cuDeviceGetAttribute"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  if (!cuInit || !cuDeviceGet || !cuDeviceGetAttribute)
    return false;

  uint32_t dev;
  int maj, min;
  if (cuInit(0) || cuDeviceGet(&dev, 0) ||
      cuDeviceGetAttribute(&maj, /*MAJOR*/ 75, dev) ||
      cuDeviceGetAttribute(&min, /*MINOR*/ 76, dev)) {
    Arch = clang::OffloadArchToString(clang::OffloadArch::CudaDefault);
    return true;
  }
  Arch = "sm_" + std::to_string(maj) + std::to_string(min);
  return true;
}

#if LLVM_VERSION_MAJOR > 21
/// Directory containing libclangCppInterOp itself, derived via
/// `dladdr` of an in-library function pointer. Returns empty when the
/// platform has no self-DSO discovery (Windows -- a `GetModuleHandleEx`
/// port can be added when Windows OOP support arrives).
inline std::string findOwnLibraryDir() {
#if !defined(_WIN32)
  Dl_info info{};
  if (!dladdr(reinterpret_cast<const void*>(&findOwnLibraryDir), &info) ||
      !info.dli_fname || !*info.dli_fname)
    return {};
  llvm::SmallString<256> P(info.dli_fname);
  llvm::sys::path::remove_filename(P);
  return std::string(P.str());
#else
  return {};
#endif
}

/// Wire CppInterOp's bundled OOP runtime parts into `B`. Probes a
/// layered list of candidate directories, in priority order:
///   1. `$CPPINTEROP_RUNTIME_DIR` -- sysadmin override.
///   2. `<dir of libclangCppInterOp>/cppinterop-rt` -- relocatable, follows
///      the .so wherever a package manager moved it.
///   3. `CPPINTEROP_RUNTIME_BUILD_DIR` -- in-tree test runs.
///   4. `CPPINTEROP_RUNTIME_INSTALL_DIR` -- baked install path; last
///      resort when self-DSO discovery isn't available (e.g. static
///      link of CppInterOp into a host binary).
/// `UpdateOrcRuntimePathCB` is replaced with a no-op so the upstream
/// resource-dir prefix check inside
/// `IncrementalExecutorBuilder::UpdateOrcRuntimePath`
/// (`clang/lib/Interpreter/IncrementalExecutor.cpp`, the
/// `consume_front(parent_path(D.Dir))` guard) doesn't run -- our
/// runtime lives outside the host's clang resource tree.
inline bool configureBundledOOPRuntime(clang::IncrementalExecutorBuilder& B) {
  llvm::SmallVector<std::string, 4> Candidates;
  if (const char* Env = std::getenv("CPPINTEROP_RUNTIME_DIR"))
    Candidates.emplace_back(Env);
  if (std::string OwnDir = findOwnLibraryDir(); !OwnDir.empty()) {
    llvm::SmallString<256> P(OwnDir);
    llvm::sys::path::append(P, "cppinterop-rt");
    Candidates.emplace_back(P.str());
  }
#if defined(CPPINTEROP_RUNTIME_BUILD_DIR)
  Candidates.emplace_back(CPPINTEROP_RUNTIME_BUILD_DIR);
#endif
#if defined(CPPINTEROP_RUNTIME_INSTALL_DIR)
  Candidates.emplace_back(CPPINTEROP_RUNTIME_INSTALL_DIR);
#endif
  for (const std::string& Dir : Candidates) {
    llvm::SmallString<256> OrcRT(Dir);
    llvm::sys::path::append(OrcRT, "liborc_rt.a");
    llvm::SmallString<256> Exec(Dir);
    llvm::sys::path::append(Exec, "llvm-jitlink-executor");
    if (!llvm::sys::fs::exists(OrcRT) || !llvm::sys::fs::exists(Exec))
      continue;
    B.OrcRuntimePath = std::string(OrcRT.str());
    B.OOPExecutor = std::string(Exec.str());
    B.UpdateOrcRuntimePathCB = [](const clang::driver::Compilation&) {
      return llvm::Error::success();
    };
    return true;
  }
  return false;
}
#endif // LLVM_VERSION_MAJOR > 21

inline std::unique_ptr<clang::Interpreter>
createClangInterpreter(std::vector<const char*>& args, int stdin_fd = -1,
                       int stdout_fd = -1, int stderr_fd = -1) {
  bool CudaEnabled = false;
  std::string OffloadArch;
  std::string CudaPath;
  std::vector<const char*> CompilerArgs;
  for (const auto* arg : args) {
    llvm::StringRef A(arg);
    llvm::StringRef Stripped = A.trim().ltrim('-');
    if (Stripped == "cuda") {
      CudaEnabled = true;
    } else if (A.starts_with("--offload-arch=")) {
      OffloadArch = A.substr(strlen("--offload-arch="));
    } else if (A.starts_with("--cuda-path=")) {
      CudaPath = A.substr(strlen("--cuda-path="));
    } else {
      CompilerArgs.push_back(arg);
    }
  }
#ifdef __APPLE__
  CudaEnabled = false;
#endif

  clang::IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs(CompilerArgs);

#if LLVM_VERSION_MAJOR > 21 && !defined(_WIN32)
  bool outOfProcess = false;
  const bool oopRequested =
      std::any_of(args.begin(), args.end(), [](const char* arg) {
        return llvm::StringRef(arg).trim() == "--use-oop-jit";
      });
  // The IncrementalExecutorBuilder must outlive the IncrementalCompiler
  // it gets attached to, so it's a unique_ptr at function scope.
  std::unique_ptr<clang::IncrementalExecutorBuilder> OutOfProcessConfig;
  if (oopRequested) {
    OutOfProcessConfig = std::make_unique<clang::IncrementalExecutorBuilder>();
    OutOfProcessConfig->IsOutOfProcess = true;
    if (configureBundledOOPRuntime(*OutOfProcessConfig)) {
      outOfProcess = true;
      CB.SetDriverCompilationCallback(
          OutOfProcessConfig->UpdateOrcRuntimePathCB);
    } else {
      llvm::errs()
          << "[CreateClangInterpreter]: --use-oop-jit requested but the "
             "bundled OOP runtime "
             "(<libdir>/cppinterop-rt/{liborc_rt.a,llvm-jitlink-executor}) "
             "is missing from CppInterOp's build/install tree. Falling "
             "back to in-process JIT.\n";
      OutOfProcessConfig.reset();
    }
  }
#endif

  std::unique_ptr<clang::CompilerInstance> DeviceCI;
  if (CudaEnabled) {
    if (OffloadArch.empty())
      detectNVPTXArch(OffloadArch);

    if (CudaPath.empty())
      detectCudaInstallPath(CompilerArgs, CudaPath);

    CB.SetOffloadArch(OffloadArch);
    if (!CudaPath.empty())
      CB.SetCudaSDK(CudaPath);
    auto devOrErr = CB.CreateCudaDevice();
    if (!devOrErr) {
      llvm::logAllUnhandledErrors(devOrErr.takeError(), llvm::errs(),
                                  "Failed to create device compiler:");
      return nullptr;
    }
    DeviceCI = std::move(*devOrErr);
  }
  auto ciOrErr = CudaEnabled ? CB.CreateCudaHost() : CB.CreateCpp();
  if (!ciOrErr) {
    llvm::logAllUnhandledErrors(ciOrErr.takeError(), llvm::errs(),
                                "Failed to build Incremental compiler:");
    return nullptr;
  }
  (*ciOrErr)->LoadRequestedPlugins();
  if (CudaEnabled)
    DeviceCI->LoadRequestedPlugins();

#if LLVM_VERSION_MAJOR > 21 && !defined(_WIN32)
  if (outOfProcess) {
    // OrcRuntimePath and OOPExecutor were populated by
    // configureBundledOOPRuntime() above; UpdateOrcRuntimePathCB was
    // replaced with a no-op there too, so the upstream auto-discovery
    // safety check doesn't run.
    OutOfProcessConfig->UseSharedMemory = false;
    OutOfProcessConfig->SlabAllocateSize = 0;
    OutOfProcessConfig->CustomizeFork = [stdin_fd, stdout_fd, stderr_fd]() {
      dup2(stdin_fd, STDIN_FILENO);
      dup2(stdout_fd, STDOUT_FILENO);
      dup2(stderr_fd, STDERR_FILENO);
      setvbuf(fdopen(stdout_fd, "w+"), nullptr, _IONBF, 0);
      setvbuf(fdopen(stderr_fd, "w+"), nullptr, _IONBF, 0);
    };
  }
  auto innerOrErr =
      CudaEnabled ? clang::Interpreter::createWithCUDA(std::move(*ciOrErr),
                                                       std::move(DeviceCI))
                  : clang::Interpreter::create(
                        std::move(*ciOrErr),
                        outOfProcess ? std::move(OutOfProcessConfig) : nullptr);
#else
  auto innerOrErr =
      CudaEnabled ? clang::Interpreter::createWithCUDA(std::move(*ciOrErr),
                                                       std::move(DeviceCI))
                  : clang::Interpreter::create(std::move(*ciOrErr));
#endif
  if (!innerOrErr) {
    llvm::logAllUnhandledErrors(innerOrErr.takeError(), llvm::errs(),
                                "Failed to build Interpreter:");
    return nullptr;
  }
  if (CudaEnabled) {
    if (auto Err = (*innerOrErr)->LoadDynamicLibrary("libcudart.so")) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Failed load libcudart.so runtime:");
      return nullptr;
    }
  }

  return std::move(*innerOrErr);
}

inline void maybeMangleDeclName(const clang::GlobalDecl& GD,
                                std::string& mangledName) {
  // copied and adapted from CodeGen::CodeGenModule::getMangledName

  clang::NamedDecl* D =
      llvm::cast<clang::NamedDecl>(const_cast<clang::Decl*>(GD.getDecl()));
  std::unique_ptr<clang::MangleContext> mangleCtx;
  mangleCtx.reset(D->getASTContext().createMangleContext());
  if (!mangleCtx->shouldMangleDeclName(D)) {
    clang::IdentifierInfo* II = D->getIdentifier();
    assert(II && "Attempt to mangle unnamed decl.");
    mangledName = II->getName().str();
    return;
  }

  llvm::raw_string_ostream RawStr(mangledName);

#if defined(_WIN32)
  // MicrosoftMangle.cpp:954 calls llvm_unreachable when mangling Dtor_Comdat
  if (llvm::isa<clang::CXXDestructorDecl>(GD.getDecl()) &&
      GD.getDtorType() == clang::Dtor_Comdat) {
    if (const clang::IdentifierInfo* II = D->getIdentifier())
      RawStr << II->getName();
  } else
#endif
    mangleCtx->mangleName(GD, RawStr);
  RawStr.flush();
}

#ifndef _WIN32
// ===========================================================================
// Bind weak globals the process already defines (in-process JIT).
//
// WORKAROUND: this compensates for a deficiency in clang's in-process JIT
// (clang-repl / ORC in clang::Interpreter), which does not demote weak
// definitions the surrounding process already exports and so ends up with a
// second copy. The demotion policy belongs in clang's Interpreter, where it
// covers every embedder rather than only CppInterOp; that is tracked upstream
// by llvm/llvm-project#211786. This pass and its guarded call sites are a
// stopgap and should be deleted once the upstream fix lands and the minimum
// supported clang carries it -- at which point the toggle below becomes a
// `CLANG_VERSION_MAJOR` guard that compiles the workaround out, and then it
// is removed entirely.
//
// A singleton *defined* in a header -- a function-local static in an inline
// function (Meyers) or a C++17 inline static data member -- compiles to a
// weak/linkonce_odr global variable. When jitted code includes such a header,
// the ORC JIT materializes its own copy: JITDylib definitions win over the
// process-symbol generator, which is only consulted for symbols the JITDylib
// lacks. The result is two instances of one singleton (and a double
// destruction at teardown) where a dynamic linker would have unified them.
//
// Mitigation: before a module reaches the JIT, demote every mutable weak
// global-variable definition whose symbol the dynamic linker already
// resolves to an external declaration; the JIT then binds the process copy.
// A dynamically initialized static's guard variable (_ZGV<...>) moves with
// its data, and only when the process exports both: sharing the data but not
// the guard would re-run initialization on the process copy, sharing the
// guard but not the data would leave the jitted copy uninitialized.
//
// Only mutable variables are demoted: duplicated constants are harmless
// under ODR (and folding them keeps jitted code fast); duplicated mutable
// state is the singleton bug. thread_locals are excluded (the JIT's
// emulated-TLS storage regime is separate), and functions are never demoted
// -- jitted code may legitimately carry its own copies of inline functions.
// ELF and in-process only (the probe is the process's own dlsym).
// ===========================================================================
//
// Single toggle for the whole workaround (definition + call sites).
// TODO: remove once ORC uniques globals across the process (ETA LLVM 25).
#if LLVM_VERSION_MAJOR >= 25
#error "ORC uniques globals now: remove bindProcessWeakGlobals"
#endif
#define CPPINTEROP_WORKAROUND_BIND_PROCESS_WEAK_GLOBALS 1
#if CPPINTEROP_WORKAROUND_BIND_PROCESS_WEAK_GLOBALS
inline bool bindProcessWeakGlobals(llvm::Module& M) {
  if (!llvm::Triple(M.getTargetTriple()).isOSBinFormatELF())
    return false;

  auto ProcessHas = [](llvm::StringRef Name) {
    return ::dlsym(RTLD_DEFAULT, Name.str().c_str()) != nullptr;
  };

  llvm::SmallVector<llvm::GlobalVariable*, 8> Demoted;
  for (llvm::GlobalVariable& GV : M.globals()) {
    if (GV.isDeclaration() || !GV.isWeakForLinker() || GV.isThreadLocal() ||
        GV.isConstant())
      continue;

    llvm::StringRef Name = GV.getName();
    // Guards are only ever demoted together with their variable, below.
    if (Name.starts_with("_ZGV"))
      continue;
    if (!ProcessHas(Name))
      continue;

    llvm::GlobalVariable* Guard = nullptr;
    if (Name.starts_with("_Z")) {
      llvm::GlobalVariable* G =
          M.getNamedGlobal(("_ZGV" + Name.drop_front(2)).str());
      if (G && !G->isDeclaration()) {
        if (!ProcessHas(G->getName()))
          continue;
        Guard = G;
      }
    }

    Demoted.push_back(&GV);
    if (Guard)
      Demoted.push_back(Guard);
  }
  if (Demoted.empty())
    return false;

  for (llvm::GlobalVariable* GV : Demoted) {
    GV->setInitializer(nullptr);
    GV->setLinkage(llvm::GlobalValue::ExternalLinkage);
    GV->setComdat(nullptr);
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
    GV->setDSOLocal(false);
  }

  // Declarations may not appear in the used lists; drop demoted entries.
  for (const char* ListName : {"llvm.used", "llvm.compiler.used"}) {
    llvm::GlobalVariable* Used = M.getNamedGlobal(ListName);
    if (!Used || !Used->hasInitializer())
      continue;
    auto* Init = llvm::cast<llvm::ConstantArray>(Used->getInitializer());
    llvm::SmallVector<llvm::Constant*, 8> Kept;
    for (llvm::Value* Op : Init->operand_values()) {
      auto* C = llvm::cast<llvm::Constant>(Op);
      if (!llvm::is_contained(Demoted, C->stripPointerCasts()))
        Kept.push_back(C);
    }
    if (Kept.size() == Init->getNumOperands())
      continue;
    Used->eraseFromParent();
    if (!Kept.empty()) {
      auto* ATy = llvm::ArrayType::get(Kept.front()->getType(), Kept.size());
      auto* NewUsed = new llvm::GlobalVariable(
          M, ATy, /*isConstant=*/false, llvm::GlobalValue::AppendingLinkage,
          llvm::ConstantArray::get(ATy, Kept), ListName);
      NewUsed->setSection("llvm.metadata");
    }
  }
  return true;
}
#endif // CPPINTEROP_WORKAROUND_BIND_PROCESS_WEAK_GLOBALS
#endif // !_WIN32

// ===========================================================================
// Workaround for an LLVM ORC emulated-TLS discard crash (in-process JIT).
//
// Fixed upstream by llvm/llvm-project#208413 (reland of #207161, which was
// reverted in #207775 over a Darwin test failure): IRMaterializationUnit's ctor
// registered a thread_local's emulated-TLS companion (`__emutls_t.<var>` /
// `__emutls_v.<var>`) in SymbolFlags but not SymbolToDefinition, so discarding
// a *duplicated* weak thread_local dereferenced end() -- an assertion in
// +Asserts builds, heap corruption otherwise. The trigger is two incremental
// modules that each define the same weak/linkonce_odr thread_local (an inline
// function or template static odr-used from two PTUs, as MakeFunctionCallable
// and repeated process() do).
//
// Mitigation: before a module reaches the JIT, demote every *duplicate* weak
// thread_local definition to available_externally. The IRMaterializationUnit
// ctor skips available_externally globals, so the duplicate contributes no
// symbol and the buggy discard is never reached; the first module stays the
// sole definition and later modules resolve to it within the JITDylib.
//
// Every weak thread_local is covered regardless of initializer: a thread_local
// with a non-trivial ctor (e.g. one holding a std::unique_ptr) is a
// zeroinitializer in IR -- its real init runs in a __tls_init function -- yet
// still emits the companion that trips discard, so the initializer must not be
// used to filter. Only weak thread_locals are demoted; demoting other weak
// globals or functions mis-resolves references/calls at runtime in the
// incremental JIT.
//
// ELF only: that is where the interpreter's emulated TLS makes the buggy
// discard reachable. On COFF the demotion orphans the comdat group's
// associated members ("Associative COMDAT symbol ... is not a key for its
// COMDAT"), and on Mach-O (native TLS, no comdats) it breaks the paired
// _ZTH/_ZTW thread_local init emission -- and on both, the JIT fails to
// resolve __emutls_get_address before the discard bug is even reachable.
//
// DEPRECATION: exists only for clang < 24 -- the upstream fix
// (llvm/llvm-project#208413) ships in clang 24, and every use site carries the
// same guard. Delete the workaround and its guarded call sites once clang-23
// support is dropped from CppInterOp.
// ===========================================================================
#if CLANG_VERSION_MAJOR < 24
inline void dedupeWeakEmulatedTLS(llvm::Module& M, llvm::StringSet<>& Defined) {
  if (!llvm::Triple(M.getTargetTriple()).isOSBinFormatELF())
    return;

  for (llvm::GlobalVariable& GV : M.globals()) {
    if (!GV.isThreadLocal() || GV.isDeclaration() || !GV.isWeakForLinker())
      continue;

    // First definer wins; later duplicates become references, mirroring what
    // discard() would have done (available_externally + no comdat) but without
    // hitting the crashing code path.
    if (!Defined.insert(GV.getName()).second) {
      GV.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
      GV.setComdat(nullptr);
    }
  }
}
#endif

// Clang 18 - Add new Interpreter methods: CodeComplete

inline llvm::orc::LLJIT* getExecutionEngine(clang::Interpreter& I) {
#if CLANG_VERSION_MAJOR < 22
  auto* engine = &llvm::cantFail(I.getExecutionEngine());
  return const_cast<llvm::orc::LLJIT*>(engine);
#else
  // FIXME: Remove the need of exposing the low-level execution engine and kill
  // this horrible hack.
  struct OrcIncrementalExecutor : public clang::IncrementalExecutor {
    std::unique_ptr<llvm::orc::LLJIT> Jit;
  };

  auto& engine = static_cast<OrcIncrementalExecutor&>(
      llvm::cantFail(I.getExecutionEngine()));
  return engine.Jit.get();
#endif
}

inline llvm::Expected<llvm::JITTargetAddress>
getSymbolAddress(clang::Interpreter& I, llvm::StringRef IRName) {

  auto AddrOrErr = I.getSymbolAddress(IRName);
  if (llvm::Error Err = AddrOrErr.takeError())
    return std::move(Err);
  return AddrOrErr->getValue();
}

inline llvm::Expected<llvm::JITTargetAddress>
getSymbolAddress(clang::Interpreter& I, clang::GlobalDecl GD) {
  std::string MangledName;
  compat::maybeMangleDeclName(GD, MangledName);
  return getSymbolAddress(I, llvm::StringRef(MangledName));
}

inline llvm::Expected<llvm::JITTargetAddress>
getSymbolAddressFromLinkerName(clang::Interpreter& I,
                               llvm::StringRef LinkerName) {
  const auto& DL = getExecutionEngine(I)->getDataLayout();
  char GlobalPrefix = DL.getGlobalPrefix();
  std::string LinkerNameTmp(LinkerName);
  if (GlobalPrefix != '\0') {
    LinkerNameTmp = std::string(1, GlobalPrefix) + LinkerNameTmp;
  }
  auto AddrOrErr = I.getSymbolAddressFromLinkerName(LinkerNameTmp);
  if (llvm::Error Err = AddrOrErr.takeError())
    return std::move(Err);
  return AddrOrErr->getValue();
}

inline llvm::Error Undo(clang::Interpreter& I, unsigned N = 1) {
  return I.Undo(N);
}

inline void codeComplete(std::vector<std::string>& Results,
                         clang::Interpreter& I, const char* code,
                         unsigned complete_line = 1U,
                         unsigned complete_column = 1U) {
  // FIXME: We should match the invocation arguments of the main interpreter.
  //        That can affect the returned completion results.
  auto CB = clang::IncrementalCompilerBuilder();
  auto CI = CB.CreateCpp();
  if (auto Err = CI.takeError()) {
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
    return;
  }
  auto Interp = clang::Interpreter::create(std::move(*CI));
  if (auto Err = Interp.takeError()) {
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
    return;
  }

  std::vector<std::string> results;
  clang::CompilerInstance* MainCI = (*Interp)->getCompilerInstance();
  auto CC = clang::ReplCodeCompleter();
  CC.codeComplete(MainCI, code, complete_line, complete_column,
                  I.getCompilerInstance(), results);
  for (llvm::StringRef r : results)
    if (r.find(CC.Prefix) == 0)
      Results.push_back(r.str());
}

} // namespace compat

#include "CppInterOpInterpreter.h"

namespace compat {
using Interpreter = CppInternal::Interpreter;

class SynthesizingCodeRAII {
private:
  [[maybe_unused]] Interpreter* m_Interpreter;

public:
  SynthesizingCodeRAII(Interpreter* i) : m_Interpreter(i) {}
  // ~SynthesizingCodeRAII() {} // TODO: implement
};

} // namespace compat

#endif // CPPINTEROP_USE_REPL

namespace compat {

// QualType for a TypeDecl. Pass a TypeDecl base pointer: Clang 22 deleted the
// TagDecl/TypedefDecl overloads, but the surviving TypeDecl one dispatches to
// getCanonicalTagType for tags, covering all decl kinds on Clang 21 and 22.
inline clang::QualType GetTypeFromDecl(const clang::TypeDecl* TD) {
  return TD->getASTContext().getTypeDeclType(TD);
}

#ifdef CPPINTEROP_USE_CLING
using Value = cling::Value;
#else
using Value = clang::Value;
#endif

// Clang >= 16 (=16 with Value patch) change castAs to convertTo
#ifdef CPPINTEROP_USE_CLING
template <typename T> inline T convertTo(cling::Value V) {
  return V.castAs<T>();
}
#else  // CLANG_REPL
template <typename T> inline T convertTo(clang::Value V) {
  return V.convertTo<T>();
}
#endif // CPPINTEROP_USE_CLING

// Refcount-shared payload wrapping a `compat::Value` for Cpp::Box's
// K_PtrOrObj slot. Boxing-via-copy (not move): clang::Value's move ctor
// releases its own storage on construction -- fixed upstream by
// llvm/llvm-project#200888. The copy ctor correctly retains.
// FIXME(llvm 23): static_assert below fails the build once the minimum
// LLVM crosses 23, prompting the move-semantics cleanup.
static_assert(LLVM_VERSION_MAJOR < 23,
              "clang::Value::Value(Value&&) was fixed upstream in "
              "llvm/llvm-project#200888; switch ValueRefCount to move "
              "semantics and drop this workaround.");

namespace detail {
struct ValueRefCount {
  std::atomic<unsigned> rc;
  Value v;
  explicit ValueRefCount(const Value& V) noexcept : rc(1), v(V) {}

  static void retain(void* p) noexcept {
    static_cast<ValueRefCount*>(p)->rc.fetch_add(1, std::memory_order_relaxed);
  }
  static void release(void* p) noexcept {
    auto* rc = static_cast<ValueRefCount*>(p);
    if (rc->rc.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete rc;
  }
  static constexpr Cpp::Box::ObjectOps Ops{&retain, &release};
};
} // namespace detail

/// Wrap a compat::Value into a refcount-shared K_PtrOrObj Cpp::Box.
/// `qt` is the opaque QualType (clang::QualType::getAsOpaquePtr()).
inline Cpp::Box MakeValueBox(const Value& V, void* qt) noexcept {
  return Cpp::Box::AdoptObject(new detail::ValueRefCount(V),
                               &detail::ValueRefCount::Ops, qt);
}

inline void InstantiateClassTemplateSpecialization(
    Interpreter& interp, clang::ClassTemplateSpecializationDecl* CTSD) {
#ifdef CPPINTEROP_USE_CLING
  cling::Interpreter::PushTransactionRAII RAII(&interp);
#endif
  interp.getSema().InstantiateClassTemplateSpecialization(
      clang::SourceLocation::getFromRawEncoding(1), CTSD,
      clang::TemplateSpecializationKind::TSK_ExplicitInstantiationDefinition,
      /*Complain=*/true,
      /*PrimaryHasMatchedPackOnParmToNonPackOnArg=*/false);
}

class DiagnosticsEngineRAII {
private:
  clang::DiagnosticsEngine& diags;

public:
  bool reset_condition; // additional condition to reset diagnostics

  DiagnosticsEngineRAII(clang::DiagnosticsEngine& d, bool c = true)
      : diags(d), reset_condition(c) {}
  ~DiagnosticsEngineRAII() {
    if (diags.hasErrorOccurred() && reset_condition) {
      // instantiation failed, need to reset DiagnosticsEngine
      diags.Reset(/*soft=*/true);
      diags.getClient()->clear();
    }
  }
};

} // namespace compat

#endif // CPPINTEROP_COMPATIBILITY_H
