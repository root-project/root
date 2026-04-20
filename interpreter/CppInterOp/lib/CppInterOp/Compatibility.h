//--------------------------------------------------------------------*- C++ -*-
// CppInterOp Compatibility
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------
#ifndef CPPINTEROP_COMPATIBILITY_H
#define CPPINTEROP_COMPATIBILITY_H

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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"

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

#if CLANG_VERSION_MAJOR < 22
#define Get_Tag_Type getTagDeclType
#else
#define Get_Tag_Type getCanonicalTagType
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

#ifdef LLVM_BUILT_WITH_OOP_JIT
#include "clang/Basic/Version.h"

#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"

#include <unistd.h>
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

  bool outOfProcess;
#if defined(_WIN32) || !defined(LLVM_BUILT_WITH_OOP_JIT)
  outOfProcess = false;
#else
  outOfProcess = std::any_of(args.begin(), args.end(), [](const char* arg) {
    return llvm::StringRef(arg).trim() == "--use-oop-jit";
  });
#endif

#ifdef LLVM_BUILT_WITH_OOP_JIT

  clang::Interpreter::JITConfig OutOfProcessConfig;
  if (outOfProcess) {
    OutOfProcessConfig.IsOutOfProcess = true;
    OutOfProcessConfig.OOPExecutor =
        LLVM_BINARY_LIB_DIR "/bin/llvm-jitlink-executor";
    OutOfProcessConfig.UseSharedMemory = false;
    OutOfProcessConfig.SlabAllocateSize = 0;
    OutOfProcessConfig.CustomizeFork = [stdin_fd, stdout_fd,
                                        stderr_fd]() { // Lambda defined inline
      dup2(stdin_fd, STDIN_FILENO);
      dup2(stdout_fd, STDOUT_FILENO);
      dup2(stderr_fd, STDERR_FILENO);

      setvbuf(fdopen(stdout_fd, "w+"), nullptr, _IONBF, 0);
      setvbuf(fdopen(stderr_fd, "w+"), nullptr, _IONBF, 0);
    };

#ifdef __APPLE__
    std::string OrcRuntimePath = LLVM_BINARY_LIB_DIR "/lib/clang/" STRINGIFY(
        LLVM_VERSION_MAJOR) "/lib/darwin/liborc_rt_osx.a";
#else
    std::string OrcRuntimePath = LLVM_BINARY_LIB_DIR "/lib/clang/" STRINGIFY(
        LLVM_VERSION_MAJOR) "/lib/x86_64-unknown-linux-gnu/liborc_rt.a";
#endif
    OutOfProcessConfig.OrcRuntimePath = OrcRuntimePath;
  }
  auto innerOrErr =
      CudaEnabled
          ? clang::Interpreter::createWithCUDA(std::move(*ciOrErr),
                                               std::move(DeviceCI))
          : clang::Interpreter::create(std::move(*ciOrErr), OutOfProcessConfig);
#else
  if (outOfProcess) {
    llvm::errs()
        << "[CreateClangInterpreter]: No compatibility with out-of-process "
           "JIT. Running in-process JIT execution."
        << "(To enable recompile CppInterOp with -DLLVM_BUILT_WITH_OOP_JIT=ON)"
        << "\n";
  }
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
} // namespace compat

#endif // CPPINTEROP_COMPATIBILITY_H
