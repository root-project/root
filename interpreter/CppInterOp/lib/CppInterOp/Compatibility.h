//--------------------------------------------------------------------*- C++ -*-
// CppInterOp Compatibility
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------
#ifndef CPPINTEROP_COMPATIBILITY_H
#define CPPINTEROP_COMPATIBILITY_H

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Sema/Sema.h"

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

#if CLANG_VERSION_MAJOR < 19
#define Template_Deduction_Result Sema::TemplateDeductionResult
#define Template_Deduction_Result_Success                                      \
  Sema::TemplateDeductionResult::TDK_Success
#else
#define Template_Deduction_Result TemplateDeductionResult
#define Template_Deduction_Result_Success TemplateDeductionResult::Success
#endif

#if CLANG_VERSION_MAJOR < 19
#define For_Visible_Redeclaration Sema::ForVisibleRedeclaration
#define Clang_For_Visible_Redeclaration clang::Sema::ForVisibleRedeclaration
#else
#define For_Visible_Redeclaration RedeclarationKind::ForVisibleRedeclaration
#define Clang_For_Visible_Redeclaration                                        \
  RedeclarationKind::ForVisibleRedeclaration
#endif

#if CLANG_VERSION_MAJOR < 19
#define CXXSpecialMemberKindDefaultConstructor                                 \
  clang::Sema::CXXDefaultConstructor
#define CXXSpecialMemberKindCopyConstructor clang::Sema::CXXCopyConstructor
#define CXXSpecialMemberKindMoveConstructor clang::Sema::CXXMoveConstructor
#else
#define CXXSpecialMemberKindDefaultConstructor                                 \
  CXXSpecialMemberKind::DefaultConstructor
#define CXXSpecialMemberKindCopyConstructor                                    \
  CXXSpecialMemberKind::CopyConstructor
#define CXXSpecialMemberKindMoveConstructor                                    \
  CXXSpecialMemberKind::MoveConstructor
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

#include "llvm/Support/Error.h"

#ifdef LLVM_BUILT_WITH_OOP_JIT
#include "clang/Basic/Version.h"
#include "llvm/TargetParser/Host.h"

#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"

#include <unistd.h>
#endif

#include <algorithm>

namespace compat {

inline std::unique_ptr<clang::Interpreter>
createClangInterpreter(std::vector<const char*>& args, int stdin_fd = -1,
                       int stdout_fd = -1, int stderr_fd = -1) {
  auto has_arg = [](const char* x, llvm::StringRef match = "cuda") {
    llvm::StringRef Arg = x;
    Arg = Arg.trim().ltrim('-');
    return Arg == match;
  };
  auto it = std::find_if(args.begin(), args.end(), has_arg);
  std::vector<const char*> gpu_args = {it, args.end()};
#ifdef __APPLE__
  bool CudaEnabled = false;
#else
  bool CudaEnabled = !gpu_args.empty();
#endif

  clang::IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs({args.begin(), it});

  std::unique_ptr<clang::CompilerInstance> DeviceCI;
  if (CudaEnabled) {
    // FIXME: Parametrize cuda-path and offload-arch.
    CB.SetOffloadArch("sm_35");
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
  auto* engine = &llvm::cantFail(I.getExecutionEngine());
  return const_cast<llvm::orc::LLJIT*>(engine);
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
  Interpreter* m_Interpreter;

public:
  SynthesizingCodeRAII(Interpreter* i) : m_Interpreter(i) {}
  ~SynthesizingCodeRAII() {
    auto GeneratedPTU = m_Interpreter->Parse("");
    if (!GeneratedPTU)
      llvm::logAllUnhandledErrors(GeneratedPTU.takeError(), llvm::errs(),
                                  "Failed to generate PTU:");
  }
};
} // namespace compat

#endif // CPPINTEROP_USE_REPL

namespace compat {

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
#if CLANG_VERSION_MAJOR < 20
  interp.getSema().InstantiateClassTemplateSpecialization(
      clang::SourceLocation::getFromRawEncoding(1), CTSD,

      clang::TemplateSpecializationKind::TSK_ExplicitInstantiationDefinition,
      /*Complain=*/true);
#else
  interp.getSema().InstantiateClassTemplateSpecialization(
      clang::SourceLocation::getFromRawEncoding(1), CTSD,
      clang::TemplateSpecializationKind::TSK_ExplicitInstantiationDefinition,
      /*Complain=*/true,
      /*PrimaryHasMatchedPackOnParmToNonPackOnArg=*/false);
#endif
}
} // namespace compat

#endif // CPPINTEROP_COMPATIBILITY_H
