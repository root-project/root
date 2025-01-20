//--------------------------------------------------------------------*- C++ -*-
// CppInterOp Compatibility
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------
#ifndef CPPINTEROP_COMPATIBILITY_H
#define CPPINTEROP_COMPATIBILITY_H

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"

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

#if LLVM_VERSION_MAJOR < 18
#define starts_with startswith
#define ends_with endswith
#endif

#if CLANG_VERSION_MAJOR >= 18
#include "clang/Interpreter/CodeCompletion.h"
#endif

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

namespace Cpp {
namespace Cpp_utils = cling::utils;
}

namespace compat {

using Interpreter = cling::Interpreter;

inline void maybeMangleDeclName(const clang::GlobalDecl& GD,
                                std::string& mangledName) {
  cling::utils::Analyze::maybeMangleDeclName(GD, mangledName);
}

/// For Cling <= LLVM 16, this is a horrible hack obtaining the private
/// llvm::orc::LLJIT by computing the object offsets in the cling::Interpreter
/// instance(IncrementalExecutor): sizeof (m_Opts) + sizeof(m_LLVMContext). The
/// IncrementalJIT and JIT itself have an offset of 0 as the first datamember.
///
/// The getExecutionEngine() interface has been added for Cling based on LLVM
/// >=18 and should be used in future releases.
inline llvm::orc::LLJIT* getExecutionEngine(cling::Interpreter& I) {
#if CLANG_VERSION_MAJOR >= 18
  return I.getExecutionEngine();
#endif

  unsigned m_ExecutorOffset = 0;

#if CLANG_VERSION_MAJOR == 13
#ifdef __APPLE__
  m_ExecutorOffset = 62;
#else
  m_ExecutorOffset = 72;
#endif // __APPLE__
#endif

// Note: The offsets changed in Cling based on LLVM 16 with the introduction of
// a thread safe context - llvm::orc::ThreadSafeContext
#if CLANG_VERSION_MAJOR == 16
#ifdef __APPLE__
  m_ExecutorOffset = 68;
#else
  m_ExecutorOffset = 78;
#endif // __APPLE__
#endif

  int* IncrementalExecutor =
      ((int*)(const_cast<cling::Interpreter*>(&I))) + m_ExecutorOffset;
  int* IncrementalJit = *(int**)IncrementalExecutor + 0;
  int* LLJIT = *(int**)IncrementalJit + 0;
  return *(llvm::orc::LLJIT**)LLJIT;
}

inline llvm::Expected<llvm::JITTargetAddress>
getSymbolAddress(cling::Interpreter& I, llvm::StringRef IRName) {
  if (void* Addr = I.getAddressOfGlobal(IRName))
    return (llvm::JITTargetAddress)Addr;

  llvm::orc::LLJIT& Jit = *compat::getExecutionEngine(I);
  llvm::orc::SymbolNameVector Names;
  llvm::orc::ExecutionSession& ES = Jit.getExecutionSession();
  Names.push_back(ES.intern(IRName));
#if CLANG_VERSION_MAJOR < 16
  return llvm::make_error<llvm::orc::SymbolsNotFound>(Names);
#else
  return llvm::make_error<llvm::orc::SymbolsNotFound>(ES.getSymbolStringPool(),
                                                      std::move(Names));
#endif // CLANG_VERSION_MAJOR
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

namespace compat {

inline std::unique_ptr<clang::Interpreter>
createClangInterpreter(std::vector<const char*>& args) {
#if CLANG_VERSION_MAJOR < 16
  auto ciOrErr = clang::IncrementalCompilerBuilder::create(args);
#else
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
#endif // CLANG_VERSION_MAJOR < 16
  if (!ciOrErr) {
    llvm::logAllUnhandledErrors(ciOrErr.takeError(), llvm::errs(),
                                "Failed to build Incremental compiler:");
    return nullptr;
  }
#if CLANG_VERSION_MAJOR < 16
  auto innerOrErr = clang::Interpreter::create(std::move(*ciOrErr));
#else
  (*ciOrErr)->LoadRequestedPlugins();
  if (CudaEnabled)
    DeviceCI->LoadRequestedPlugins();
  auto innerOrErr =
      CudaEnabled ? clang::Interpreter::createWithCUDA(std::move(*ciOrErr),
                                                       std::move(DeviceCI))
                  : clang::Interpreter::create(std::move(*ciOrErr));
#endif // CLANG_VERSION_MAJOR < 16

  if (!innerOrErr) {
    llvm::logAllUnhandledErrors(innerOrErr.takeError(), llvm::errs(),
                                "Failed to build Interpreter:");
    return nullptr;
  }
  if (CudaEnabled) {
    if (auto Err = (*innerOrErr)->LoadDynamicLibrary("libcudart.so"))
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Failed load libcudart.so runtime:");
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

// Clang 13 - Initial implementation of Interpreter and clang-repl
// Clang 14 - Add new Interpreter methods: getExecutionEngine,
//            getSymbolAddress, getSymbolAddressFromLinkerName
// Clang 15 - Add new Interpreter methods: Undo
// Clang 18 - Add new Interpreter methods: CodeComplete

inline llvm::orc::LLJIT* getExecutionEngine(clang::Interpreter& I) {
#if CLANG_VERSION_MAJOR >= 14
  auto* engine = &llvm::cantFail(I.getExecutionEngine());
  return const_cast<llvm::orc::LLJIT*>(engine);
#else
  assert(0 && "Not implemented in Clang <14!");
  return nullptr;
#endif
}

inline llvm::Expected<llvm::JITTargetAddress>
getSymbolAddress(clang::Interpreter& I, llvm::StringRef IRName) {
#if CLANG_VERSION_MAJOR < 14
  assert(0 && "Not implemented in Clang <14!");
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Not implemented in Clang <14!");
#endif // CLANG_VERSION_MAJOR < 14

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
#if CLANG_VERSION_MAJOR >= 14
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
#else
  assert(0 && "Not implemented in Clang <14!");
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Not implemented in Clang <14!");
#endif
}

inline llvm::Error Undo(clang::Interpreter& I, unsigned N = 1) {
#if CLANG_VERSION_MAJOR >= 15
  return I.Undo(N);
#else
  assert(0 && "Not implemented in Clang <15!");
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Not implemented in Clang <15!");
#endif
}

inline void codeComplete(std::vector<std::string>& Results,
                         clang::Interpreter& I, const char* code,
                         unsigned complete_line = 1U,
                         unsigned complete_column = 1U) {
#if CLANG_VERSION_MAJOR >= 18
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
  std::vector<std::string> Comps;
  clang::CompilerInstance* MainCI = (*Interp)->getCompilerInstance();
  auto CC = clang::ReplCodeCompleter();
  CC.codeComplete(MainCI, code, complete_line, complete_column,
                  I.getCompilerInstance(), results);
  for (llvm::StringRef r : results)
    if (r.find(CC.Prefix) == 0)
      Results.push_back(r.str());
#else
  assert(false && "CodeCompletion API only available in Clang >= 18.");
#endif
}

} // namespace compat

#include "CppInterOpInterpreter.h"

namespace Cpp {
namespace Cpp_utils = Cpp::utils;
}

namespace compat {
using Interpreter = Cpp::Interpreter;
}

#endif // CPPINTEROP_USE_REPL

namespace compat {

// Clang >= 14 change type name to string (spaces formatting problem)
#if CLANG_VERSION_MAJOR >= 14
inline std::string FixTypeName(const std::string type_name) {
  return type_name;
}
#else
inline std::string FixTypeName(const std::string type_name) {
  std::string result = type_name;
  size_t pos = 0;
  while ((pos = result.find(" [", pos)) != std::string::npos) {
    result.erase(pos, 1);
    pos++;
  }
  return result;
}
#endif

// Clang >= 16 change CLANG_LIBDIR_SUFFIX to CLANG_INSTALL_LIBDIR_BASENAME
#if CLANG_VERSION_MAJOR < 16
#define CLANG_INSTALL_LIBDIR_BASENAME (llvm::Twine("lib") + CLANG_LIBDIR_SUFFIX)
#endif
inline std::string MakeResourceDir(llvm::StringRef Dir) {
  llvm::SmallString<128> P(Dir);
  llvm::sys::path::append(P, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
#if CLANG_VERSION_MAJOR < 16
                          CLANG_VERSION_STRING
#else
                          CLANG_VERSION_MAJOR_STRING
#endif
  );
  return std::string(P.str());
}

// Clang >= 16 (=16 with Value patch) change castAs to converTo
#ifdef CPPINTEROP_USE_CLING
template <typename T> inline T convertTo(cling::Value V) {
  return V.castAs<T>();
}
#else  // CLANG_REPL
template <typename T> inline T convertTo(clang::Value V) {
  return V.convertTo<T>();
}
#endif // CPPINTEROP_USE_CLING

} // namespace compat

#endif // CPPINTEROP_COMPATIBILITY_H
