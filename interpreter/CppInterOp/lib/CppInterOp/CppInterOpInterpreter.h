//--------------------------------------------------------------------*- C++ -*-
// CppInterOp Interpreter (clang-repl)
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------

#ifndef CPPINTEROP_INTERPRETER_H
#define CPPINTEROP_INTERPRETER_H

#include "Compatibility.h"
#include "DynamicLibraryManager.h"
#include "Paths.h"

#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/PartialTranslationUnit.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Redeclaration.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ModuleFileExtension.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#ifndef _WIN32
#include <sched.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <cstdio>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace clang {
class CompilerInstance;
}

namespace {
template <typename D> static D* LookupResult2Decl(clang::LookupResult& R) {
  if (R.empty())
    return nullptr;

  R.resolveKind();

  if (R.isSingleResult())
    return llvm::dyn_cast<D>(R.getFoundDecl());
  return (D*)-1;
}
} // namespace

namespace CppInternal {
namespace utils {
namespace Lookup {

inline clang::NamespaceDecl* Namespace(clang::Sema* S, const char* Name,
                                       const clang::DeclContext* Within) {
  clang::DeclarationName DName = &(S->Context.Idents.get(Name));
  clang::LookupResult R(*S, DName, clang::SourceLocation(),
                        clang::Sema::LookupNestedNameSpecifierName);
  R.suppressDiagnostics();
  if (!Within)
    S->LookupName(R, S->TUScope);
  else {
    if (const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(Within)) {
      if (!TD->getDefinition()) {
        // No definition, no lookup result.
        return nullptr;
      }
    }
    S->LookupQualifiedName(R, const_cast<clang::DeclContext*>(Within));
  }

  if (R.empty())
    return nullptr;

  R.resolveKind();

  return llvm::dyn_cast<clang::NamespaceDecl>(R.getFoundDecl());
}

inline void Named(clang::Sema* S, clang::LookupResult& R,
                  const clang::DeclContext* Within = nullptr) {
  R.suppressDiagnostics();
  if (!Within)
    S->LookupName(R, S->TUScope);
  else {
    const clang::DeclContext* primaryWithin = nullptr;
    if (const clang::TagDecl* TD = llvm::dyn_cast<clang::TagDecl>(Within)) {
      primaryWithin =
          llvm::dyn_cast_or_null<clang::DeclContext>(TD->getDefinition());
    } else {
      primaryWithin = Within->getPrimaryContext();
    }
    if (!primaryWithin) {
      // No definition, no lookup result.
      return;
    }
    S->LookupQualifiedName(R, const_cast<clang::DeclContext*>(primaryWithin));
  }
}

inline clang::NamedDecl* Named(clang::Sema* S,
                               const clang::DeclarationName& Name,
                               const clang::DeclContext* Within = nullptr) {
  clang::LookupResult R(*S, Name, clang::SourceLocation(),
                        clang::Sema::LookupOrdinaryName,
                        RedeclarationKind::ForVisibleRedeclaration);
  Named(S, R, Within);
  return LookupResult2Decl<clang::NamedDecl>(R);
}

inline clang::NamedDecl* Named(clang::Sema* S, llvm::StringRef Name,
                               const clang::DeclContext* Within = nullptr) {
  clang::DeclarationName DName = &S->Context.Idents.get(Name);
  return Named(S, DName, Within);
}

inline clang::NamedDecl* Named(clang::Sema* S, const char* Name,
                               const clang::DeclContext* Within = nullptr) {
  return Named(S, llvm::StringRef(Name), Within);
}

} // namespace Lookup
} // namespace utils
} // namespace CppInternal

namespace CppInternal {

/// CppInterOp Interpreter
///
class Interpreter {
public:
  struct FileDeleter {
    void operator()(FILE* f /* owns */) {
      if (f)
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        fclose(f);
    }
  };

  struct IOContext {
    std::unique_ptr<FILE, FileDeleter> stdin_file;
    std::unique_ptr<FILE, FileDeleter> stdout_file;
    std::unique_ptr<FILE, FileDeleter> stderr_file;

    bool initializeTempFiles() {
      stdin_file.reset(tmpfile());  // NOLINT(cppcoreguidelines-owning-memory)
      stdout_file.reset(tmpfile()); // NOLINT(cppcoreguidelines-owning-memory)
      stderr_file.reset(tmpfile()); // NOLINT(cppcoreguidelines-owning-memory)
      return stdin_file && stdout_file && stderr_file;
    }
  };

private:
  static std::tuple<int, int, int>
  initAndGetFileDescriptors(std::vector<const char*>& vargs,
                            std::unique_ptr<IOContext>& io_ctx) {
    int stdin_fd = 0;
    int stdout_fd = 1;
    int stderr_fd = 2;

    // Only initialize temp files if not already initialized
    if (!io_ctx->stdin_file || !io_ctx->stdout_file || !io_ctx->stderr_file) {
      bool init = io_ctx->initializeTempFiles();
      if (!init) {
        llvm::errs() << "Can't start out-of-process JIT execution.\n";
        stdin_fd = -1;
        stdout_fd = -1;
        stderr_fd = -1;
      }
    }
    stdin_fd = fileno(io_ctx->stdin_file.get());
    stdout_fd = fileno(io_ctx->stdout_file.get());
    stderr_fd = fileno(io_ctx->stderr_file.get());

    return std::make_tuple(stdin_fd, stdout_fd, stderr_fd);
  }

  std::unique_ptr<clang::Interpreter> inner;
  std::unique_ptr<IOContext> io_context;
  bool outOfProcess;

public:
  Interpreter(std::unique_ptr<clang::Interpreter> CI,
              std::unique_ptr<IOContext> ctx = nullptr, bool oop = false)
      : inner(std::move(CI)), io_context(std::move(ctx)), outOfProcess(oop) {}

public:
  static std::unique_ptr<Interpreter>
  create(int argc, const char* const* argv, const char* llvmdir = nullptr,
         const std::vector<std::shared_ptr<clang::ModuleFileExtension>>&
             moduleExtensions = {},
         void* extraLibHandle = nullptr, bool noRuntime = true) {
    // Initialize all targets (required for device offloading)
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    std::vector<const char*> vargs(argv + 1, argv + argc);

    int stdin_fd = 0;
    int stdout_fd = 1;
    int stderr_fd = 2;
    auto io_ctx = std::make_unique<IOContext>();
    bool outOfProcess = false;

#if defined(_WIN32) || !defined(LLVM_BUILT_WITH_OOP_JIT)
    outOfProcess = false;
#else
    outOfProcess = std::any_of(vargs.begin(), vargs.end(), [](const char* arg) {
      return llvm::StringRef(arg).trim() == "--use-oop-jit";
    });
#endif

    if (outOfProcess) {
      std::tie(stdin_fd, stdout_fd, stderr_fd) =
          initAndGetFileDescriptors(vargs, io_ctx);

      if (stdin_fd == -1 || stdout_fd == -1 || stderr_fd == -1) {
        llvm::errs()
            << "Redirection files creation failed for Out-Of-Process JIT\n";
        return nullptr;
      }
    }

    // Currently, we can't pass IOContext in `createClangInterpreter`, that's
    // why fd's are passed. This should be refactored later.
    auto CI =
        compat::createClangInterpreter(vargs, stdin_fd, stdout_fd, stderr_fd);
    if (!CI) {
      llvm::errs() << "Interpreter creation failed\n";
      return nullptr;
    }

    return std::make_unique<Interpreter>(std::move(CI), std::move(io_ctx),
                                         outOfProcess);
  }

  ~Interpreter() {}

  operator const clang::Interpreter&() const { return *inner; }
  operator clang::Interpreter&() { return *inner; }

  [[nodiscard]] bool isOutOfProcess() const { return outOfProcess; }

// Since, we are using custom pipes instead of stdout, sterr,
// it is kind of necessary to have this complication in StreamCaptureInfo.

// TODO(issues/733): Refactor the stream redirection
#ifndef _WIN32
  FILE* getRedirectionFileForOutOfProcess(int FD) {
    if (!io_context)
      return nullptr;
    switch (FD) {
    case (STDIN_FILENO):
      return io_context->stdin_file.get();
    case (STDOUT_FILENO):
      return io_context->stdout_file.get();
    case (STDERR_FILENO):
      return io_context->stderr_file.get();
    default:
      llvm::errs() << "No temp file for the FD\n";
      return nullptr;
    }
  }
#endif

  ///\brief Describes the return result of the different routines that do the
  /// incremental compilation.
  ///
  enum CompilationResult { kSuccess, kFailure, kMoreInputExpected };

  const clang::CompilerInstance* getCompilerInstance() const {
    return inner->getCompilerInstance();
  }

  llvm::orc::LLJIT* getExecutionEngine() const {
    return compat::getExecutionEngine(*inner);
  }

  llvm::Expected<clang::PartialTranslationUnit&> Parse(llvm::StringRef Code) {
    return inner->Parse(Code);
  }

  llvm::Error Execute(clang::PartialTranslationUnit& T) {
    return inner->Execute(T);
  }

  llvm::Error ParseAndExecute(llvm::StringRef Code, clang::Value* V = nullptr) {
    return inner->ParseAndExecute(Code, V);
  }

  llvm::Error Undo(unsigned N = 1) { return compat::Undo(*inner, N); }

  void makeEngineOnce() const {
    static bool make_engine_once = true;
    if (make_engine_once) {
      if (auto Err = inner->ParseAndExecute(""))
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "Error:");
      make_engine_once = false;
    }
  }

  /// \returns the \c ExecutorAddr of a \c GlobalDecl. This interface uses
  /// the CodeGenModule's internal mangling cache to avoid recomputing the
  /// mangled name.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(clang::GlobalDecl GD) const {
    makeEngineOnce();
    auto AddrOrErr = compat::getSymbolAddress(*inner, GD);
    if (llvm::Error Err = AddrOrErr.takeError())
      return std::move(Err);
    return llvm::orc::ExecutorAddr(*AddrOrErr);
  }

  /// \returns the \c ExecutorAddr of a given name as written in the IR.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef IRName) const {
    makeEngineOnce();
    auto AddrOrErr = compat::getSymbolAddress(*inner, IRName);
    if (llvm::Error Err = AddrOrErr.takeError())
      return std::move(Err);
    return llvm::orc::ExecutorAddr(*AddrOrErr);
  }

#ifndef _WIN32
  [[nodiscard]] pid_t getOutOfProcessExecutorPID() const {
#ifdef LLVM_BUILT_WITH_OOP_JIT
    return inner->getOutOfProcessExecutorPID();
#endif
    return 0;
  }
#endif

  /// \returns the \c ExecutorAddr of a given name as written in the object
  /// file.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddressFromLinkerName(llvm::StringRef LinkerName) const {
    auto AddrOrErr = compat::getSymbolAddressFromLinkerName(*inner, LinkerName);
    if (llvm::Error Err = AddrOrErr.takeError())
      return std::move(Err);
    return llvm::orc::ExecutorAddr(*AddrOrErr);
  }

  bool isInSyntaxOnlyMode() const {
    return getCompilerInstance()->getFrontendOpts().ProgramAction ==
           clang::frontend::ParseSyntaxOnly;
  }

  // FIXME: Mangle GD and call the other overload.
  void* getAddressOfGlobal(const clang::GlobalDecl& GD) const {
    auto addressOrErr = getSymbolAddress(GD);
    if (addressOrErr)
      return addressOrErr->toPtr<void*>();

    llvm::consumeError(addressOrErr.takeError()); // okay to be missing
    return nullptr;
  }

  void* getAddressOfGlobal(llvm::StringRef SymName) const {
    if (isInSyntaxOnlyMode())
      return nullptr;

    auto addressOrErr =
        getSymbolAddressFromLinkerName(SymName); // TODO: Or getSymbolAddress
    if (addressOrErr)
      return addressOrErr->toPtr<void*>();

    llvm::consumeError(addressOrErr.takeError()); // okay to be missing
    return nullptr;
  }

  CompilationResult declare(const std::string& input,
                            clang::PartialTranslationUnit** PTU = nullptr) {
    return process(input, /*Value=*/nullptr, PTU);
  }

  ///\brief Maybe transform the input line to implement cint command line
  /// semantics (declarations are global) and compile to produce a module.
  ///
  CompilationResult process(const std::string& input, clang::Value* V = 0,
                            clang::PartialTranslationUnit** PTU = nullptr,
                            bool disableValuePrinting = false) {
    auto PTUOrErr = Parse(input);
    if (!PTUOrErr) {
      llvm::logAllUnhandledErrors(PTUOrErr.takeError(), llvm::errs(),
                                  "Failed to parse via ::process:");
      return Interpreter::kFailure;
    }

    if (PTU)
      *PTU = &*PTUOrErr;

    if (auto Err = Execute(*PTUOrErr)) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Failed to execute via ::process:");
      return Interpreter::kFailure;
    }
    return Interpreter::kSuccess;
  }

  CompilationResult evaluate(const std::string& input, clang::Value& V) {
    if (auto Err = ParseAndExecute(input, &V)) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Failed to execute via ::evaluate:");
      return Interpreter::kFailure;
    }
    return Interpreter::kSuccess;
  }

  void* compileFunction(llvm::StringRef name, llvm::StringRef code,
                        bool ifUnique, bool withAccessControl) {
    //
    //  Compile the wrapper code.
    //

    if (isInSyntaxOnlyMode())
      return nullptr;

    if (ifUnique) {
      if (void* Addr = (void*)getAddressOfGlobal(name)) {
        return Addr;
      }
    }

    clang::LangOptions& LO =
        const_cast<clang::LangOptions&>(getCompilerInstance()->getLangOpts());
    bool SavedAccessControl = LO.AccessControl;
    LO.AccessControl = withAccessControl;

    if (auto Err = ParseAndExecute(code)) {
      LO.AccessControl = SavedAccessControl;
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Failed to compileFunction: ");
      return nullptr;
    }

    LO.AccessControl = SavedAccessControl;

    return getAddressOfGlobal(name);
  }

  const clang::CompilerInstance* getCI() const { return getCompilerInstance(); }

  clang::Sema& getSema() const { return getCI()->getSema(); }

  const DynamicLibraryManager* getDynamicLibraryManager() const {
    assert(compat::getExecutionEngine(*inner) && "We must have an executor");
    static std::unique_ptr<DynamicLibraryManager> DLM = nullptr;
    if (!DLM) {
      DLM.reset(new DynamicLibraryManager());
      DLM->initializeDyld([](llvm::StringRef) { /*ignore*/ return false; });
    }
    return DLM.get();
    // TODO: Add DLM to InternalExecutor and use executor->getDML()
    //      return inner->getExecutionEngine()->getDynamicLibraryManager();
  }

  DynamicLibraryManager* getDynamicLibraryManager() {
    return const_cast<DynamicLibraryManager*>(
        const_cast<const Interpreter*>(this)->getDynamicLibraryManager());
  }

  ///\brief Adds multiple include paths separated by a delimiter.
  ///
  ///\param[in] PathsStr - Path(s)
  ///\param[in] Delim - Delimiter to separate paths or NULL if a single path
  ///
  void AddIncludePaths(llvm::StringRef PathsStr, const char* Delim = ":") {
    const clang::CompilerInstance* CI = getCompilerInstance();
    clang::HeaderSearchOptions& HOpts =
        const_cast<clang::HeaderSearchOptions&>(CI->getHeaderSearchOpts());

    // Save the current number of entries
    size_t Idx = HOpts.UserEntries.size();
    CppInternal::utils::AddIncludePaths(PathsStr, HOpts, Delim);

    clang::Preprocessor& PP = CI->getPreprocessor();
    clang::SourceManager& SM = PP.getSourceManager();
    clang::FileManager& FM = SM.getFileManager();
    clang::HeaderSearch& HSearch = PP.getHeaderSearchInfo();
    const bool isFramework = false;

    // Add all the new entries into Preprocessor
    for (const size_t N = HOpts.UserEntries.size(); Idx < N; ++Idx) {
      const clang::HeaderSearchOptions::Entry& E = HOpts.UserEntries[Idx];
      if (auto DE = FM.getOptionalDirectoryRef(E.Path))
        HSearch.AddSearchPath(
            clang::DirectoryLookup(*DE, clang::SrcMgr::C_User, isFramework),
            E.Group == clang::frontend::Angled);
    }
  }

  ///\brief Adds a single include path (-I).
  ///
  void AddIncludePath(llvm::StringRef PathsStr) {
    return AddIncludePaths(PathsStr, nullptr);
  }

  ///\brief Get the current include paths that are used.
  ///
  ///\param[out] incpaths - Pass in a llvm::SmallVector<std::string, N> with
  ///       sufficiently sized N, to hold the result of the call.
  ///\param[in] withSystem - if true, incpaths will also contain system
  ///       include paths (framework, STL etc).
  ///\param[in] withFlags - if true, each element in incpaths will be prefixed
  ///       with a "-I" or similar, and some entries of incpaths will signal
  ///       a new include path region (e.g. "-cxx-isystem"). Also, flags
  ///       defining header search behavior will be included in incpaths, e.g.
  ///       "-nostdinc".
  ///
  void GetIncludePaths(llvm::SmallVectorImpl<std::string>& incpaths,
                       bool withSystem, bool withFlags) const {
    CppInternal::utils::CopyIncludePaths(getCI()->getHeaderSearchOpts(),
                                         incpaths, withSystem, withFlags);
  }

  CompilationResult loadLibrary(const std::string& filename, bool lookup) {
    llvm::Triple triple(getCompilerInstance()->getTargetOpts().Triple);
    if (triple.isWasm()) {
      // On WASM, dlopen-style canonical lookup has no effect.
      if (auto Err = inner->LoadDynamicLibrary(filename.c_str())) {
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                    "loadLibrary: ");
        return kFailure;
      }
      return kSuccess;
    }

    DynamicLibraryManager* DLM = getDynamicLibraryManager();
    std::string canonicalLib;
    if (lookup)
      canonicalLib = DLM->lookupLibrary(filename);

    const std::string& library = lookup ? canonicalLib : filename;
    if (!library.empty()) {
      switch (
          DLM->loadLibrary(library, /*permanent*/ false, /*resolved*/ true)) {
      case DynamicLibraryManager::kLoadLibSuccess: // Intentional fall through
      case DynamicLibraryManager::kLoadLibAlreadyLoaded:
        return kSuccess;
      case DynamicLibraryManager::kLoadLibNotFound:
        assert(0 && "Cannot find library with existing canonical name!");
        return kFailure;
      default:
        // Not a source file (canonical name is non-empty) but can't load.
        return kFailure;
      }
    }
    return kMoreInputExpected;
  }

  std::string toString(const char* type, void* obj) {
    assert(0 && "toString is not implemented!");
    std::string ret;
    return ret; // TODO: Implement
  }

  CompilationResult undo(unsigned N = 1) {
    if (llvm::Error Err = Undo(N)) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "Failed to undo via ::undo");
      return kFailure;
    }
    return kSuccess;
  }

}; // Interpreter
} // namespace CppInternal

#endif // CPPINTEROP_INTERPRETER_H
