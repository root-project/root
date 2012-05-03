//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#include "cling/Interpreter/Interpreter.h"

#include "DynamicLookup.h"
#include "ExecutionContext.h"
#include "IncrementalParser.h"

#include "cling/Interpreter/CIFactory.h"
#include "cling/Interpreter/CompilationOptions.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Value.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#define private public
#include "clang/Parse/Parser.h"
#undef private
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/Linker.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

using namespace clang;

namespace {
static bool tryLinker(const std::string& filename,
                      const cling::InvocationOptions& Opts,
                      llvm::Module* module) {
  assert(module && "Module must exist for linking!");
  llvm::Linker L("cling", module, llvm::Linker::QuietWarnings
                 | llvm::Linker::QuietErrors);
  for (std::vector<llvm::sys::Path>::const_iterator I
         = Opts.LibSearchPath.begin(), E = Opts.LibSearchPath.end(); I != E;
       ++I) {
    L.addPath(*I);
  }
  L.addSystemPaths();
  bool Native = true;
  if (L.LinkInLibrary(filename, Native)) {
    // that didn't work, try bitcode:
    llvm::sys::Path FilePath(filename);
    std::string Magic;
    if (!FilePath.getMagicNumber(Magic, 64)) {
      // filename doesn't exist...
      L.releaseModule();
      return false;
    }
    if (llvm::sys::IdentifyFileType(Magic.c_str(), 64)
        == llvm::sys::Bitcode_FileType) {
      // We are promised a bitcode file, complain if it fails
      L.setFlags(0);
      if (L.LinkInFile(llvm::sys::Path(filename), Native)) {
        L.releaseModule();
        return false;
      }
    } else {
      // Nothing the linker can handle
      L.releaseModule();
      return false;
    }
  } else if (Native) {
    // native shared library, load it!
    llvm::sys::Path SoFile = L.FindLib(filename);
    assert(!SoFile.isEmpty() && "The shared lib exists but can't find it!");
    std::string errMsg;
    bool hasError = llvm::sys::DynamicLibrary
      ::LoadLibraryPermanently(SoFile.str().c_str(), &errMsg);
    if (hasError) {
      llvm::errs() << "Could not load shared library!\n" 
                   << "\n"
                   << errMsg.c_str();
      L.releaseModule();
      return false;
    }
  }
  L.releaseModule();
  return true;
}

static bool canWrapForCall(const std::string& input_line) {
   // Whether input_line can be wrapped into a function.
   // "1" can, "#include <vector>" can't.
   if (input_line.length() > 1 && input_line[0] == '#') return false;
   if (input_line.compare(0, strlen("extern "), "extern ") == 0) return false;
   if (input_line.compare(0, strlen("using "), "using ") == 0) return false;
   return true;
}

} // unnamed namespace

namespace cling {

  // "Declared" to the JIT in RuntimeUniverse.h
  namespace runtime {
    namespace internal {
      int local_cxa_atexit(void (*func) (void*), void* arg,
                           void* dso, Interpreter* interp) {
        return interp->CXAAtExit(func, arg, dso);
      }
      struct __trigger__cxa_atexit {
        ~__trigger__cxa_atexit();
      };
      __trigger__cxa_atexit::~__trigger__cxa_atexit() {}
    }
  }

  // This function isn't referenced outside its translation unit, but it
  // can't use the "static" keyword because its address is used for
  // GetMainExecutable (since some platforms don't support taking the
  // address of main, and some platforms can't implement GetMainExecutable
  // without being given the address of a function in the main executable).
  llvm::sys::Path GetExecutablePath(const char *Argv0) {
    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *MainAddr = (void*) (intptr_t) GetExecutablePath;
    return llvm::sys::Path::GetMainExecutable(Argv0, MainAddr);
  }


  Interpreter::NamedDeclResult::NamedDeclResult(llvm::StringRef Decl, 
                                                Interpreter* interp, 
                                                const DeclContext* Within)
    : m_Interpreter(interp),
      m_Context(m_Interpreter->getCI()->getASTContext()),
      m_CurDeclContext(Within),
      m_Result(0)
  {
    LookupDecl(Decl);
  }

  Interpreter::NamedDeclResult&
  Interpreter::NamedDeclResult::LookupDecl(llvm::StringRef Decl) {
    DeclarationName Name(&m_Context.Idents.get(Decl));
    DeclContext::lookup_const_result Lookup = m_CurDeclContext->lookup(Name);
    // If more than one found return 0. Cannot handle ambiguities.
    if (Lookup.second - Lookup.first == 1) {
      if (DeclContext* DC = dyn_cast<DeclContext>(*Lookup.first))
        m_CurDeclContext = DC;
      else
        m_CurDeclContext = (*Lookup.first)->getDeclContext();
      
      m_Result = (*Lookup.first);
    }
    else {
      m_Result = 0;
    }

    return *this;
  }

  NamedDecl* Interpreter::NamedDeclResult::getSingleDecl() const {
    // TODO: Check whether it is only one decl if (end-begin == 1 )
    return dyn_cast<NamedDecl>(m_Result);
  }

  const char* DynamicExprInfo::getExpr() {
    int i = 0;
    size_t found;

    while ((found = m_Result.find("@")) && (found != std::string::npos)) { 
      std::stringstream address;
      address << m_Addresses[i];
      m_Result = m_Result.insert(found + 1, address.str());
      m_Result = m_Result.erase(found, 1);
      ++i;
    }

    return m_Result.c_str();
  }

  Interpreter::Interpreter(int argc, const char* const *argv, 
                           const char* llvmdir /*= 0*/) :
    m_UniqueCounter(0), m_PrintAST(false), m_ValuePrinterEnabled(false) {

    std::vector<unsigned> LeftoverArgsIdx;
    m_Opts = InvocationOptions::CreateFromArgs(argc, argv, LeftoverArgsIdx);
    std::vector<const char*> LeftoverArgs;

    for (size_t I = 0, N = LeftoverArgsIdx.size(); I < N; ++I) {
      LeftoverArgs.push_back(argv[LeftoverArgsIdx[I]]);
    }
 
    m_IncrParser.reset(new IncrementalParser(this, LeftoverArgs.size(),
                                             &LeftoverArgs[0],
                                             llvmdir));
    m_ExecutionContext.reset(new ExecutionContext());

    m_ValuePrintStream.reset(new llvm::raw_os_ostream(std::cout));

    // Add path to interpreter's include files
    // Try to find the headers in the src folder first
#ifdef CLING_SRCDIR_INCL
    llvm::sys::Path SrcP(CLING_SRCDIR_INCL);
    if (SrcP.canRead())
      AddIncludePath(SrcP.str());
#endif

    llvm::sys::Path P = GetExecutablePath(argv[0]);
    if (!P.isEmpty()) {
      P.eraseComponent();  // Remove /cling from foo/bin/clang
      P.eraseComponent();  // Remove /bin   from foo/bin
      // Get foo/include
      P.appendComponent("include");
      if (P.canRead())
        AddIncludePath(P.str());
      else {
#ifdef CLING_INSTDIR_INCL
        llvm::sys::Path InstP(CLING_INSTDIR_INCL);
        if (InstP.canRead())
          AddIncludePath(InstP.str());
#endif
      }
    }

    // Warm them up
    m_IncrParser->Initialize();

    m_ExecutionContext->addSymbol("local_cxa_atexit", 
                  (void*)(intptr_t)&cling::runtime::internal::local_cxa_atexit);

    if (getCI()->getLangOpts().CPlusPlus) {
      // Set up common declarations which are going to be available
      // only at runtime
      // Make sure that the universe won't be included to compile time by using
      // -D __CLING__ as CompilerInstance's arguments

      declare("#include \"cling/Interpreter/RuntimeUniverse.h\"");
      declare("#include \"cling/Interpreter/ValuePrinter.h\"\n");

      // Set up the gCling variable
      std::stringstream initializer;
      initializer << "gCling=(cling::Interpreter*)" << (uintptr_t)this << ";";
      evaluate(initializer.str());
    }
    else {
      declare("#include \"cling/Interpreter/CValuePrinter.h\"\n");
    }

    handleFrontendOptions();
  }

  Interpreter::~Interpreter() {
    for (size_t I = 0, N = m_AtExitFuncs.size(); I < N; ++I) {
      const CXAAtExitElement& AEE = m_AtExitFuncs[N - I - 1];
      (*AEE.m_Func)(AEE.m_Arg);
    }
  }

  const char* Interpreter::getVersion() const {
    return "$Id$";
  }

  void Interpreter::handleFrontendOptions() {
    if (m_Opts.ShowVersion) {
      llvm::outs() << getVersion() << '\n';
    }
    if (m_Opts.Help) {
      m_Opts.PrintHelp();
    }
  }
   
  void Interpreter::AddIncludePath(llvm::StringRef incpath)
  {
    // Add the given path to the list of directories in which the interpreter
    // looks for include files. Only one path item can be specified at a
    // time, i.e. "path1:path2" is not supported.
      
    CompilerInstance* CI = getCI();
    HeaderSearchOptions& headerOpts = CI->getHeaderSearchOpts();
    const bool IsUserSupplied = false;
    const bool IsFramework = false;
    const bool IsSysRootRelative = true;
    headerOpts.AddPath(incpath, frontend::Angled, IsUserSupplied, IsFramework, 
                       IsSysRootRelative);
      
    Preprocessor& PP = CI->getPreprocessor();
    ApplyHeaderSearchOptions(PP.getHeaderSearchInfo(), headerOpts,
                                    PP.getLangOpts(),
                                    PP.getTargetInfo().getTriple());      
  }

  // Copied from clang/lib/Frontend/CompilerInvocation.cpp
  void Interpreter::DumpIncludePath() {
    const HeaderSearchOptions Opts(getCI()->getHeaderSearchOpts());
    std::vector<std::string> Res;
    if (Opts.Sysroot != "/") {
      Res.push_back("-isysroot");
      Res.push_back(Opts.Sysroot);
    }

    /// User specified include entries.
    for (unsigned i = 0, e = Opts.UserEntries.size(); i != e; ++i) {
      const HeaderSearchOptions::Entry &E = Opts.UserEntries[i];
      if (E.IsFramework && (E.Group != frontend::Angled || !E.IsUserSupplied))
        llvm::report_fatal_error("Invalid option set!");
      if (E.IsUserSupplied) {
        switch (E.Group) {
        case frontend::After:
          Res.push_back("-idirafter");
          break;
        
        case frontend::Quoted:
          Res.push_back("-iquote");
          break;
        
        case frontend::System:
          Res.push_back("-isystem");
          break;
        
        case frontend::IndexHeaderMap:
          Res.push_back("-index-header-map");
          Res.push_back(E.IsFramework? "-F" : "-I");
          break;
        
        case frontend::CSystem:
          Res.push_back("-c-isystem");
          break;

        case frontend::CXXSystem:
          Res.push_back("-cxx-isystem");
          break;

        case frontend::ObjCSystem:
          Res.push_back("-objc-isystem");
          break;

        case frontend::ObjCXXSystem:
          Res.push_back("-objcxx-isystem");
          break;
        
        case frontend::Angled:
          Res.push_back(E.IsFramework ? "-F" : "-I");
          break;
        }
      } else {
        if (E.Group != frontend::Angled && E.Group != frontend::System)
          llvm::report_fatal_error("Invalid option set!");
        Res.push_back(E.Group == frontend::Angled ? "-iwithprefixbefore" :
                      "-iwithprefix");
      }
      Res.push_back(E.Path);
    }

    if (!Opts.ResourceDir.empty()) {
      Res.push_back("-resource-dir");
      Res.push_back(Opts.ResourceDir);
    }
    if (!Opts.ModuleCachePath.empty()) {
      Res.push_back("-fmodule-cache-path");
      Res.push_back(Opts.ModuleCachePath);
    }
    if (!Opts.UseStandardSystemIncludes)
      Res.push_back("-nostdinc");
    if (!Opts.UseStandardCXXIncludes)
      Res.push_back("-nostdinc++");
    if (Opts.UseLibcxx)
      Res.push_back("-stdlib=libc++");
    if (Opts.Verbose)
      Res.push_back("-v");

    // print'em all
    for (unsigned i = 0; i < Res.size(); ++i) {
      llvm::errs() << Res[i] <<"\n";
    }
  }

  CompilerInstance* Interpreter::getCI() const {
    return m_IncrParser->getCI();
  }

  Parser* Interpreter::getParser() const {
    return m_IncrParser->getParser();
  }

  ///\brief Maybe transform the input line to implement cint command line 
  /// semantics (declarations are global) and compile to produce a module.
  ///
  Interpreter::CompilationResult
  Interpreter::process(const std::string& input, Value* V /* = 0 */,
                       const Decl** D /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 1;
    CO.ValuePrinting = CompilationOptions::VPAuto;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();

    if (!canWrapForCall(input))
      return declare(input, D);

    if (Evaluate(input, CO, V) == Interpreter::kFailure) {
      if (D)
        *D = 0;
      return Interpreter::kFailure;
    }

    if (D)
      *D = m_IncrParser->getLastTransaction().getFirstDecl();

    return Interpreter::kSuccess;
  }

  Interpreter::CompilationResult
  Interpreter::declare(const std::string& input, const Decl** D /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;
    CO.DynamicScoping = isDynamicLookupEnabled();
    CO.Debug = isPrintingAST();

    return Declare(input, CO, D);
  }

  Interpreter::CompilationResult
  Interpreter::evaluate(const std::string& input, Value* V /* = 0 */) {
    // Here we might want to enforce further restrictions like: Only one
    // ExprStmt can be evaluated and etc. Such enforcement cannot happen in the
    // worker, because it is used from various places, where there is no such
    // rule
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 0;

    return Evaluate(input, CO, V);
  }

  Interpreter::CompilationResult
  Interpreter::echo(const std::string& input, Value* V /* = 0 */) {
    CompilationOptions CO;
    CO.DeclarationExtraction = 0;
    CO.ValuePrinting = 2;

    return Evaluate(input, CO, V);
  }

  void Interpreter::WrapInput(std::string& input, std::string& fname) {
    fname = createUniqueWrapper();
    input.insert(0, "void " + fname + "() {\n ");
    input.append("\n;\n}");
  }

  llvm::StringRef Interpreter::createUniqueWrapper() {
    const size_t size = sizeof("__cling_Un1Qu3") + sizeof(m_UniqueCounter);
    llvm::SmallString<size> out("__cling_Un1Qu3");
    llvm::raw_svector_ostream(out) << m_UniqueCounter++;

    return (getCI()->getASTContext().Idents.getOwn(out)).getName();
  }

  bool Interpreter::RunFunction(llvm::StringRef fname, llvm::GenericValue* res) {
    if (getCI()->getDiagnostics().hasErrorOccurred())
      return false;

    if (m_IncrParser->isSyntaxOnly()) {
      return true;
    }

    std::string mangledNameIfNeeded;
    FunctionDecl* FD = cast_or_null<FunctionDecl>(LookupDecl(fname).
                                                  getSingleDecl()
                                                  );
    if (FD) {
      if (!FD->isExternC()) {
        llvm::raw_string_ostream RawStr(mangledNameIfNeeded);
        llvm::OwningPtr<MangleContext> 
          Mangle(getCI()->getASTContext().createMangleContext());
        Mangle->mangleName(FD, RawStr);
        RawStr.flush();
        fname = mangledNameIfNeeded;
      }
      m_ExecutionContext->executeFunction(fname, res);
      return true;
    }

    return false;
  }

  void Interpreter::createUniqueName(std::string& out) {
    out = "Un1Qu3";
    llvm::raw_string_ostream(out) << m_UniqueCounter++;
  }

  Interpreter::CompilationResult
  Interpreter::Declare(const std::string& input, const CompilationOptions& CO,
                       const clang::Decl** D /* = 0 */) {

    if (m_IncrParser->Compile(input, CO) != IncrementalParser::kFailed) {
      if (D)
        *D = m_IncrParser->getLastTransaction().getFirstDecl();
      return Interpreter::kSuccess;
    }

    return Interpreter::kFailure;
  }

  Interpreter::CompilationResult
  Interpreter::Evaluate(const std::string& input, const CompilationOptions& CO,
                        Value* V /* = 0 */) {

    Sema& TheSema = getCI()->getSema();

    DiagnosticsEngine& Diag = getCI()->getDiagnostics();
    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    Diag.setDiagnosticMapping(clang::diag::warn_unused_expr,
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(clang::diag::warn_unused_call,
                              clang::diag::MAP_IGNORE, SourceLocation());

    // Wrap the expression
    std::string WrapperName;
    std::string Wrapper = input;
    WrapInput(Wrapper, WrapperName);
    QualType RetTy = getCI()->getASTContext().VoidTy;

    if (V) {
      llvm::SmallVector<clang::DeclGroupRef, 4> DGRs;
      m_IncrParser->Parse(Wrapper, DGRs);
      assert(DGRs.size() && "No decls created by Parse!");

      // Find the wrapper function declaration.
      //
      // Note: The parse may have created a whole set of decls if a template
      //       instantiation happened.  Our wrapper function should be the
      //       last decl in the set.
      //
      FunctionDecl* TopLevelFD 
        = dyn_cast<FunctionDecl>(DGRs.back().getSingleDecl());
      assert(TopLevelFD && "No Decls Parsed?");
      DeclContext* CurContext = TheSema.CurContext;
      TheSema.CurContext = TopLevelFD;
      ASTContext& Context(getCI()->getASTContext());
      // We have to be able to mark the expression for printout. There are three
      // scenarios:
      // 0: Expression printing disabled - don't do anything just disable the 
      //    consumer
      //    is our marker, even if there wasn't missing ';'.
      // 1: Expression printing enabled - make sure we don't have NullStmt, which
      //    is used as a marker to suppress the print out.
      // 2: Expression printing auto - do nothing - rely on the omitted ';' to 
      //    not produce the suppress marker.
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(TopLevelFD->getBody())) {
        // Collect all Stmts, contained in the CompoundStmt
        llvm::SmallVector<Stmt *, 4> Stmts;
        for (CompoundStmt::body_iterator iStmt = CS->body_begin(), 
               eStmt = CS->body_end(); iStmt != eStmt; ++iStmt)
          Stmts.push_back(*iStmt);

        size_t indexOfLastExpr = Stmts.size();
        while(indexOfLastExpr--) {
          // find the trailing expression statement (skip e.g. null statements)
          if (Expr* E = dyn_cast_or_null<Expr>(Stmts[indexOfLastExpr])) {
            RetTy = E->getType();
            if (!RetTy->isVoidType()) {
              // Change the void function's return type
              FunctionProtoType::ExtProtoInfo EPI;
              QualType FuncTy = Context.getFunctionType(RetTy,/* ArgArray = */0,
                                                        /* NumArgs = */0, EPI);
              TopLevelFD->setType(FuncTy);
              // Strip the parenthesis if any
              if (ParenExpr* PE = dyn_cast<ParenExpr>(E))
                E = PE->getSubExpr();

              // Change it with return stmt
              Stmts[indexOfLastExpr] 
                = TheSema.ActOnReturnStmt(SourceLocation(), E).take();
            }
            // even if void: we found an expression
            break;
          }
        }

        // case 1:
        if (CO.ValuePrinting == CompilationOptions::VPEnabled) 
          if (indexOfLastExpr < Stmts.size() - 1 && 
              isa<NullStmt>(Stmts[indexOfLastExpr + 1]))
            Stmts.erase(Stmts.begin() + indexOfLastExpr);
        // Stmts.insert(Stmts.begin() + indexOfLastExpr + 1, 
        //              TheSema.ActOnNullStmt(SourceLocation()).take());

        // Update the CompoundStmt body
        CS->setStmts(TheSema.getASTContext(), Stmts.data(), Stmts.size());

      }

      TheSema.CurContext = CurContext;

      // FIXME: Finish the transaction in better way
      m_IncrParser->Compile("", CO);
    }
    else 
      m_IncrParser->Compile(Wrapper, CO);

    // get the result
    llvm::GenericValue val;
    if (RunFunction(WrapperName, &val)) {
      if (V)
        *V = Value(val, RetTy);

      return Interpreter::kSuccess;
    }

    return Interpreter::kFailure;
  }

  bool Interpreter::loadFile(const std::string& filename,
                             bool allowSharedLib /*=true*/) {
    if (allowSharedLib) {
      llvm::Module* module = m_IncrParser->GetCodeGenerator()->GetModule();
      if (module) {
        if (tryLinker(filename, getOptions(), module))
          return true;
        if (filename.compare(0, 3, "lib") == 0) {
          // starts with "lib", try without (the llvm::Linker forces
          // a "lib" in front, which makes it liblib...
          if (tryLinker(filename.substr(3, std::string::npos),
                        getOptions(), module))
            return true;
        }
      }
    }

    std::string code;
    code += "#include \"" + filename + "\"\n";
    return declare(code) == Interpreter::kSuccess;
  }
  
  QualType
  Interpreter::lookupType(const std::string& typeName)
  {
    //
    //  Our return value.
    //
    QualType TheQT;
    //
    //  Some utilities.
    //
    CompilerInstance* CI = getCI();
    Parser* P = getParser();
    Preprocessor& PP = CI->getPreprocessor();
    //
    //  Tell the diagnostic engine to ignore all diagnostics.
    //
    bool OldSuppressAllDiagnostics =
      PP.getDiagnostics().getSuppressAllDiagnostics();
    PP.getDiagnostics().setSuppressAllDiagnostics(true);
    //
    //  Tell the parser to not attempt spelling correction.
    //
    bool OldSpellChecking = PP.getLangOpts().SpellChecking;
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = 0;
    //
    //  Tell the diagnostic consumer we are switching files.
    //
    DiagnosticConsumer& DClient = CI->getDiagnosticClient();
    DClient.BeginSourceFile(CI->getLangOpts(), &PP);
    //
    //  Create a fake file to parse the type name.
    //
    llvm::MemoryBuffer* SB = llvm::MemoryBuffer::getMemBufferCopy(
      std::string(typeName) + "\n", "lookup.type.by.name.file");
    FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
    //
    //  Turn on ignoring of the main file eof token.
    //
    //  Note: We need this because token readahead in the following
    //        routine calls ends up parsing it multiple times.
    //
    bool ResetIncrementalProcessing = false;
    if (!PP.isIncrementalProcessingEnabled()) {
      ResetIncrementalProcessing = true;
      PP.enableIncrementalProcessing();
    }
    //
    //  Switch to the new file the way #include does.
    //
    //  Note: To switch back to the main file we must consume an eof token.
    //
    PP.EnterSourceFile(FID, 0, SourceLocation());
    PP.Lex(const_cast<Token&>(P->getCurToken()));
    //
    //  Try parsing the type name.
    //
    TypeResult Res(P->ParseTypeName());
    if (Res.isUsable()) {
      // Accept it only if the whole name was parsed.
      if (P->NextToken().getKind() == clang::tok::eof) {
        TheQT = Res.get().get();
      }
    }
    //
    // Advance the parser to the end of the file, and pop the include stack.
    //
    // Note: Consuming the EOF token will pop the include stack.
    //
    P->SkipUntil(tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
      /*StopAtCodeCompletion*/false);
    if (ResetIncrementalProcessing) {
      PP.enableIncrementalProcessing(false);
    }
    DClient.EndSourceFile();
    CI->getDiagnostics().Reset();
    PP.getDiagnostics().setSuppressAllDiagnostics(OldSuppressAllDiagnostics);
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = OldSpellChecking;
    return TheQT;
  }

  Decl*
  Interpreter::lookupClass(const std::string& className)
  {
    //
    //  Our return value.
    //
    Decl* TheDecl = 0;
    //
    //  Some utilities.
    //
    CompilerInstance* CI = getCI();
    Parser* P = getParser();
    Preprocessor& PP = CI->getPreprocessor();
    ASTContext& Context = CI->getASTContext();
    //
    //  Tell the diagnostic engine to ignore all diagnostics.
    //
    bool OldSuppressAllDiagnostics =
      PP.getDiagnostics().getSuppressAllDiagnostics();
    PP.getDiagnostics().setSuppressAllDiagnostics(true);
    //
    //  Tell the parser to not attempt spelling correction.
    //
    bool OldSpellChecking = PP.getLangOpts().SpellChecking;
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = 0;
    //
    //  Tell the diagnostic consumer we are switching files.
    //
    DiagnosticConsumer& DClient = CI->getDiagnosticClient();
    DClient.BeginSourceFile(CI->getLangOpts(), &PP);
    //
    //  Convert the class name to a nested name specifier for parsing.
    //
    std::string classNameAsNNS = className + "::\n";
    //
    //  Create a fake file to parse the class name.
    //
    llvm::MemoryBuffer* SB = llvm::MemoryBuffer::getMemBufferCopy(
      classNameAsNNS, "lookup.class.by.name.file");
    FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
    //
    //  Turn on ignoring of the main file eof token.
    //
    //  Note: We need this because token readahead in the following
    //        routine calls ends up parsing it multiple times.
    //
    bool ResetIncrementalProcessing = false;
    if (!PP.isIncrementalProcessingEnabled()) {
      ResetIncrementalProcessing = true;
      PP.enableIncrementalProcessing();
    }
    //
    //  Switch to the new file the way #include does.
    //
    //  Note: To switch back to the main file we must consume an eof token.
    //
    PP.EnterSourceFile(FID, 0, SourceLocation());
    PP.Lex(const_cast<Token&>(P->getCurToken()));
    //
    //  Try parsing the name as a nested-name-specifier.
    //
    CXXScopeSpec SS;
    if (P->TryAnnotateCXXScopeToken(false)) {
      // error path
      goto lookupClassDone;
    }
    if (P->getCurToken().getKind() == tok::annot_cxxscope) {
      CI->getSema().RestoreNestedNameSpecifierAnnotation(
        P->getCurToken().getAnnotationValue(),
        P->getCurToken().getAnnotationRange(),
        SS);
      if (SS.isValid()) {
        NestedNameSpecifier* NNS = SS.getScopeRep();
        NestedNameSpecifier::SpecifierKind Kind = NNS->getKind();
        // Only accept the parse if we consumed all of the name.
        if (P->NextToken().getKind() == clang::tok::eof) {
          //
          //  Be careful, not all nested name specifiers refer to classes
          //  and namespaces, and those are the only things we want.
          //
          switch (Kind) {
            case NestedNameSpecifier::Identifier: {
                // Dependent type.
                // We do not accept these.
              }
              break;
            case NestedNameSpecifier::Namespace: {
                // Namespace.
                NamespaceDecl* NSD = NNS->getAsNamespace();
                NSD = NSD->getCanonicalDecl();
                TheDecl = NSD;
              }
              break;
            case NestedNameSpecifier::NamespaceAlias: {
                // Namespace alias.
                // Note: In the future, should we return the alias instead? 
                NamespaceAliasDecl* NSAD = NNS->getAsNamespaceAlias();
                NamespaceDecl* NSD = NSAD->getNamespace();
                NSD = NSD->getCanonicalDecl();
                TheDecl = NSD;
              }
              break;
            case NestedNameSpecifier::TypeSpec:
                // Type name.
            case NestedNameSpecifier::TypeSpecWithTemplate: {
                // Type name qualified with "template".
                // Note: Do we need to check for a dependent type here?
                const Type* Ty = NNS->getAsType();
                const TagType* TagTy = Ty->getAs<TagType>();
                if (TagTy) {
                  // It is a class, struct, or union.
                  TagDecl* TD = TagTy->getDecl();
                  if (TD) {
                    // Make sure it is not just forward declared, and
                    // instantiate any templates.
                    if (!CI->getSema().RequireCompleteDeclContext(SS, TD)) {
                      // Success, type is complete, instantiations have
                      // been done.
                      TagDecl* Def = TD->getDefinition();
                      if (Def) {
                        TheDecl = Def;
                      }
                    }
                  }
                }
              }
              break;
            case clang::NestedNameSpecifier::Global: {
                // Name was just "::" and nothing more.
                TheDecl = Context.getTranslationUnitDecl();
              }
              break;
          }
          goto lookupClassDone;
        }
      }
    }
    //
    //  Cleanup after failed parse as a nested-name-specifier.
    //
    P->SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
      /*StopAtCodeCompletion*/false);
    DClient.EndSourceFile();
    CI->getDiagnostics().Reset();
    //
    //  Setup to reparse as a type.
    //
    DClient.BeginSourceFile(CI->getLangOpts(), &PP);
    {
      llvm::MemoryBuffer* SB =
        llvm::MemoryBuffer::getMemBufferCopy(className + "\n",
          "lookup.type.file");
      clang::FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
      CI->getPreprocessor().EnterSourceFile(FID, 0, clang::SourceLocation());
      CI->getPreprocessor().Lex(const_cast<clang::Token&>(P->getCurToken()));
    }
    //
    //  Now try to parse the name as a type.
    //
    if (P->TryAnnotateTypeOrScopeToken(false, false)) {
      // error path
      goto lookupClassDone;
    }
    if (P->getCurToken().getKind() == tok::annot_typename) {
      ParsedType T = Parser::getTypeAnnotation(
        const_cast<Token&>(P->getCurToken()));
      // Only accept the parse if we consumed all of the name.
      if (P->NextToken().getKind() == clang::tok::eof) {
        QualType QT = T.get();
        if (const EnumType* ET = QT->getAs<EnumType>()) {
           EnumDecl* ED = ET->getDecl();
           TheDecl = ED->getDefinition();
        }
      }
    }
  lookupClassDone:
    //
    // Advance the parser to the end of the file, and pop the include stack.
    //
    // Note: Consuming the EOF token will pop the include stack.
    //
    P->SkipUntil(tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
      /*StopAtCodeCompletion*/false);
    if (ResetIncrementalProcessing) {
      PP.enableIncrementalProcessing(false);
    }
    DClient.EndSourceFile();
    CI->getDiagnostics().Reset();
    PP.getDiagnostics().setSuppressAllDiagnostics(OldSuppressAllDiagnostics);
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = OldSpellChecking;
    return TheDecl;
  }

  static
  bool
  FuncArgTypesMatch(CompilerInstance* CI, std::vector<QualType>& GivenArgTypes,
    const FunctionProtoType* FPT)
  {
    FunctionProtoType::arg_type_iterator ATI = FPT->arg_type_begin();
    FunctionProtoType::arg_type_iterator E = FPT->arg_type_end();
    std::vector<QualType>::iterator GAI = GivenArgTypes.begin();
    for (; ATI && (ATI != E); ++ATI, ++GAI) {
      if (!CI->getASTContext().hasSameType(*ATI, *GAI)) {
        return false;
      }
    }
    return true;
  }

  static
  bool
  IsOverload(CompilerInstance* CI,
    const TemplateArgumentListInfo* FuncTemplateArgs,
    std::vector<QualType>& GivenArgTypes, FunctionDecl* FD,
    bool UseUsingDeclRules)
  {
    //FunctionTemplateDecl* FTD = FD->getDescribedFunctionTemplate();
    QualType FQT = CI->getASTContext().getCanonicalType(FD->getType());
    if (llvm::isa<FunctionNoProtoType>(FQT.getTypePtr())) {
      // A K&R-style function (no prototype), is considered to match the args.
      return false;
    }
    const FunctionProtoType* FPT = llvm::cast<FunctionProtoType>(FQT);
    if (
      (GivenArgTypes.size() != FPT->getNumArgs()) ||
      //(GivenArgsAreEllipsis != FPT->isVariadic()) ||
      !FuncArgTypesMatch(CI, GivenArgTypes, FPT)) {
      return true;
    }
    return false;
  }

  Decl*
  Interpreter::lookupFunctionProto(Decl* classDecl,
    const std::string& funcName, const std::string& funcProto)
  {
    //
    //  Our return value.
    //
    Decl* TheDecl = 0;
    //
    //  Some utilities.
    //
    CompilerInstance* CI = getCI();
    Parser* P = getParser();
    Preprocessor& PP = CI->getPreprocessor();
    ASTContext& Context = CI->getASTContext();
    //
    //  Get the DeclContext we will search for the function.
    //
    NestedNameSpecifier* classNNS = 0;
    if (const NamespaceDecl* NSD = llvm::dyn_cast<NamespaceDecl>(classDecl)) {
      classNNS = NestedNameSpecifier::Create(Context, 0,
        const_cast<NamespaceDecl*>(NSD));
    }
    else if (const RecordDecl* RD = llvm::dyn_cast<RecordDecl>(classDecl)) {
      const Type* T = Context.getRecordType(RD).getTypePtr();
      classNNS = NestedNameSpecifier::Create(Context, 0, false, T);
    }
    else if (llvm::isa<TranslationUnitDecl>(classDecl)) {
      classNNS = NestedNameSpecifier::GlobalSpecifier(Context);
    }
    else {
      // Not a namespace or class, we cannot use it.
      return 0;
    }
    DeclContext* foundDC = llvm::dyn_cast<DeclContext>(classDecl);
    //
    //  Tell the diagnostic engine to ignore all diagnostics.
    //
    bool OldSuppressAllDiagnostics =
      PP.getDiagnostics().getSuppressAllDiagnostics();
    PP.getDiagnostics().setSuppressAllDiagnostics(true);
    //
    //  Tell the parser to not attempt spelling correction.
    //
    bool OldSpellChecking = PP.getLangOpts().SpellChecking;
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = 0;
    //
    //  Tell the diagnostic consumer we are switching files.
    //
    DiagnosticConsumer& DClient = CI->getDiagnosticClient();
    DClient.BeginSourceFile(CI->getLangOpts(), &PP);
    //
    //  Create a fake file to parse the prototype.
    //
    llvm::MemoryBuffer* SB = llvm::MemoryBuffer::getMemBufferCopy(
      funcProto + "\n", "func.prototype.file");
    FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
    //
    //  Turn on ignoring of the main file eof token.
    //
    //  Note: We need this because token readahead in the following
    //        routine calls ends up parsing it multiple times.
    //
    bool ResetIncrementalProcessing = false;
    if (!PP.isIncrementalProcessingEnabled()) {
      ResetIncrementalProcessing = true;
      PP.enableIncrementalProcessing();
    }
    //
    //  Switch to the new file the way #include does.
    //
    //  Note: To switch back to the main file we must consume an eof token.
    //
    PP.EnterSourceFile(FID, 0, SourceLocation());
    PP.Lex(const_cast<Token&>(P->getCurToken()));
    //
    //  Parse the prototype now.
    //
    std::vector<QualType> GivenArgTypes;
    std::vector<Expr*> GivenArgs;
    while (P->getCurToken().isNot(tok::eof)) {
      TypeResult Res(P->ParseTypeName());
      if (!Res.isUsable()) {
        // Bad parse, done.
        goto lookupFuncProtoDone;
      }
      clang::QualType QT(Res.get().get());
      QT = QT.getCanonicalType();
      GivenArgTypes.push_back(QT);
      {
        // FIXME: Make an attempt to release these.
        clang::QualType NonRefQT(QT.getNonReferenceType());
        Expr* val = new (Context) OpaqueValueExpr(SourceLocation(), NonRefQT,
          Expr::getValueKindForType(NonRefQT));
        GivenArgs.push_back(val);
      }
      // Type names should be comma separated.
      if (!P->getCurToken().is(clang::tok::comma)) {
        break;
      }
      // Eat the comma.
      P->ConsumeToken();
    }
    if (P->getCurToken().isNot(tok::eof)) {
      // We did not consume all of the prototype, bad parse.
      goto lookupFuncProtoDone;
    }
    //
    //  Cleanup after prototype parse.
    //
    P->SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
      /*StopAtCodeCompletion*/false);
    DClient.EndSourceFile();
    CI->getDiagnostics().Reset();
    //
    //  Create a fake file to parse the function name.
    //
    {
      llvm::MemoryBuffer* SB = llvm::MemoryBuffer::getMemBufferCopy(
        funcName + "\n", "lookup.funcname.file");
      clang::FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
      CI->getPreprocessor().EnterSourceFile(FID, 0, clang::SourceLocation());
      CI->getPreprocessor().Lex(const_cast<clang::Token&>(P->getCurToken()));
    }
    {
      //
      //  Parse the function name.
      //
      SourceLocation TemplateKWLoc;
      UnqualifiedId FuncId;
      CXXScopeSpec SS;
      SS.MakeTrivial(Context, classNNS, SourceRange());
      //
      //  Make the class we are looking up the function
      //  in the current scope to please the constructor
      //  name lookup.  We do not need to do this otherwise,
      //  and may be able to remove it in the future if
      //  the way constructors are looked up changes.
      //
      //  Note:  We cannot use P->EnterScope(Scope::DeclScope)
      //         and P->ExitScope() because they do things
      //         we do not want to happen.
      //
      Scope* OldScope = CI->getSema().CurScope;
      CI->getSema().CurScope = new Scope(OldScope, Scope::DeclScope,
        PP.getDiagnostics());
      CI->getSema().EnterDeclaratorContext(P->getCurScope(), foundDC);
      if (P->ParseUnqualifiedId(SS, /*EnteringContext*/false,
          /*AllowDestructorName*/true,
          /*AllowConstructorName*/true,
          clang::ParsedType(), TemplateKWLoc, FuncId)) {
        // Bad parse.
        // Destroy the scope we created first, and
        // restore the original.
        CI->getSema().ExitDeclaratorContext(P->getCurScope());
        delete CI->getSema().CurScope;
        CI->getSema().CurScope = OldScope;
        // Then cleanup and exit.
        goto lookupFuncProtoDone;
      }
      //
      //  Get any template args in the function name.
      //
      TemplateArgumentListInfo FuncTemplateArgsBuffer;
      DeclarationNameInfo FuncNameInfo;
      const TemplateArgumentListInfo* FuncTemplateArgs;
      CI->getSema().DecomposeUnqualifiedId(FuncId, FuncTemplateArgsBuffer,
        FuncNameInfo, FuncTemplateArgs);
      //
      //  Lookup the function name in the given class now.
      //
      DeclarationName FuncName = FuncNameInfo.getName();
      SourceLocation FuncNameLoc = FuncNameInfo.getLoc();
      LookupResult Result(CI->getSema(), FuncName, FuncNameLoc,
        Sema::LookupMemberName, Sema::ForRedeclaration);
      if (!CI->getSema().LookupQualifiedName(Result, foundDC)) {
        // Lookup failed.
        // Destroy the scope we created first, and
        // restore the original.
        CI->getSema().ExitDeclaratorContext(P->getCurScope());
        delete CI->getSema().CurScope;
        CI->getSema().CurScope = OldScope;
        // Then cleanup and exit.
        goto lookupFuncProtoDone;
      }
      // Destroy the scope we created, and
      // restore the original.
      CI->getSema().ExitDeclaratorContext(P->getCurScope());
      delete CI->getSema().CurScope;
      CI->getSema().CurScope = OldScope;
      //
      //  Check for lookup failure.
      //
      if (!(Result.getResultKind() == LookupResult::Found) &&
          !(Result.getResultKind() == LookupResult::FoundOverloaded)) {
        // Lookup failed.
        goto lookupFuncProtoDone;
      }
      //
      //  Now that we have a set of matching function names
      //  in the class, we have to choose the one being asked
      //  for given the passed template args and prototype.
      //
      for (LookupResult::iterator I = Result.begin(), E = Result.end();
          I != E; ++I) {
        NamedDecl* ND = *I;
        //
        //  Check if this decl is from a using decl, it will not
        //  be a match in some cases.
        //
        bool IsUsingDecl = false;
        if (llvm::isa<UsingShadowDecl>(ND)) {
          IsUsingDecl = true;
          ND = llvm::cast<UsingShadowDecl>(ND)->getTargetDecl();
        }
        //
        //  If found declaration was introduced by a using declaration,
        //  we'll need to use slightly different rules for matching.
        //  Essentially, these rules are the normal rules, except that
        //  function templates hide function templates with different
        //  return types or template parameter lists.
        //
        bool UseMemberUsingDeclRules = IsUsingDecl && foundDC->isRecord();
        if (FunctionTemplateDecl* FTD =
            llvm::dyn_cast<FunctionTemplateDecl>(ND)) {
          // This decl is a function template.
          //
          //  Do template argument deduction and function argument matching.
          //
          FunctionDecl* Specialization;
          sema::TemplateDeductionInfo TDI(Context, SourceLocation());
          Sema::TemplateDeductionResult TDR =
            CI->getSema().DeduceTemplateArguments(FTD,
              const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
              llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
              Specialization, TDI);
          if (TDR == Sema::TDK_Success) {
            // We have a template argument match and func arg match.
            TheDecl = Specialization;
            break;
          }
        } else if (FunctionDecl* FD = llvm::dyn_cast<FunctionDecl>(ND)) {
          // This decl is a function.
          //
          //  Do function argument matching.
          //
          if (!IsOverload(CI, FuncTemplateArgs, GivenArgTypes, FD,
              UseMemberUsingDeclRules)) {
            // We have a function argument match.
            if (UseMemberUsingDeclRules && IsUsingDecl) {
              // But it came from a using decl and we are
              // looking up a class member func, ignore it.
              continue;
            }
            TheDecl = *I;
            break;
          }
        }
      }
    }
  lookupFuncProtoDone:
    //
    // Advance the parser to the end of the file, and pop the include stack.
    //
    // Note: Consuming the EOF token will pop the include stack.
    //
    P->SkipUntil(tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
      /*StopAtCodeCompletion*/false);
    if (ResetIncrementalProcessing) {
      PP.enableIncrementalProcessing(false);
    }
    DClient.EndSourceFile();
    CI->getDiagnostics().Reset();
    PP.getDiagnostics().setSuppressAllDiagnostics(OldSuppressAllDiagnostics);
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = OldSpellChecking;
    return TheDecl;
  }

  Decl*
  Interpreter::lookupFunctionArgs(Decl* classDecl,
    const std::string& funcName, const std::string& funcArgs)
  {
    //
    //  Our return value.
    //
    Decl* TheDecl = 0;
    //
    //  Some utilities.
    //
    CompilerInstance* CI = getCI();
    Parser* P = getParser();
    Preprocessor& PP = CI->getPreprocessor();
    ASTContext& Context = CI->getASTContext();
    //
    //  Tell the diagnostic engine to ignore all diagnostics.
    //
    bool OldSuppressAllDiagnostics =
      PP.getDiagnostics().getSuppressAllDiagnostics();
    PP.getDiagnostics().setSuppressAllDiagnostics(true);
    //
    //  Convert the passed decl into a nested name specifier,
    //  a scope spec, and a decl context.
    //
    NestedNameSpecifier* classNNS = 0;
    if (const NamespaceDecl* NSD = llvm::dyn_cast<NamespaceDecl>(classDecl)) {
      classNNS = NestedNameSpecifier::Create(Context, 0,
        const_cast<NamespaceDecl*>(NSD));
    }
    else if (const RecordDecl* RD = llvm::dyn_cast<RecordDecl>(classDecl)) {
      const Type* T = Context.getRecordType(RD).getTypePtr();
      classNNS = NestedNameSpecifier::Create(Context, 0, false, T);
    }
    else if (llvm::isa<TranslationUnitDecl>(classDecl)) {
      classNNS = NestedNameSpecifier::GlobalSpecifier(Context);
    }
    else {
      // Not a namespace or class, we cannot use it.
      return 0;
    }
    CXXScopeSpec SS;
    SS.MakeTrivial(Context, classNNS, SourceRange());
    DeclContext* foundDC = llvm::dyn_cast<DeclContext>(classDecl);
    //
    //  Some validity checks on the passed decl.
    //
    if (foundDC->isDependentContext()) {
      // Passed decl is a template, we cannot use it.
      return 0;
    }
    if (CI->getSema().RequireCompleteDeclContext(SS, foundDC)) {
      // Forward decl or instantiation failure, we cannot use it.
      return 0;
    }
    //
    //  Get ready for arg list parsing.
    //
    std::vector<QualType> GivenArgTypes;
    std::vector<Expr*> GivenArgs;
    //
    //  If we are looking up a member function, construct
    //  the implicit object argument.
    //
    //  Note: For now this is always a non-CV qualified lvalue.
    //
    QualType ClassType;
    Expr* ObjExpr = 0;
    Expr::Classification ObjExprClassification;
    if (CXXRecordDecl* CRD = dyn_cast<CXXRecordDecl>(foundDC)) {
      ClassType = Context.getTypeDeclType(CRD).getCanonicalType();
      ObjExpr = new (Context) OpaqueValueExpr(SourceLocation(),
        ClassType, VK_LValue);
      ObjExprClassification = ObjExpr->Classify(Context);
      //GivenArgTypes.insert(GivenArgTypes.begin(), ClassType);
      //GivenArgs.insert(GivenArgs.begin(), ObjExpr);
    }
    //
    //  Tell the parser to not attempt spelling correction.
    //
    bool OldSpellChecking = PP.getLangOpts().SpellChecking;
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = 0;
    //
    //  Tell the diagnostic consumer we are switching files.
    //
    DiagnosticConsumer& DClient = CI->getDiagnosticClient();
    DClient.BeginSourceFile(CI->getLangOpts(), &PP);
    //
    //  Create a fake file to parse the arguments.
    //
    llvm::MemoryBuffer* SB = llvm::MemoryBuffer::getMemBufferCopy(
      funcArgs + "\n", "func.args.file");
    FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
    //
    //  Turn on ignoring of the main file eof token.
    //
    //  Note: We need this because token readahead in the following
    //        routine calls ends up parsing it multiple times.
    //
    bool ResetIncrementalProcessing = false;
    if (!PP.isIncrementalProcessingEnabled()) {
      ResetIncrementalProcessing = true;
      PP.enableIncrementalProcessing();
    }
    //
    //  Switch to the new file the way #include does.
    //
    //  Note: To switch back to the main file we must consume an eof token.
    //
    PP.EnterSourceFile(FID, 0, SourceLocation());
    PP.Lex(const_cast<Token&>(P->getCurToken()));
    //
    //  Parse the arguments now.
    //
    {
      PrintingPolicy Policy(Context.getPrintingPolicy());
      Policy.SuppressTagKeyword = true;
      Policy.SuppressUnwrittenScope = true;
      Policy.SuppressInitializers = true;
      Policy.AnonymousTagLocations = false;
      std::string proto;
      {
        bool first_time = true;
        while (P->getCurToken().isNot(tok::eof)) {
          ExprResult Res = P->ParseAssignmentExpression();
          if (Res.isUsable()) {
            Expr* expr = Res.release();
            GivenArgs.push_back(expr);
            QualType QT = expr->getType().getCanonicalType();
            QualType NonRefQT(QT.getNonReferenceType());
            GivenArgTypes.push_back(NonRefQT);
            if (first_time) {
              first_time = false;
            }
            else {
              proto += ',';
            }
            std::string empty;
            llvm::raw_string_ostream tmp(empty);
            expr->printPretty(tmp, Context, /*PrinterHelper=*/0,
              Policy, /*Indentation=*/0);
            proto += tmp.str();
            fprintf(stderr, "%s\n", proto.c_str());
          }
          if (!P->getCurToken().is(tok::comma)) {
            break;
          }
          P->ConsumeToken();
        }
      }
    }
    if (P->getCurToken().isNot(tok::eof)) {
      // We did not consume all of the arg list, bad parse.
      goto lookupFuncArgsDone;
    }
    {
      //
      //  Cleanup after the arg list parse.
      //
      P->SkipUntil(clang::tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
        /*StopAtCodeCompletion*/false);
      DClient.EndSourceFile();
      CI->getDiagnostics().Reset();
      //
      //  Create a fake file to parse the function name.
      //
      {
        llvm::MemoryBuffer* SB = llvm::MemoryBuffer::getMemBufferCopy(
          funcName + "\n", "lookup.funcname.file");
        clang::FileID FID = CI->getSourceManager().createFileIDForMemBuffer(SB);
        CI->getPreprocessor().EnterSourceFile(FID, 0, clang::SourceLocation());
        CI->getPreprocessor().Lex(const_cast<clang::Token&>(P->getCurToken()));
      }
      //
      //  Make the class we are looking up the function
      //  in the current scope to please the constructor
      //  name lookup.  We do not need to do this otherwise,
      //  and may be able to remove it in the future if
      //  the way constructors are looked up changes.
      //
      //  Note:  We cannot use P->EnterScope(Scope::DeclScope)
      //         and P->ExitScope() because they do things
      //         we do not want to happen.
      //
      Scope* OldScope = CI->getSema().CurScope;
      CI->getSema().CurScope = new Scope(OldScope, Scope::DeclScope,
        PP.getDiagnostics());
      CI->getSema().EnterDeclaratorContext(P->getCurScope(), foundDC);
      //
      //  Parse the function name.
      //
      SourceLocation TemplateKWLoc;
      UnqualifiedId FuncId;
      if (P->ParseUnqualifiedId(SS, /*EnteringContext*/false,
          /*AllowDestructorName*/true, /*AllowConstructorName*/true,
          ParsedType(), TemplateKWLoc, FuncId)) {
        // Failed parse, cleanup.
        CI->getSema().ExitDeclaratorContext(P->getCurScope());
        delete CI->getSema().CurScope;
        CI->getSema().CurScope = OldScope;
        goto lookupFuncArgsDone;
      }
      //
      //  Get any template args in the function name.
      //
      TemplateArgumentListInfo FuncTemplateArgsBuffer;
      DeclarationNameInfo FuncNameInfo;
      const TemplateArgumentListInfo* FuncTemplateArgs;
      CI->getSema().DecomposeUnqualifiedId(FuncId, FuncTemplateArgsBuffer,
        FuncNameInfo, FuncTemplateArgs);
      //
      //  Lookup the function name in the given class now.
      //
      DeclarationName FuncName = FuncNameInfo.getName();
      SourceLocation FuncNameLoc = FuncNameInfo.getLoc();
      LookupResult Result(CI->getSema(), FuncName, FuncNameLoc,
        Sema::LookupMemberName, Sema::ForRedeclaration);
      if (!CI->getSema().LookupQualifiedName(Result, foundDC)) {
        // Lookup failed.
        // Destroy the scope we created first, and
        // restore the original.
        CI->getSema().ExitDeclaratorContext(P->getCurScope());
        delete CI->getSema().CurScope;
        CI->getSema().CurScope = OldScope;
        // Then cleanup and exit.
        goto lookupFuncArgsDone;
      }
      //
      //  Destroy the scope we created, and restore the original.
      //
      CI->getSema().ExitDeclaratorContext(P->getCurScope());
      delete CI->getSema().CurScope;
      CI->getSema().CurScope = OldScope;
      //
      //  Check for lookup failure.
      //
      if (!(Result.getResultKind() == LookupResult::Found) &&
          !(Result.getResultKind() == LookupResult::FoundOverloaded)) {
        // Lookup failed.
        goto lookupFuncArgsDone;
      }
      //
      //  Dump what was found.
      //
      if (Result.getResultKind() == LookupResult::Found) {
        NamedDecl* ND = Result.getFoundDecl();
        std::string buf;
        llvm::raw_string_ostream tmp(buf);
        ND->print(tmp, 0);
        fprintf(stderr, "Found: %s\n", tmp.str().c_str());
      } else if (Result.getResultKind() == LookupResult::FoundOverloaded) {
        fprintf(stderr, "Found overload set!\n");
        Result.print(llvm::outs());
        fprintf(stderr, "\n");
      }
      {
        //
        //  Construct the overload candidate set.
        //
        OverloadCandidateSet Candidates(FuncNameInfo.getLoc());
        for (LookupResult::iterator I = Result.begin(), E = Result.end();
            I != E; ++I) {
          NamedDecl* ND = *I;
          if (FunctionDecl* FD = dyn_cast<FunctionDecl>(ND)) {
            if (isa<CXXMethodDecl>(FD) &&
                !cast<CXXMethodDecl>(FD)->isStatic() &&
                !isa<CXXConstructorDecl>(FD)) {
              // Class method, not static, not a constructor, so has
              // an implicit object argument.
              CXXMethodDecl* MD = cast<CXXMethodDecl>(FD);
              {
                std::string buf;
                llvm::raw_string_ostream tmp(buf);
                MD->print(tmp, 0);
                fprintf(stderr, "Considering method: %s\n",
                  tmp.str().c_str());
              }
              if (FuncTemplateArgs && (FuncTemplateArgs->size() != 0)) {
                // Explicit template args were given, cannot use a plain func.
                fprintf(stderr, "rejected: template args given\n");
                continue;
              }
              CI->getSema().AddMethodCandidate(MD, I.getPair(),
                MD->getParent(),
                /*ObjectType=*/ClassType,
                /*ObjectClassification=*/ObjExprClassification,
                llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                Candidates);
            }
            else {
              {
                std::string buf;
                llvm::raw_string_ostream tmp(buf);
                FD->print(tmp, 0);
                fprintf(stderr, "Considering func: %s\n", tmp.str().c_str());
              }
              const FunctionProtoType* Proto = dyn_cast<FunctionProtoType>(
                FD->getType()->getAs<clang::FunctionType>());
              if (!Proto) {
                // Function has no prototype, cannot do overloading.
                fprintf(stderr, "rejected: no prototype\n");
                continue;
              }
              if (FuncTemplateArgs && (FuncTemplateArgs->size() != 0)) {
                // Explicit template args were given, cannot use a plain func.
                fprintf(stderr, "rejected: template args given\n");
                continue;
              }
              CI->getSema().AddOverloadCandidate(FD, I.getPair(),
                llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                Candidates);
            }
          }
          else if (FunctionTemplateDecl* FTD =
              dyn_cast<FunctionTemplateDecl>(ND)) {
            if (isa<CXXMethodDecl>(FTD->getTemplatedDecl()) &&
                !cast<CXXMethodDecl>(FTD->getTemplatedDecl())->isStatic() &&
                !isa<CXXConstructorDecl>(FTD->getTemplatedDecl())) {
              // Class method template, not static, not a constructor, so has
              // an implicit object argument.
              {
                std::string buf;
                llvm::raw_string_ostream tmp(buf);
                FTD->print(tmp, 0);
                fprintf(stderr, "Considering method template: %s\n",
                  tmp.str().c_str());
              }
              CI->getSema().AddMethodTemplateCandidate(FTD, I.getPair(),
                cast<CXXRecordDecl>(FTD->getDeclContext()),
                const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
                /*ObjectType=*/ClassType,
                /*ObjectClassification=*/ObjExprClassification,
                llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                Candidates);
            }
            else {
              {
                std::string buf;
                llvm::raw_string_ostream tmp(buf);
                FTD->print(tmp, 0);
                fprintf(stderr, "Considering func template: %s\n",
                  tmp.str().c_str());
              }
              CI->getSema().AddTemplateOverloadCandidate(FTD, I.getPair(),
                const_cast<TemplateArgumentListInfo*>(FuncTemplateArgs),
                llvm::makeArrayRef<Expr*>(GivenArgs.data(), GivenArgs.size()),
                Candidates, /*SuppressUserConversions=*/false);
            }
          }
          else {
            {
              std::string buf;
              llvm::raw_string_ostream tmp(buf);
              FD->print(tmp, 0);
              fprintf(stderr, "Considering non-func: %s\n",
                tmp.str().c_str());
              fprintf(stderr, "rejected: not a function\n");
            }
          }
        }
        //
        //  Find the best viable function from the set.
        //
        {
          OverloadCandidateSet::iterator Best;
          OverloadingResult OR = Candidates.BestViableFunction(CI->getSema(),
            Result.getNameLoc(), Best);
          if (OR == OR_Success) {
            TheDecl = Best->Function;
          }
        }
      }
      //
      //  Dump the overloading result.
      //
      if (TheDecl) {
        std::string buf;
        llvm::raw_string_ostream tmp(buf);
        TheDecl->print(tmp, 0);
        fprintf(stderr, "Match: %s\n", tmp.str().c_str());
        TheDecl->dump();
        fprintf(stderr, "\n");
      }
    }
  lookupFuncArgsDone:
    //
    // Advance the parser to the end of the file, and pop the include stack.
    //
    // Note: Consuming the EOF token will pop the include stack.
    //
    P->SkipUntil(tok::eof, /*StopAtSemi*/false, /*DontConsume*/false,
      /*StopAtCodeCompletion*/false);
    if (ResetIncrementalProcessing) {
      PP.enableIncrementalProcessing(false);
    }
    DClient.EndSourceFile();
    CI->getDiagnostics().Reset();
    PP.getDiagnostics().setSuppressAllDiagnostics(OldSuppressAllDiagnostics);
    const_cast<LangOptions&>(PP.getLangOpts()).SpellChecking = OldSpellChecking;
    return TheDecl;
  }

  Interpreter::NamedDeclResult Interpreter::LookupDecl(llvm::StringRef Decl, 
                                                       const DeclContext* Within) {
    if (!Within)
      Within = getCI()->getASTContext().getTranslationUnitDecl();
    return Interpreter::NamedDeclResult(Decl, this, Within);
  }

  void Interpreter::installLazyFunctionCreator(void* (*fp)(const std::string&)) {
    m_ExecutionContext->installLazyFunctionCreator(fp);
  }
  
  Value Interpreter::Evaluate(const char* expr, DeclContext* DC,
                              bool ValuePrinterReq) {
    Sema& TheSema = getCI()->getSema();
    if (!DC)
      DC = TheSema.getASTContext().getTranslationUnitDecl();
    
    // Set up the declaration context
    DeclContext* CurContext;

    CurContext = TheSema.CurContext;
    TheSema.CurContext = DC;

    Value Result;
    if (TheSema.ExternalSource) {
      DynamicIDHandler* DIDH = 
        static_cast<DynamicIDHandler*>(TheSema.ExternalSource);
      DIDH->Callbacks->setEnabled();
      (ValuePrinterReq) ? echo(expr, &Result) : evaluate(expr, &Result);
      DIDH->Callbacks->setEnabled(false);
    }
    else 
      (ValuePrinterReq) ? echo(expr, &Result) : evaluate(expr, &Result);

    TheSema.CurContext = CurContext;

    return Result;
  }

  void Interpreter::setCallbacks(InterpreterCallbacks* C) {
    Sema& S = getCI()->getSema();
    assert(S.ExternalSource && "No ExternalSource set!");
    static_cast<DynamicIDHandler*>(S.ExternalSource)->Callbacks = C;
  }

  void Interpreter::enableDynamicLookup(bool value /*=true*/) {
    m_IncrParser->enableDynamicLookup(value);
  }

  bool Interpreter::isDynamicLookupEnabled() {
    return m_IncrParser->isDynamicLookupEnabled();
  }

  void Interpreter::enablePrintAST(bool print /*=true*/) {
    m_IncrParser->enablePrintAST(print);
    m_PrintAST = !m_PrintAST;
  }

  void Interpreter::runStaticInitializersOnce() const {
    // Forward to ExecutionContext; should not be called by
    // anyone except for IncrementalParser.
    llvm::Module* module = m_IncrParser->GetCodeGenerator()->GetModule();
    m_ExecutionContext->runStaticInitializersOnce(module);
  }

  int Interpreter::CXAAtExit(void (*func) (void*), void* arg, void* dso) {
     // Register a CXAAtExit function
     clang::Decl* LastTLD = m_IncrParser->getLastTopLevelDecl();
     m_AtExitFuncs.push_back(CXAAtExitElement(func, arg, dso, LastTLD));
     return 0; // happiness
  }

  bool Interpreter::addSymbol(const char* symbolName,  void* symbolAddress){
    // Forward to ExecutionContext;
    if (!symbolName || !symbolAddress )
      return false;

    return m_ExecutionContext->addSymbol(symbolName,  symbolAddress);
  }
  
} // namespace cling

