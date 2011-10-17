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
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Value.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/Linker.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"

#include <cstdio>
#include <iostream>
#include <sstream>

using namespace clang;

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
    assert(!SoFile.isEmpty() && "We know the shared lib exists but can't find it back!");
    std::string errMsg;
    bool err =
      llvm::sys::DynamicLibrary::LoadLibraryPermanently(SoFile.str().c_str(), &errMsg);
    if (err) {
      fprintf(stderr,
              "Interpreter::loadFile: Could not load shared library!\n");
      fprintf(stderr, "%s\n", errMsg.c_str());
      L.releaseModule();
      return false;
    }
  }
  L.releaseModule();
  return true;
}

namespace cling {

  // "Declared" to the JIT in RuntimeUniverse.h
  namespace runtime {
    namespace internal {
      int local_cxa_atexit(void (*func) (void*), void* arg,
                           void* dso, Interpreter* interp) {
        return interp->CXAAtExit(func, arg, dso);
      }
      struct __trigger__cxa_atexit {
        ~__trigger__cxa_atexit(); // implemented in Interpreter.cpp
      };
      __trigger__cxa_atexit::~__trigger__cxa_atexit() {}
    }
  }

  Interpreter::NamedDeclResult::NamedDeclResult(llvm::StringRef Decl, 
                                                Interpreter* interp, 
                                                DeclContext* Within)
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
    DeclContext::lookup_result Lookup = m_CurDeclContext->lookup(Name);
    // FIXME: We need to traverse over each found result in the pair in order to
    // solve possible ambiguities.
    if (Lookup.first != Lookup.second) {
      if (DeclContext* DC = dyn_cast<DeclContext>(*Lookup.first))
        m_CurDeclContext = DC;
      else
        m_CurDeclContext = (*Lookup.first)->getDeclContext();
      
      m_Result = (*Lookup.first);
    }
    else {
      // TODO: Find the template instantiations with using a wrapper (getQualType). 
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

  //
  //  Interpreter
  //
  
  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------
   Interpreter::Interpreter(int argc, const char* const *argv,
                            const char* startupPCH /*= 0*/,
                            const char* llvmdir /*= 0*/):
  m_UniqueCounter(0),
  m_PrintAST(false),
  m_ValuePrinterEnabled(false),
  m_LastDump(0)
  {
    m_PragmaHandler = new PragmaNamespace("cling");

    std::vector<unsigned> LeftoverArgsIdx;
    m_Opts = InvocationOptions::CreateFromArgs(argc, argv, LeftoverArgsIdx);
    std::vector<const char*> LeftoverArgs;

    // We do C++ by default:
    LeftoverArgs.push_back("-x");
    LeftoverArgs.push_back("c++");

    for (size_t I = 0, N = LeftoverArgsIdx.size(); I < N; ++I) {
      LeftoverArgs.push_back(argv[LeftoverArgsIdx[I]]);
    }
 
    m_IncrParser.reset(new IncrementalParser(this, &getPragmaHandler(),
                                             LeftoverArgs.size(), &LeftoverArgs[0],
                                             llvmdir));
    m_ExecutionContext.reset(new ExecutionContext());

    m_ValuePrintStream.reset(new llvm::raw_os_ostream(std::cout));

    // Allow the interpreter to find itself.
    // OBJ first: if it exists it should be more up to date
    AddIncludePath(CLING_SRCDIR_INCL);
    AddIncludePath(CLING_INSTDIR_INCL);

    // Warm them up
    m_IncrParser->Initialize(startupPCH);
    if (m_IncrParser->usingStartupPCH()) {
      processStartupPCH();
    }

    if (getCI()->getLangOpts().CPlusPlus) {
       // Set up the gCling variable - even if we use PCH ('this' is different)
       processLine("#include \"cling/Interpreter/ValuePrinter.h\"\n");
       std::stringstream initializer;
       initializer << "gCling=(cling::Interpreter*)" << (long)this << ";";
       processLine(initializer.str());
    }

    handleFrontendOptions();
  }
  
  //---------------------------------------------------------------------------
  // Destructor
  //---------------------------------------------------------------------------
  Interpreter::~Interpreter()
  {
    for (size_t I = 0, N = m_AtExitFuncs.size(); I < N; ++I) {
      const CXAAtExitElement& AEE = m_AtExitFuncs[N - I - 1];
      (*AEE.m_Func)(AEE.m_Arg);
    }

    //delete m_prev_module; // Don't do this, the engine does it.
  }
   
  const char* Interpreter::getVersion() const {
    return "$Id$";
  }

  void Interpreter::writeStartupPCH() {
    m_IncrParser->writeStartupPCH();
  }

  void Interpreter::handleFrontendOptions() {
    if (m_Opts.ShowVersion) {
      llvm::outs() << getVersion() << '\n';
    }
    if (m_Opts.Help) {
      m_Opts.PrintHelp();
    }
  }

  void Interpreter::processStartupPCH() {
    clang::TranslationUnitDecl* TU = m_IncrParser->getCI()->getASTContext().getTranslationUnitDecl();
    for (clang::DeclContext::decl_iterator D = TU->decls_begin(),
           E = TU->decls_end(); D != E; ++D) {
      // That's probably overestimating
      ++m_UniqueCounter;
      const clang::FunctionDecl* F = dyn_cast<const clang::FunctionDecl>(*D);
      if (F) {
        clang::DeclarationName N = F->getDeclName();
        if (N.isIdentifier()) {
          clang::IdentifierInfo* II = N.getAsIdentifierInfo();
          if (II->getName().find("__cling_Un1Qu3") == 0) {
            RunFunction(II->getName());
          }
        }
      }
    }
  }
   
  void Interpreter::AddIncludePath(const char *incpath)
  {
    // Add the given path to the list of directories in which the interpreter
    // looks for include files. Only one path item can be specified at a
    // time, i.e. "path1:path2" is not supported.
      
    CompilerInstance* CI = getCI();
    HeaderSearchOptions& headerOpts = CI->getHeaderSearchOpts();
    const bool IsUserSupplied = false;
    const bool IsFramework = false;
    const bool IsSysRootRelative = true;
    headerOpts.AddPath (incpath, frontend::Angled, IsUserSupplied, IsFramework, IsSysRootRelative);
      
    Preprocessor& PP = CI->getPreprocessor();
    ApplyHeaderSearchOptions(PP.getHeaderSearchInfo(), headerOpts,
                                    PP.getLangOptions(),
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
  
  Interpreter::CompilationResult
  Interpreter::processLine(const std::string& input_line, 
                           bool rawInput /*= false*/) {
    //
    //  Transform the input line to implement cint
    //  command line semantics (declarations are global),
    //  and compile to produce a module.
    //
    
    std::string functName;
    std::string wrapped = input_line;
    if (strncmp(input_line.c_str(),"#",strlen("#")) != 0 &&
        strncmp(input_line.c_str(),"extern ",strlen("extern ")) != 0 &&
        !rawInput) {
      WrapInput(wrapped, functName);
    }

    DiagnosticsEngine& Diag = getCI()->getDiagnostics();
    // Disable warnings which doesn't make sense when using the prompt
    // This gets reset with the clang::Diagnostics().Reset()
    Diag.setDiagnosticMapping(DiagnosticIDs::getIdFromName("warn_unused_expr"),
                              clang::diag::MAP_IGNORE, SourceLocation());
    Diag.setDiagnosticMapping(DiagnosticIDs::getIdFromName("warn_unused_call"),
                              clang::diag::MAP_IGNORE, SourceLocation());
    CompilationResult Result = handleLine(wrapped, functName);
    return Result;
  }

  void Interpreter::WrapInput(std::string& input, std::string& fname) {
    fname = createUniqueName();
    input.insert(0, "void " + fname + "() {\n ");
    input.append("\n;\n}");
  }

  bool Interpreter::RunFunction(llvm::StringRef fname, llvm::GenericValue* res) {
    if (getCI()->getDiagnostics().hasErrorOccurred())
      return false;

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

  std::string Interpreter::createUniqueName()
  {
    // Create an unique name
    
    std::ostringstream swrappername;
    swrappername << "__cling_Un1Qu3" << m_UniqueCounter++;
    return swrappername.str();
  }
  
  
  Interpreter::CompilationResult
  Interpreter::handleLine(llvm::StringRef input, llvm::StringRef FunctionName) {
    // if we are using the preprocessor
    if (input[0] == '#') {
      if (m_IncrParser->CompileAsIs(input) != IncrementalParser::kFailed)
        return Interpreter::kSuccess;
      else
        return Interpreter::kFailure;
    }

    if (m_IncrParser->CompileLineFromPrompt(input) 
        == IncrementalParser::kFailed)
        return Interpreter::kFailure;
    //
    //  Run it using the JIT.
    //
    // TODO: Handle the case when RunFunction wasn't able to run the function
    bool RunRes = RunFunction(FunctionName);

    if (RunRes)
      return Interpreter::kSuccess;

    return Interpreter::kFailure;
  }
  
  bool
  Interpreter::loadFile(const std::string& filename,
                        const std::string* trailcode /*=0*/,
                        bool allowSharedLib /*=true*/)
  {
    if (allowSharedLib) {
      llvm::Module* module = m_IncrParser->GetCodeGenerator()->GetModule();
      if (module) {
        if (tryLinker(filename, getOptions(), module))
          return 0;
        if (filename.compare(0, 3, "lib") == 0) {
          // starts with "lib", try without (the llvm::Linker forces
          // a "lib" in front, which makes it liblib...
          if (tryLinker(filename.substr(3, std::string::npos),
                        getOptions(), module))
            return 0;
        }
      }
    }
    
    std::string code;
    code += "#include \"" + filename + "\"\n";
    if (trailcode)
      code += *trailcode;
    return (m_IncrParser->CompileAsIs(code) != IncrementalParser::kFailed);
  }
  
  bool
  Interpreter::executeFile(const std::string& fileWithArgs)
  {
    // Look for start of parameters:

    typedef std::pair<llvm::StringRef,llvm::StringRef> StringRefPair;

    StringRefPair pairFileArgs = llvm::StringRef(fileWithArgs).split('(');
    if (pairFileArgs.second.empty()) {
      pairFileArgs.second = ")";
    }
    StringRefPair pairPathFile = pairFileArgs.first.rsplit('/');
    if (pairPathFile.second.empty()) {
       pairPathFile.second = pairPathFile.first;
    }
    StringRefPair pairFuncExt = pairPathFile.second.rsplit('.');

    //fprintf(stderr, "funcname: %s\n", pairFuncExt.first.data());
    
    std::string func;
    std::string wrapper = pairFuncExt.first.str()+"("+pairFileArgs.second.str();
    WrapInput(wrapper, func);

    if (loadFile(pairFileArgs.first, &wrapper)) {
      return RunFunction(func);
    }
    return false;
  }

  Interpreter::NamedDeclResult Interpreter::LookupDecl(llvm::StringRef Decl, 
                                                       DeclContext* Within) {
    if (!Within)
      Within = getCI()->getASTContext().getTranslationUnitDecl();
    return Interpreter::NamedDeclResult(Decl, this, Within);
  }

  void Interpreter::installLazyFunctionCreator(void* (*fp)(const std::string&)) {
    m_ExecutionContext->installLazyFunctionCreator(fp);
  }
  
  Value Interpreter::Evaluate(const char* expr, DeclContext* DC,
                              bool ValuePrinterReq) {
    assert(DC && "DeclContext cannot be null!");

    // Execute and get the result
    Value Result;

    // Wrap the expression
    std::string WrapperName;
    std::string Wrapper = expr;
    WrapInput(Wrapper, WrapperName);
    
    // Set up the declaration context
    DeclContext* CurContext;
    Sema& TheSema = getCI()->getSema();

    CurContext = TheSema.CurContext;
    TheSema.CurContext = DC;

    llvm::SmallVector<clang::DeclGroupRef, 4> DGRs;
    assert(TheSema.ExternalSource && "No ExternalSource set!");

    DynamicIDHandler* DIDH = 
      static_cast<DynamicIDHandler*>(TheSema.ExternalSource);
    DIDH->Callbacks->setEnabled();
    m_IncrParser->Parse(Wrapper, DGRs);
    DIDH->Callbacks->setEnabled(false);

    assert((DGRs.size() || DGRs.size() > 2) && "Only FunctionDecl expected!");

    TheSema.CurContext = CurContext;
    // get the Type
    FunctionDecl* TopLevelFD 
      = dyn_cast<FunctionDecl>(DGRs.front().getSingleDecl());
    CurContext = TheSema.CurContext;
    TheSema.CurContext = TopLevelFD;
    ASTContext& Context(getCI()->getASTContext());
    QualType RetTy;
    if (Stmt* S = TopLevelFD->getBody())
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(S))
        if (Expr* E = dyn_cast_or_null<Expr>(CS->body_back())) {
          RetTy = E->getType();
          if (!RetTy->isVoidType()) {
            // Change the void function's return type
            FunctionProtoType::ExtProtoInfo EPI;
            QualType FuncTy = Context.getFunctionType(RetTy,
                                                      /*ArgArray*/0,
                                                      /*NumArgs*/0,
                                                      EPI);
            TopLevelFD->setType(FuncTy);
            // add return stmt
            Stmt* RetS = TheSema.ActOnReturnStmt(SourceLocation(), E).take();
            CS->setStmts(Context, &RetS, 1);
          }
        }
    TheSema.CurContext = CurContext;

    // FIXME: Finish the transaction in better way
    m_IncrParser->CompileAsIs("");

    // Attach the value printer
    if (ValuePrinterReq) {
      std::string VPAttached = WrapperName + "()";
      WrapInput(VPAttached, WrapperName);
      m_IncrParser->CompileLineFromPrompt(VPAttached);
    }

    // get the result
    llvm::GenericValue val;
    RunFunction(WrapperName, &val);

    return Value(val, RetTy.getTypePtrOrNull());
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

  void Interpreter::dumpAST(bool showAST, int last) {
    Decl* D = m_LastDump;
    PrintingPolicy Policy = m_IncrParser->getCI()->getASTContext().getPrintingPolicy();
    
    if (!D && last == -1 ) {
      fprintf(stderr, "No last dump found! Assuming ALL \n");
      last = 0;
      showAST = false;        
    }
    
    Policy.Dump = showAST;
    
    if (last == -1) {
      while ((D = D->getNextDeclInContext())) {
        D->print(llvm::errs(), Policy);
      }
    }
    else if (last == 0) {
      m_IncrParser->getCI()->getASTContext().getTranslationUnitDecl()->print(llvm::errs(), Policy);
    } else {
      // First Decl to print
      Decl *FD = m_IncrParser->getFirstTopLevelDecl();
      Decl *LD = FD;
      
      // FD and LD are first
      
      Decl *NextLD = 0;
      for (int i = 1; i < last; ++i) {
        NextLD = LD->getNextDeclInContext();
        if (NextLD) {
          LD = NextLD;
        }
      }
      
      // LD is last Decls after FD: [FD x y z LD a b c d]
      
      while ((NextLD = LD->getNextDeclInContext())) {
        // LD not yet at end: move window
        FD = FD->getNextDeclInContext();
        LD = NextLD;
      }
      
      // Now LD is == getLastDeclinContext(), and FD is last decls before
      // LD is last Decls after FD: [x y z a FD b c LD]
      
      while (FD) {
        FD->print(llvm::errs(), Policy);
        fprintf(stderr, "\n"); // New line for every decl
        FD = FD->getNextDeclInContext();
      }        
    }
    
    m_LastDump = m_IncrParser->getLastTopLevelDecl();     
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

  
} // namespace cling
