// @(#)root/core/meta:$Id$
// Author: Vassil Vassilev   7/10/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClingCallbacks.h"

#include <ROOT/FoundationUtils.hxx>

#include "cling/Interpreter/DynamicLibraryManager.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/GlobalModuleIndex.h"
#include "clang/Basic/DiagnosticSema.h"

#include "llvm/ExecutionEngine/Orc/Core.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "TClingUtils.h"
#include "ClingRAII.h"

#include <optional>

using namespace clang;
using namespace cling;
using namespace ROOT::Internal;

R__EXTERN int gDebug;
class TObject;

// Functions used to forward calls from code compiled with no-rtti to code
// compiled with rtti.
extern "C" {
   void TCling__UpdateListsOnCommitted(const cling::Transaction&, Interpreter*);
   void TCling__UpdateListsOnUnloaded(const cling::Transaction&);
   void TCling__InvalidateGlobal(const clang::Decl*);
   void TCling__TransactionRollback(const cling::Transaction&);
   void TCling__GetNormalizedContext(const ROOT::TMetaUtils::TNormalizedCtxt*&);
   TObject* TCling__GetObjectAddress(const char *Name, void *&LookupCtx);
   Decl* TCling__GetObjectDecl(TObject *obj);
   int TCling__AutoLoadCallback(const char* className);
   int TCling__AutoParseCallback(const char* className);
   const char* TCling__GetClassSharedLibs(const char* className);
   int TCling__IsAutoLoadNamespaceCandidate(const clang::NamespaceDecl* name);
   int TCling__CompileMacro(const char *fileName, const char *options);
   void TCling__SplitAclicMode(const char* fileName, std::string &mode,
                  std::string &args, std::string &io, std::string &fname);
   int TCling__LoadLibrary(const char *library);
   bool TCling__LibraryLoadingFailed(const std::string&, const std::string&, bool, bool);
   void TCling__LibraryLoadedRTTI(const void* dyLibHandle,
                                  llvm::StringRef canonicalName);
   void TCling__LibraryUnloadedRTTI(const void* dyLibHandle,
                                    llvm::StringRef canonicalName);
   void TCling__PrintStackTrace();
   void *TCling__ResetInterpreterMutex();
   void TCling__RestoreInterpreterMutex(void *state);
   void *TCling__LockCompilationDuringUserCodeExecution();
   void TCling__UnlockCompilationDuringUserCodeExecution(void *state);
}

class AutoloadLibraryMU : public llvm::orc::MaterializationUnit {
   const TClingCallbacks &fCallbacks;
   std::string fLibrary;
   llvm::orc::SymbolNameVector fSymbols;
public:
   AutoloadLibraryMU(const TClingCallbacks &cb, const std::string &Library, const llvm::orc::SymbolNameVector &Symbols)
      : MaterializationUnit({getSymbolFlagsMap(Symbols), nullptr}), fCallbacks(cb), fLibrary(Library), fSymbols(Symbols)
   {
   }

   StringRef getName() const override { return "<Symbols from Autoloaded Library>"; }

   void materialize(std::unique_ptr<llvm::orc::MaterializationResponsibility> R) override
   {
      if (!fCallbacks.IsAutoLoadingEnabled()) {
         R->failMaterialization();
         return;
      }

      llvm::orc::SymbolMap loadedSymbols;
      llvm::orc::SymbolNameSet failedSymbols;
      bool loadedLibrary = false;

      for (auto symbol : fSymbols) {
         std::string symbolStr = (*symbol).str();
         std::string nameForDlsym = ROOT::TMetaUtils::DemangleNameForDlsym(symbolStr);

         // Check if the symbol is available without loading the library.
         void *addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(nameForDlsym);

         if (!addr && !loadedLibrary) {
            // Try to load the library which should provide the symbol definition.
            // TODO: Should this interface with the DynamicLibraryManager directly?
            if (TCling__LoadLibrary(fLibrary.c_str()) < 0) {
               ROOT::TMetaUtils::Error("AutoloadLibraryMU", "Failed to load library %s", fLibrary.c_str());
            }

            // Only try loading the library once.
            loadedLibrary = true;

            addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(nameForDlsym);
         }

         if (addr) {
            loadedSymbols[symbol] =
               llvm::JITEvaluatedSymbol(llvm::pointerToJITTargetAddress(addr), llvm::JITSymbolFlags::Exported);
         } else {
            // Collect all failing symbols, delegate their responsibility and then
            // fail their materialization. R->defineNonExistent() sounds like it
            // should do that, but it's not implemented?!
            failedSymbols.insert(symbol);
         }
      }

      if (!failedSymbols.empty()) {
         auto failingMR = R->delegate(failedSymbols);
         if (failingMR) {
            (*failingMR)->failMaterialization();
         }
      }

      if (!loadedSymbols.empty()) {
         llvm::cantFail(R->notifyResolved(loadedSymbols));
         llvm::cantFail(R->notifyEmitted());
      }
   }

   void discard(const llvm::orc::JITDylib &JD, const llvm::orc::SymbolStringPtr &Name) override {}

private:
   static llvm::orc::SymbolFlagsMap getSymbolFlagsMap(const llvm::orc::SymbolNameVector &Symbols)
   {
      llvm::orc::SymbolFlagsMap map;
      for (auto symbolName : Symbols)
         map[symbolName] = llvm::JITSymbolFlags::Exported;
      return map;
   }
};

class AutoloadLibraryGenerator : public llvm::orc::DefinitionGenerator {
   const TClingCallbacks &fCallbacks;
   cling::Interpreter *fInterpreter;
public:
   AutoloadLibraryGenerator(cling::Interpreter *interp, const TClingCallbacks& cb)
      : fCallbacks(cb), fInterpreter(interp) {}

   llvm::Error tryToGenerate(llvm::orc::LookupState &LS, llvm::orc::LookupKind K, llvm::orc::JITDylib &JD,
                             llvm::orc::JITDylibLookupFlags JDLookupFlags,
                             const llvm::orc::SymbolLookupSet &Symbols) override
   {
      if (!fCallbacks.IsAutoLoadingEnabled())
         llvm::Error::success();

      // If we get here, the symbols have not been found in the current process,
      // so no need to check that again. Instead search for the library that
      // provides the symbol and create one MaterializationUnit per library to
      // actually load it if needed.
      std::unordered_map<std::string, llvm::orc::SymbolNameVector> found;
      llvm::orc::SymbolNameSet missing;

      // TODO: Do we need to take gInterpreterMutex?
      // R__LOCKGUARD(gInterpreterMutex);

      for (auto &&KV : Symbols) {
         llvm::orc::SymbolStringPtr name = KV.first;

         const cling::DynamicLibraryManager &DLM = *fInterpreter->getDynamicLibraryManager();

         std::string libName = DLM.searchLibrariesForSymbol((*name).str(),
                                                            /*searchSystem=*/true);

         // libNew overrides memory management functions; must never autoload that.
         assert(libName.find("/libNew.") == std::string::npos && "We must not autoload libNew!");

         // libCling symbols are intentionally hidden from the process, and libCling must not be
         // dlopened. Instead, symbols must be resolved by specifically querying the dynlib handle of
         // libCling, which by definition is loaded - else we could not call this code. The handle
         // is made available as argument to `CreateInterpreter`.
         assert(libName.find("/libCling.") == std::string::npos && "Must not autoload libCling!");

         if (libName.empty())
            missing.insert(name);
         else
            found[libName].push_back(name);
      }

      for (auto &&KV : found) {
         auto MU = std::make_unique<AutoloadLibraryMU>(fCallbacks, KV.first, std::move(KV.second));
         if (auto Err = JD.define(MU))
            return Err;
      }

      if (!missing.empty())
         return llvm::make_error<llvm::orc::SymbolsNotFound>(
            JD.getExecutionSession().getSymbolStringPool(), std::move(missing));

      return llvm::Error::success();
   }
};

TClingCallbacks::TClingCallbacks(cling::Interpreter *interp, bool hasCodeGen) : InterpreterCallbacks(interp)
{
   if (hasCodeGen) {
      Transaction* T = nullptr;
      m_Interpreter->declare("namespace __ROOT_SpecialObjects{}", &T);
      fROOTSpecialNamespace = dyn_cast<NamespaceDecl>(T->getFirstDecl().getSingleDecl());

      interp->addGenerator(std::make_unique<AutoloadLibraryGenerator>(interp, *this));
   }
}

//pin the vtable here
TClingCallbacks::~TClingCallbacks() {}

void TClingCallbacks::InclusionDirective(clang::SourceLocation sLoc/*HashLoc*/,
                                         const clang::Token &/*IncludeTok*/,
                                         llvm::StringRef FileName,
                                         bool /*IsAngled*/,
                                         clang::CharSourceRange /*FilenameRange*/,
                                         clang::OptionalFileEntryRef FE,
                                         llvm::StringRef /*SearchPath*/,
                                         llvm::StringRef /*RelativePath*/,
                                         const clang::Module * Imported,
                                         clang::SrcMgr::CharacteristicKind FileType) {
   // We found a module. Do not try to do anything else.
   Sema &SemaR = m_Interpreter->getSema();
   if (Imported) {
      // FIXME: We should make the module visible at that point.
      if (!SemaR.isModuleVisible(Imported))
         ROOT::TMetaUtils::Info("TClingCallbacks::InclusionDirective",
                                "Module %s resolved but not visible!", Imported->Name.c_str());
      else
        return;
   }

   // Method called via Callbacks->InclusionDirective()
   // in Preprocessor::HandleIncludeDirective(), invoked whenever an
   // inclusion directive has been processed, and allowing us to try
   // to autoload libraries using their header file name.
   // Two strategies are tried:
   // 1) The header name is looked for in the list of autoload keys
   // 2) Heurists are applied to the header name to distill a classname.
   //    For example try to autoload TGClient (libGui) when seeing #include "TGClient.h"
   //    or TH1F in presence of TH1F.h.
   // Strategy 2) is tried only if 1) fails.

   bool isHeaderFile = FileName.endswith(".h") || FileName.endswith(".hxx") || FileName.endswith(".hpp");
   if (!IsAutoLoadingEnabled() || fIsAutoLoadingRecursively || !isHeaderFile)
      return;

   std::string localString(FileName.str());

   DeclarationName Name = &SemaR.getASTContext().Idents.get(localString.c_str());
   LookupResult RHeader(SemaR, Name, sLoc, Sema::LookupOrdinaryName);

   tryAutoParseInternal(localString, RHeader, SemaR.getCurScope(), FE);
}

// TCling__LibraryLoadingFailed is a function in TCling which handles errmessage
bool TClingCallbacks::LibraryLoadingFailed(const std::string& errmessage, const std::string& libStem,
    bool permanent, bool resolved) {
  return TCling__LibraryLoadingFailed(errmessage, libStem, permanent, resolved);
}

// Preprocessor callbacks used to handle special cases like for example:
// #include "myMacro.C+"
//
bool TClingCallbacks::FileNotFound(llvm::StringRef FileName) {
   // Method called via Callbacks->FileNotFound(Filename)
   // in Preprocessor::HandleIncludeDirective(), initially allowing to
   // change the include path, and allowing us to compile code via ACLiC
   // when specifying #include "myfile.C+", and suppressing the preprocessor
   // error message:
   // input_line_23:1:10: fatal error: 'myfile.C+' file not found

   Preprocessor& PP = m_Interpreter->getCI()->getPreprocessor();

   // remove any trailing "\n
   std::string filename(FileName.str().substr(0,FileName.str().find_last_of('"')));
   std::string fname, mode, arguments, io;
   // extract the filename and ACliC mode
   TCling__SplitAclicMode(filename.c_str(), mode, arguments, io, fname);
   if (mode.length() > 0) {
      if (llvm::sys::fs::exists(fname)) {
         // format the CompileMacro() option string
         std::string options = "k";
         if (mode.find("++") != std::string::npos) options += "f";
         if (mode.find("g")  != std::string::npos) options += "g";
         if (mode.find("O")  != std::string::npos) options += "O";

         // Save state of the preprocessor
         Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
         Parser& P = const_cast<Parser&>(m_Interpreter->getParser());
         // We parsed 'include' token. Store it.
         clang::Parser::ParserCurTokRestoreRAII fSavedCurToken(P);
         // We provide our own way of handling the entire #include "file.c+"
         // After we have saved the token reset the current one to
         // something which is safe (semi colon usually means empty decl)
         Token& Tok = const_cast<Token&>(P.getCurToken());
         Tok.setKind(tok::semi);
         // We can't PushDeclContext, because we go up and the routine that pops
         // the DeclContext assumes that we drill down always.
         // We have to be on the global context. At that point we are in a
         // wrapper function so the parent context must be the global.
         // This is needed to solve potential issues when using #include "myFile.C+"
         // after a scope declaration like:
         // void Check(TObject* obj) {
         //   if (obj) cout << "Found the referenced object\n";
         //   else cout << "Error: Could not find the referenced object\n";
         // }
         // #include "A.C+"
         Sema& SemaR = m_Interpreter->getSema();
         ASTContext& C = SemaR.getASTContext();
         Sema::ContextAndScopeRAII pushedDCAndS(SemaR, C.getTranslationUnitDecl(),
                                                SemaR.TUScope);
         int retcode = TCling__CompileMacro(fname.c_str(), options.c_str());
         if (retcode) {
            // compilation was successful, tell the preprocess to silently
            // skip the file
            return true;
         }
      }
   }
   return false;
}


static bool topmostDCIsFunction(Scope* S) {
   if (!S)
      return false;

   DeclContext* DC = S->getEntity();
   // For DeclContext-less scopes like if (dyn_expr) {}
   // Find the DC enclosing S.
   while (!DC) {
      S = S->getParent();
      DC = S->getEntity();
   }

   // DynamicLookup only happens inside topmost functions:
   clang::DeclContext* MaybeTU = DC;
   while (MaybeTU && !isa<TranslationUnitDecl>(MaybeTU)) {
      DC = MaybeTU;
      MaybeTU = MaybeTU->getParent();
   }
   return isa<FunctionDecl>(DC);
}

// On a failed lookup we have to try to more things before issuing an error.
// The symbol might need to be loaded by ROOT's AutoLoading mechanism or
// it might be a ROOT special object.
//
// Try those first and if still failing issue the diagnostics.
//
// returns true when a declaration is found and no error should be emitted.
//
bool TClingCallbacks::LookupObject(LookupResult &R, Scope *S) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   // Don't do any extra work if an error that is not still recovered occurred.
   if (m_Interpreter->getSema().getDiagnostics().hasErrorOccurred())
      return false;

   if (tryAutoParseInternal(R.getLookupName().getAsString(), R, S))
      return true; // happiness.

   // The remaining lookup routines only work on global scope functions
   // ("macros"), not in classes, namespaces etc - anything that looks like
   // it has seen any trace of software development.
   if (!topmostDCIsFunction(S))
      return false;

   // If the autoload wasn't successful try ROOT specials.
   if (tryFindROOTSpecialInternal(R, S))
      return true;

   // For backward-compatibility with CINT we must support stmts like:
   // x = 4; y = new MyClass();
   // I.e we should "inject" a C++11 auto keyword in front of "x" and "y"
   // This has to have higher precedence than the dynamic scopes. It is claimed
   // that if one assigns to a name and the lookup of that name fails if *must*
   // auto keyword must be injected and the stmt evaluation must not be delayed
   // until runtime.
   // For now supported only at the prompt.
   if (tryInjectImplicitAutoKeyword(R, S)) {
      return true;
   }

   if (fIsAutoLoadingRecursively)
      return false;

   // Finally try to resolve this name as a dynamic name, i.e delay its
   // resolution for runtime.
   return tryResolveAtRuntimeInternal(R, S);
}

bool TClingCallbacks::findInGlobalModuleIndex(DeclarationName Name, bool loadFirstMatchOnly /*=true*/)
{
   std::optional<std::string> envUseGMI = llvm::sys::Process::GetEnv("ROOT_USE_GMI");
   if (envUseGMI.has_value())
      if (!envUseGMI->empty() && !ROOT::FoundationUtils::ConvertEnvValueToBool(*envUseGMI))
         return false;

   const CompilerInstance *CI = m_Interpreter->getCI();
   const LangOptions &LangOpts = CI->getPreprocessor().getLangOpts();

   if (!LangOpts.Modules)
      return false;

   // We are currently building a module, we should not import .
   if (LangOpts.isCompilingModule())
      return false;

   if (fIsCodeGening)
      return false;

   // We are currently instantiating one (or more) templates. At that point,
   // all Decls are present in the AST (with possibly deserialization pending),
   // and we should not load more modules which could find an implicit template
   // instantiation that is lazily loaded.
   Sema &SemaR = m_Interpreter->getSema();
   if (SemaR.InstantiatingSpecializations.size() > 0)
      return false;

   GlobalModuleIndex *Index = CI->getASTReader()->getGlobalIndex();
   if (!Index)
      return false;

   // FIXME: We should load only the first available and rely on other callbacks
   // such as RequireCompleteType and LookupUnqualified to load all.
   GlobalModuleIndex::FileNameHitSet FoundModules;

   // Find the modules that reference the identifier.
   // Note that this only finds top-level modules.
   if (Index->lookupIdentifier(Name.getAsString(), FoundModules)) {
      for (llvm::StringRef FileName : FoundModules) {
         StringRef ModuleName = llvm::sys::path::stem(FileName);

         // Skip to the first not-yet-loaded module.
         if (m_LoadedModuleFiles.count(FileName)) {
            if (gDebug > 2)
               llvm::errs() << "Module '" << ModuleName << "' already loaded"
                            << " for '" << Name.getAsString() << "'\n";
            continue;
         }

         fIsLoadingModule = true;
         if (gDebug > 2)
            llvm::errs() << "Loading '" << ModuleName << "' on demand"
                         << " for '" << Name.getAsString() << "'\n";

         m_Interpreter->loadModule(ModuleName.str());
         fIsLoadingModule = false;
         m_LoadedModuleFiles[FileName] = Name;
         if (loadFirstMatchOnly)
            break;
      }
      return true;
   }
   return false;
}

bool TClingCallbacks::LookupObject(const DeclContext* DC, DeclarationName Name) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   if (fIsLoadingModule)
      return false;

   if (fIsAutoParsingSuspended || fIsAutoLoadingRecursively)
      return false;

   if (Name.getNameKind() != DeclarationName::Identifier)
      return false;

   Sema &SemaR = m_Interpreter->getSema();
   auto *D = cast<Decl>(DC);
   SourceLocation Loc = D->getLocation();
   if (Loc.isValid() && SemaR.getSourceManager().isInSystemHeader(Loc)) {
      // This declaration comes from a system module, we do not want to try
      // autoparsing it and find instantiations in our ROOT modules.
      return false;
   }

   // Get the 'lookup' decl context.
   // We need to cast away the constness because we will lookup items of this
   // namespace/DeclContext
   NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(const_cast<DeclContext*>(DC));

   // When GMI is mixed with rootmaps, we might have a name for two different
   // entities provided by the two systems. In that case check if the rootmaps
   // registered the enclosing namespace as a rootmap name resolution namespace
   // and only if that was not the case use the information in the GMI.
   if (!NSD || !TCling__IsAutoLoadNamespaceCandidate(NSD)) {
      // After loading modules, we must update the redeclaration chains.
      return findInGlobalModuleIndex(Name, /*loadFirstMatchOnly*/ false) && D->getMostRecentDecl();
   }

   const DeclContext* primaryDC = NSD->getPrimaryContext();
   if (primaryDC != DC)
      return false;

   LookupResult R(SemaR, Name, SourceLocation(), Sema::LookupOrdinaryName);
   R.suppressDiagnostics();
   // We need the qualified name for TCling to find the right library.
   std::string qualName
      = NSD->getQualifiedNameAsString() + "::" + Name.getAsString();


   // We want to avoid qualified lookups, because they are expensive and
   // difficult to construct. This is why we *artificially* push a scope and
   // a decl context, where Sema should do the lookup.
   clang::Scope S(SemaR.TUScope, clang::Scope::DeclScope, SemaR.getDiagnostics());
   S.setEntity(const_cast<DeclContext*>(DC));
   Sema::ContextAndScopeRAII pushedDCAndS(SemaR, const_cast<DeclContext*>(DC), &S);

   if (tryAutoParseInternal(qualName, R, SemaR.getCurScope())) {
      llvm::SmallVector<NamedDecl*, 4> lookupResults;
      for(LookupResult::iterator I = R.begin(), E = R.end(); I < E; ++I)
         lookupResults.push_back(*I);
      UpdateWithNewDecls(DC, Name, llvm::ArrayRef(lookupResults.data(), lookupResults.size()));
      return true;
   }
   return false;
}

bool TClingCallbacks::LookupObject(clang::TagDecl* Tag) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   if (fIsLoadingModule)
      return false;

   // Clang needs Tag's complete definition. Can we parse it?
   if (fIsAutoLoadingRecursively || fIsAutoParsingSuspended) return false;

   // if (findInGlobalModuleIndex(Tag->getDeclName(), /*loadFirstMatchOnly*/false))
   //    return true;

   Sema &SemaR = m_Interpreter->getSema();

   SourceLocation Loc = Tag->getLocation();
   if (SemaR.getSourceManager().isInSystemHeader(Loc)) {
      // This declaration comes from a system module, we do not want to try
      // autoparsing it and find instantiations in our ROOT modules.
      return false;
   }

   for (auto ReRD: Tag->redecls()) {
      // Don't autoparse a TagDecl while we are parsing its definition!
      if (ReRD->isBeingDefined())
         return false;
   }


   if (RecordDecl* RD = dyn_cast<RecordDecl>(Tag)) {
      ASTContext& C = SemaR.getASTContext();
      Parser& P = const_cast<Parser&>(m_Interpreter->getParser());

      ParsingStateRAII raii(P,SemaR);

      // Use the Normalized name for the autoload
      std::string Name;
      const ROOT::TMetaUtils::TNormalizedCtxt* tNormCtxt = nullptr;
      TCling__GetNormalizedContext(tNormCtxt);
      ROOT::TMetaUtils::GetNormalizedName(Name,
                                          C.getTypeDeclType(RD),
                                          *m_Interpreter,
                                          *tNormCtxt);
      // Autoparse implies autoload
      if (TCling__AutoParseCallback(Name.c_str())) {
         // We have read it; remember that.
         Tag->setHasExternalLexicalStorage(false);
         return true;
      }
   }
   return false;
}


// The symbol might be defined in the ROOT class AutoLoading map so we have to
// try to autoload it first and do secondary lookup to try to find it.
//
// returns true when a declaration is found and no error should be emitted.
// If FileEntry, this is a reacting on a #include and Name is the included
// filename.
//
bool TClingCallbacks::tryAutoParseInternal(llvm::StringRef Name, LookupResult &R,
                                           Scope *S, clang::OptionalFileEntryRef FE) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   Sema &SemaR = m_Interpreter->getSema();

   // Try to autoload first if AutoLoading is enabled
   if (IsAutoLoadingEnabled()) {
     // Avoid tail chasing.
     if (fIsAutoLoadingRecursively)
       return false;

     // We should try autoload only for special lookup failures.
     Sema::LookupNameKind kind = R.getLookupKind();
     if (!(kind == Sema::LookupTagName || kind == Sema::LookupOrdinaryName
           || kind == Sema::LookupNestedNameSpecifierName
           || kind == Sema::LookupNamespaceName))
        return false;

     fIsAutoLoadingRecursively = true;

     bool lookupSuccess = false;
     // Save state of the PP
     Parser &P = const_cast<Parser &>(m_Interpreter->getParser());

     ParsingStateRAII raii(P, SemaR);

     // First see whether we have a fwd decl of this name.
     // We shall only do that if lookup makes sense for it (!FE).
     if (!FE) {
        lookupSuccess = SemaR.LookupName(R, S);
        if (lookupSuccess) {
           if (R.isSingleResult()) {
              if (isa<clang::RecordDecl>(R.getFoundDecl())) {
                 // Good enough; RequireCompleteType() will tell us if we
                 // need to auto parse.
                 // But we might need to auto-load.
                 TCling__AutoLoadCallback(Name.data());
                 fIsAutoLoadingRecursively = false;
                 return true;
              }
           }
        }
     }

     if (TCling__AutoParseCallback(Name.str().c_str())) {
        // Shouldn't we pop more?
        raii.fPushedDCAndS.pop();
        raii.fCleanupRAII.pop();
        lookupSuccess = FE || SemaR.LookupName(R, S);
     } else if (FE && TCling__GetClassSharedLibs(Name.str().c_str())) {
        // We are "autoparsing" a header, and the header was not parsed.
        // But its library is known - so we do know about that header.
        // Do the parsing explicitly here, while recursive AutoLoading is
        // disabled.
        std::string incl = "#include \"";
        incl += FE->getName();
        incl += '"';
        m_Interpreter->declare(incl);
     }

     fIsAutoLoadingRecursively = false;

     if (lookupSuccess)
       return true;
   }

   return false;
}

// If cling cannot find a name it should ask ROOT before it issues an error.
// If ROOT knows the name then it has to create a new variable with that name
// and type in dedicated for that namespace (eg. __ROOT_SpecialObjects).
// For example if the interpreter is looking for h in h-Draw(), this routine
// will create
// namespace __ROOT_SpecialObjects {
//   THist* h = (THist*) the_address;
// }
//
// Later if h is called again it again won't be found by the standart lookup
// because it is in our hidden namespace (nobody should do using namespace
// __ROOT_SpecialObjects). It caches the variable declarations and their
// last address. If the newly found decl with the same name (h) has different
// address than the cached one it goes directly at the address and updates it.
//
// returns true when declaration is found and no error should be emitted.
//
bool TClingCallbacks::tryFindROOTSpecialInternal(LookupResult &R, Scope *S) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   // User must be able to redefine the names that come from a file.
   if (R.isForRedeclaration())
      return false;
   // If there is a result abort.
   if (!R.empty())
      return false;
   const Sema::LookupNameKind LookupKind = R.getLookupKind();
   if (LookupKind != Sema::LookupOrdinaryName)
      return false;


   Sema &SemaR = m_Interpreter->getSema();
   ASTContext& C = SemaR.getASTContext();
   Preprocessor &PP = SemaR.getPreprocessor();
   DeclContext *CurDC = SemaR.CurContext;
   DeclarationName Name = R.getLookupName();

   // Make sure that the failed lookup comes from a function body.
   if(!CurDC || !CurDC->isFunctionOrMethod())
      return false;

   // Save state of the PP, because TCling__GetObjectAddress may induce nested
   // lookup.
   Preprocessor::CleanupAndRestoreCacheRAII cleanupPPRAII(PP);
   TObject *obj = TCling__GetObjectAddress(Name.getAsString().c_str(),
                                           fLastLookupCtx);
   cleanupPPRAII.pop(); // force restoring the cache

   if (obj) {

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
      // Register the address in TCling::fgSetOfSpecials
      // to speed-up the execution of TCling::RecursiveRemove when
      // the object is not a special.
      // See http://root.cern.ch/viewvc/trunk/core/meta/src/TCint.cxx?view=log#rev18109
      if (!fgSetOfSpecials) {
         fgSetOfSpecials = new std::set<TObject*>;
      }
      ((std::set<TObject*>*)fgSetOfSpecials)->insert((TObject*)*obj);
#endif
#endif

     VarDecl *VD = cast_or_null<VarDecl>(utils::Lookup::Named(&SemaR, Name,
                                                        fROOTSpecialNamespace));
      if (VD) {
         //TODO: Check for same types.
         GlobalDecl GD(VD);
         TObject **address = (TObject**)m_Interpreter->getAddressOfGlobal(GD);
         // Since code was generated already we cannot rely on the initializer
         // of the decl in the AST, however we will update that init so that it
         // will be easier while debugging.
         CStyleCastExpr *CStyleCast = cast<CStyleCastExpr>(VD->getInit());
         Expr* newInit = utils::Synthesize::IntegerLiteralExpr(C, (uint64_t)obj);
         CStyleCast->setSubExpr(newInit);

         // The actual update happens here, directly in memory.
         *address = obj;
      }
      else {
         // Save state of the PP
         Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);

         const Decl *TD = TCling__GetObjectDecl(obj);
         // We will declare the variable as pointer.
         QualType QT = C.getPointerType(C.getTypeDeclType(cast<TypeDecl>(TD)));

         VD = VarDecl::Create(C, fROOTSpecialNamespace, SourceLocation(),
                              SourceLocation(), Name.getAsIdentifierInfo(), QT,
                              /*TypeSourceInfo*/nullptr, SC_None);
         // Build an initializer
         Expr* Init
           = utils::Synthesize::CStyleCastPtrExpr(&SemaR, QT, (uint64_t)obj);
         // Register the decl in our hidden special namespace
         VD->setInit(Init);
         fROOTSpecialNamespace->addDecl(VD);

         cling::CompilationOptions CO;
         CO.DeclarationExtraction = 0;
         CO.ValuePrinting = CompilationOptions::VPDisabled;
         CO.ResultEvaluation = 0;
         CO.DynamicScoping = 0;
         CO.Debug = 0;
         CO.CodeGeneration = 1;

         cling::Transaction* T = new cling::Transaction(CO, SemaR);
         T->append(VD);
         T->setState(cling::Transaction::kCompleted);

         m_Interpreter->emitAllDecls(T);
      }
      assert(VD && "Cannot be null!");
      R.addDecl(VD);
      return true;
   }

   return false;
}

bool TClingCallbacks::tryResolveAtRuntimeInternal(LookupResult &R, Scope *S) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   if (!shouldResolveAtRuntime(R, S))
      return false;

   DeclarationName Name = R.getLookupName();
   IdentifierInfo* II = Name.getAsIdentifierInfo();
   SourceLocation Loc = R.getNameLoc();
   Sema& SemaRef = R.getSema();
   ASTContext& C = SemaRef.getASTContext();
   DeclContext* TU = C.getTranslationUnitDecl();
   assert(TU && "Must not be null.");

   // DynamicLookup only happens inside wrapper functions:
   clang::FunctionDecl* Wrapper = nullptr;
   Scope* Cursor = S;
   do {
      DeclContext* DCCursor = Cursor->getEntity();
      if (DCCursor == TU)
         return false;
      Wrapper = dyn_cast_or_null<FunctionDecl>(DCCursor);
      if (Wrapper) {
         if (utils::Analyze::IsWrapper(Wrapper)) {
            break;
         } else {
            // Can't have a function inside the wrapper:
            return false;
         }
      }
   } while ((Cursor = Cursor->getParent()));

   if (!Wrapper) {
      // The parent of S wasn't the TU?!
      return false;
   }

   VarDecl* Result = VarDecl::Create(C, TU, Loc, Loc, II, C.DependentTy,
                                     /*TypeSourceInfo*/nullptr, SC_None);

   if (!Result) {
      // We cannot handle the situation. Give up
      return false;
   }

   // Annotate the decl to give a hint in cling. FIXME: Current implementation
   // is a gross hack, because TClingCallbacks shouldn't know about
   // EvaluateTSynthesizer at all!

   Wrapper->addAttr(AnnotateAttr::CreateImplicit(C, "__ResolveAtRuntime"));

   // Here we have the scope but we cannot do Sema::PushDeclContext, because
   // on pop it will try to go one level up, which we don't want.
   Sema::ContextRAII pushedDC(SemaRef, TU);
   R.addDecl(Result);
   //SemaRef.PushOnScopeChains(Result, SemaRef.TUScope, /*Add to ctx*/true);
   // Say that we can handle the situation. Clang should try to recover
   return true;
}

bool TClingCallbacks::shouldResolveAtRuntime(LookupResult& R, Scope* S) {
   if (m_IsRuntime)
      return false;

   if (R.getLookupKind() != Sema::LookupOrdinaryName)
      return false;

   if (R.isForRedeclaration())
      return false;

   if (!R.empty())
      return false;

   const Transaction* T = getInterpreter()->getCurrentTransaction();
   if (!T)
      return false;
   const cling::CompilationOptions& COpts = T->getCompilationOpts();
   if (!COpts.DynamicScoping)
      return false;

   auto &PP = R.getSema().PP;
   // In `foo bar`, `foo` is certainly a type name and must not be resolved. We
   // cannot rely on `PP.LookAhead(0)` as the parser might have already consumed
   // some tokens.
   SourceLocation LocAfterIdent = PP.getLocForEndOfToken(R.getNameLoc());
   Token LookAhead0;
   PP.getRawToken(LocAfterIdent, LookAhead0, /*IgnoreWhiteSpace=*/true);
   if (LookAhead0.is(tok::raw_identifier))
      return false;

   // FIXME: Figure out better way to handle:
   // C++ [basic.lookup.classref]p1:
   //   In a class member access expression (5.2.5), if the . or -> token is
   //   immediately followed by an identifier followed by a <, the
   //   identifier must be looked up to determine whether the < is the
   //   beginning of a template argument list (14.2) or a less-than operator.
   //   The identifier is first looked up in the class of the object
   //   expression. If the identifier is not found, it is then looked up in
   //   the context of the entire postfix-expression and shall name a class
   //   or function template.
   //
   // We want to ignore object(.|->)member<template>
   //if (R.getSema().PP.LookAhead(0).getKind() == tok::less)
      // TODO: check for . or -> in the cached token stream
   //   return false;

   for (Scope* DepScope = S; DepScope; DepScope = DepScope->getParent()) {
      if (DeclContext* Ctx = static_cast<DeclContext*>(DepScope->getEntity())) {
         if (!Ctx->isDependentContext())
            // For now we support only the prompt.
            if (isa<FunctionDecl>(Ctx))
               return true;
      }
   }

   return false;
}

bool TClingCallbacks::tryInjectImplicitAutoKeyword(LookupResult &R, Scope *S) {
   if (!fROOTSpecialNamespace) {
      // init error or rootcling
      return false;
   }

   // Should be disabled with the dynamic scopes.
   if (m_IsRuntime)
      return false;

   if (R.isForRedeclaration())
      return false;

   if (R.getLookupKind() != Sema::LookupOrdinaryName)
      return false;

   if (!isa<FunctionDecl>(R.getSema().CurContext))
      return false;

   {
      // ROOT-8538: only top-most (function-level) scope is supported.
      DeclContext* ScopeDC = S->getEntity();
      if (!ScopeDC || !llvm::isa<FunctionDecl>(ScopeDC))
         return false;

      // Make sure that the failed lookup comes the prompt. Currently, we
      // support only the prompt.
      Scope* FnScope = S->getFnParent();
      if (!FnScope)
         return false;
      auto FD = dyn_cast_or_null<FunctionDecl>(FnScope->getEntity());
      if (!FD || !utils::Analyze::IsWrapper(FD))
         return false;
   }

   Sema& SemaRef = R.getSema();
   ASTContext& C = SemaRef.getASTContext();
   DeclContext* DC = SemaRef.CurContext;
   assert(DC && "Must not be null.");


   Preprocessor& PP = R.getSema().getPreprocessor();
   //Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
   //PP.EnableBacktrackAtThisPos();
   if (PP.LookAhead(0).isNot(tok::equal)) {
      //PP.Backtrack();
      return false;
   }
   //PP.CommitBacktrackedTokens();
   //cleanupRAII.pop();
   DeclarationName Name = R.getLookupName();
   IdentifierInfo* II = Name.getAsIdentifierInfo();
   SourceLocation Loc = R.getNameLoc();
   VarDecl* Result = VarDecl::Create(C, DC, Loc, Loc, II,
                                     C.getAutoType(QualType(),
                                                   clang::AutoTypeKeyword::Auto,
                                                   /*IsDependent*/false),
                                     /*TypeSourceInfo*/nullptr, SC_None);

   if (!Result) {
      ROOT::TMetaUtils::Error("TClingCallbacks::tryInjectImplicitAutoKeyword",
                              "Cannot create VarDecl");
      return false;
   }

   // Annotate the decl to give a hint in cling.
   // FIXME: We should move this in cling, when we implement turning it on
   // and off.
   Result->addAttr(AnnotateAttr::CreateImplicit(C, "__Auto"));

   R.addDecl(Result);

   // Raise a warning when trying to use implicit auto injection feature.
   SemaRef.getDiagnostics().setSeverity(diag::warn_deprecated_message, diag::Severity::Warning, SourceLocation());
   SemaRef.Diag(Loc, diag::warn_deprecated_message)
      << "declaration without the 'auto' keyword" << DC << Loc << FixItHint::CreateInsertion(Loc, "auto ");

   // Say that we can handle the situation. Clang should try to recover
   return true;
}

void TClingCallbacks::Initialize() {
   // Replay existing decls from the AST.
   if (fFirstRun) {
      // Before setting up the callbacks register what cling have seen during init.
      Sema& SemaR = m_Interpreter->getSema();
      cling::Transaction TPrev((cling::CompilationOptions(), SemaR));
      TPrev.append(SemaR.getASTContext().getTranslationUnitDecl());
      TCling__UpdateListsOnCommitted(TPrev, m_Interpreter);

      fFirstRun = false;
   }
}

// The callback is used to update the list of globals in ROOT.
//
void TClingCallbacks::TransactionCommitted(const Transaction &T) {
   if (fFirstRun && T.empty())
      Initialize();

   TCling__UpdateListsOnCommitted(T, m_Interpreter);
}

// The callback is used to update the list of globals in ROOT.
//
void TClingCallbacks::TransactionUnloaded(const Transaction &T) {
   if (T.empty())
      return;

   TCling__UpdateListsOnUnloaded(T);
}

// The callback is used to clear the autoparsing caches.
//
void TClingCallbacks::TransactionRollback(const Transaction &T) {
   if (T.empty())
      return;

   TCling__TransactionRollback(T);
}

void TClingCallbacks::DefinitionShadowed(const clang::NamedDecl *D) {
   TCling__InvalidateGlobal(D);
}

void TClingCallbacks::LibraryLoaded(const void* dyLibHandle,
                                    llvm::StringRef canonicalName) {
   TCling__LibraryLoadedRTTI(dyLibHandle, canonicalName);
}

void TClingCallbacks::LibraryUnloaded(const void* dyLibHandle,
                                      llvm::StringRef canonicalName) {
   TCling__LibraryUnloadedRTTI(dyLibHandle, canonicalName);
}

void TClingCallbacks::PrintStackTrace() {
   TCling__PrintStackTrace();
}

void *TClingCallbacks::EnteringUserCode()
{
   // We can safely assume that if the lock exist already when we are in Cling code,
   // then the lock has (or should been taken) already. Any action (that caused callers
   // to take the lock) is halted during ProcessLine. So it is fair to unlock it.
   return TCling__ResetInterpreterMutex();
}

void TClingCallbacks::ReturnedFromUserCode(void *stateInfo)
{
   TCling__RestoreInterpreterMutex(stateInfo);
}

void *TClingCallbacks::LockCompilationDuringUserCodeExecution()
{
   return TCling__LockCompilationDuringUserCodeExecution();
}

void TClingCallbacks::UnlockCompilationDuringUserCodeExecution(void *StateInfo)
{
   TCling__UnlockCompilationDuringUserCodeExecution(StateInfo);
}
