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

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/InterpreterCallbacks.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/FileSystem.h"

#include "TMetaUtils.h"

using namespace clang;
using namespace cling;

class TObject;

// Functions used to forward calls from code compiled with no-rtti to code
// compiled with rtti.
extern "C" {
   void TCling__UpdateListsOnCommitted(const cling::Transaction&, Interpreter*);
   void TCling__UpdateListsOnUnloaded(const cling::Transaction&);
   void TCling__TransactionRollback(const cling::Transaction&);
   void TCling__GetNormalizedContext(const ROOT::TMetaUtils::TNormalizedCtxt*&);
   TObject* TCling__GetObjectAddress(const char *Name, void *&LookupCtx);
   Decl* TCling__GetObjectDecl(TObject *obj);
   int TCling__AutoLoadCallback(const char* className);
   int TCling__AutoParseCallback(const char* className);
   const char* TCling__GetClassSharedLibs(const char* className);
//    int TCling__IsAutoLoadNamespaceCandidate(const char* name);
   int TCling__IsAutoLoadNamespaceCandidate(const clang::NamespaceDecl* name);
   int TCling__CompileMacro(const char *fileName, const char *options);
   void TCling__SplitAclicMode(const char* fileName, std::string &mode,
                  std::string &args, std::string &io, std::string &fname);
   void TCling__LibraryLoadedRTTI(const void* dyLibHandle,
                                  llvm::StringRef canonicalName);
   void TCling__LibraryUnloadedRTTI(const void* dyLibHandle,
                                    llvm::StringRef canonicalName);
}

TClingCallbacks::TClingCallbacks(cling::Interpreter* interp)
   : InterpreterCallbacks(interp),
     fLastLookupCtx(0), fROOTSpecialNamespace(0),
     fFirstRun(true), fIsAutoloading(false), fIsAutoloadingRecursively(false),
     fPPOldFlag(false), fPPChanged(false) {
   Transaction* T = 0;
   m_Interpreter->declare("namespace __ROOT_SpecialObjects{}", &T);
   fROOTSpecialNamespace = dyn_cast<NamespaceDecl>(T->getFirstDecl().getSingleDecl());
}

//pin the vtable here
TClingCallbacks::~TClingCallbacks() {}

void TClingCallbacks::InclusionDirective(clang::SourceLocation sLoc/*HashLoc*/,
                                         const clang::Token &/*IncludeTok*/,
                                         llvm::StringRef FileName,
                                         bool /*IsAngled*/,
                                         clang::CharSourceRange /*FilenameRange*/,
                                         const clang::FileEntry *FE,
                                         llvm::StringRef /*SearchPath*/,
                                         llvm::StringRef /*RelativePath*/,
                                         const clang::Module */*Imported*/) {
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

   if (!IsAutoloadingEnabled() || fIsAutoloadingRecursively || !FileName.endswith(".h")) return;

   std::string localString(FileName.str());

   Sema &SemaR = m_Interpreter->getSema();
   DeclarationName Name = &SemaR.getASTContext().Idents.get(localString.c_str());
   LookupResult RHeader(SemaR, Name, sLoc, Sema::LookupOrdinaryName);

   tryAutoParseInternal(localString, RHeader, SemaR.getCurScope(), FE);
}

// Preprocessor callbacks used to handle special cases like for example:
// #include "myMacro.C+"
//
bool TClingCallbacks::FileNotFound(llvm::StringRef FileName,
                                   llvm::SmallVectorImpl<char> &RecoveryPath) {
   // Method called via Callbacks->FileNotFound(Filename, RecoveryPath)
   // in Preprocessor::HandleIncludeDirective(), initally allowing to
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
         // After we have saved the token reset the current one to
         // something which is safe (semi colon usually means empty decl)
         Token& Tok = const_cast<Token&>(P.getCurToken());
         // We parsed 'include' token. We don't need to restore it, because
         // we provide our own way of handling the entire #include "file.c+"
         // Thus if we reverted the token back to the parser, we are in
         // a trouble.
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
            // complation was successful, let's remember the original
            // preprocessor "include not found" error suppression flag
            if (!fPPChanged)
               fPPOldFlag = PP.GetSuppressIncludeNotFoundError();
            PP.SetSuppressIncludeNotFoundError(true);
            fPPChanged = true;
         }
         return false;
      }
   }
   if (fPPChanged) {
      // restore the original preprocessor "include not found" error
      // suppression flag
      PP.SetSuppressIncludeNotFoundError(fPPOldFlag);
      fPPChanged = false;
   }
   return false;
}

// On a failed lookup we have to try to more things before issuing an error.
// The symbol might need to be loaded by ROOT's autoloading mechanism or
// it might be a ROOT special object.
//
// Try those first and if still failing issue the diagnostics.
//
// returns true when a declaration is found and no error should be emitted.
//
bool TClingCallbacks::LookupObject(LookupResult &R, Scope *S) {
   // Don't do any extra work if an error that is not still recovered occurred.
   if (m_Interpreter->getSema().getDiagnostics().hasErrorOccurred())
      return false;

   if (tryAutoParseInternal(R.getLookupName().getAsString(), R, S))
      return true; // happiness.

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

   if (fIsAutoloadingRecursively)
      return false;

   // Finally try to resolve this name as a dynamic name, i.e delay its
   // resolution for runtime.
   return tryResolveAtRuntimeInternal(R, S);
}

bool TClingCallbacks::LookupObject(const DeclContext* DC, DeclarationName Name) {
   if (!IsAutoloadingEnabled() || fIsAutoloadingRecursively) return false;

   if (Name.getNameKind() != DeclarationName::Identifier) return false;



   // Get the 'lookup' decl context.
   // We need to cast away the constness because we will lookup items of this
   // namespace/DeclContext
   NamespaceDecl* NSD = dyn_cast<NamespaceDecl>(const_cast<DeclContext*>(DC));
   if (!NSD)
      return false;

   if ( !TCling__IsAutoLoadNamespaceCandidate(NSD) )
      return false;

   const DeclContext* primaryDC = NSD->getPrimaryContext();
   if (primaryDC != DC)
      return false;

   Sema &SemaR = m_Interpreter->getSema();
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
      UpdateWithNewDecls(DC, Name, llvm::makeArrayRef(lookupResults.data(),
                                                      lookupResults.size()));
      return true;
   }
   return false;
}

bool TClingCallbacks::LookupObject(clang::TagDecl* Tag) {
   // Clang needs Tag's complete definition. Can we parse it?
   if (fIsAutoloadingRecursively || fIsAutoParsingSuspended) return false;

   if (RecordDecl* RD = dyn_cast<RecordDecl>(Tag)) {
      Sema &SemaR = m_Interpreter->getSema();
      ASTContext& C = SemaR.getASTContext();
      Preprocessor &PP = SemaR.getPreprocessor();
      Parser& P = const_cast<Parser&>(m_Interpreter->getParser());
      Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
      Parser::ParserCurTokRestoreRAII savedCurToken(P);
      // After we have saved the token reset the current one to something which
      // is safe (semi colon usually means empty decl)
      Token& Tok = const_cast<Token&>(P.getCurToken());
      Tok.setKind(tok::semi);

      // We can't PushDeclContext, because we go up and the routine that pops
      // the DeclContext assumes that we drill down always.
      // We have to be on the global context. At that point we are in a
      // wrapper function so the parent context must be the global.
      Sema::ContextAndScopeRAII pushedDCAndS(SemaR, C.getTranslationUnitDecl(),
                                             SemaR.TUScope);

      // Use the Normalized name for the autoload
      std::string Name;
      const ROOT::TMetaUtils::TNormalizedCtxt* tNormCtxt = NULL;
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


// The symbol might be defined in the ROOT class autoloading map so we have to
// try to autoload it first and do secondary lookup to try to find it.
//
// returns true when a declaration is found and no error should be emitted.
// If FileEntry, this is a reacting on a #include and Name is the included
// filename.
//
bool TClingCallbacks::tryAutoParseInternal(llvm::StringRef Name, LookupResult &R,
                                           Scope *S, const FileEntry* FE /*=0*/) {
   Sema &SemaR = m_Interpreter->getSema();

   // Try to autoload first if autoloading is enabled
   if (IsAutoloadingEnabled()) {
     // Avoid tail chasing.
     if (fIsAutoloadingRecursively)
       return false;

     // We should try autoload only for special lookup failures.
     Sema::LookupNameKind kind = R.getLookupKind();
     if (!(kind == Sema::LookupTagName || kind == Sema::LookupOrdinaryName
           || kind == Sema::LookupNestedNameSpecifierName
           || kind == Sema::LookupNamespaceName))
        return false;

     fIsAutoloadingRecursively = true;

     bool lookupSuccess = false;
     if (getenv("ROOT_MODULES")) {
        if (TCling__AutoParseCallback(Name.str().c_str())) {
           lookupSuccess = FE || SemaR.LookupName(R, S);
        }
     }
     else {
        // Save state of the PP
        ASTContext& C = SemaR.getASTContext();
        Preprocessor &PP = SemaR.getPreprocessor();
        Parser& P = const_cast<Parser&>(m_Interpreter->getParser());
        Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
        Parser::ParserCurTokRestoreRAII savedCurToken(P);
        // After we have saved the token reset the current one to something which
        // is safe (semi colon usually means empty decl)
        Token& Tok = const_cast<Token&>(P.getCurToken());
        Tok.setKind(tok::semi);

        // We can't PushDeclContext, because we go up and the routine that pops
        // the DeclContext assumes that we drill down always.
        // We have to be on the global context. At that point we are in a
        // wrapper function so the parent context must be the global.
        Sema::ContextAndScopeRAII pushedDCAndS(SemaR, C.getTranslationUnitDecl(),
                                               SemaR.TUScope);

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
                    fIsAutoloadingRecursively = false;
                    return true;
                 }
              }
           }
        }

        if (TCling__AutoParseCallback(Name.str().c_str())) {
           pushedDCAndS.pop();
           cleanupRAII.pop();
           lookupSuccess = FE || SemaR.LookupName(R, S);
        } else if (FE && TCling__GetClassSharedLibs(Name.str().c_str())) {
           // We are "autoparsing" a header, and the header was not parsed.
           // But its library is known - so we do know about that header.
           // Do the parsing explicitly here, while recursive autoloading is
           // disabled.
           std::string incl = "#include \"";
           incl += FE->getName();
           incl += '"';
           m_Interpreter->declare(incl);
        }
     }

     fIsAutoloadingRecursively = false;

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
   cleanupPPRAII.pop(); // force restroing the cache

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
                              /*TypeSourceInfo*/0, SC_None);
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
   if (!shouldResolveAtRuntime(R, S))
      return false;

   DeclarationName Name = R.getLookupName();
   IdentifierInfo* II = Name.getAsIdentifierInfo();
   SourceLocation Loc = R.getNameLoc();
   Sema& SemaRef = R.getSema();
   ASTContext& C = SemaRef.getASTContext();
   DeclContext* DC = C.getTranslationUnitDecl();
   assert(DC && "Must not be null.");
   VarDecl* Result = VarDecl::Create(C, DC, Loc, Loc, II, C.DependentTy,
                                     /*TypeSourceInfo*/0, SC_None);

   // Annotate the decl to give a hint in cling. FIXME: Current implementation
   // is a gross hack, because TClingCallbacks shouldn't know about
   // EvaluateTSynthesizer at all!

   SourceRange invalidRange;
   Result->addAttr(new (C) AnnotateAttr(invalidRange, C, "__ResolveAtRuntime", 0));
   if (Result) {
      // Here we have the scope but we cannot do Sema::PushDeclContext, because
      // on pop it will try to go one level up, which we don't want.
      Sema::ContextRAII pushedDC(SemaRef, DC);
      R.addDecl(Result);
      //SemaRef.PushOnScopeChains(Result, SemaRef.TUScope, /*Add to ctx*/true);
      // Say that we can handle the situation. Clang should try to recover
      return true;
   }
   // We cannot handle the situation. Give up
   return false;
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
   // Make sure that the failed lookup comes the prompt. Currently, we support
   // only the prompt.

   // Should be disabled with the dynamic scopes.
   if (m_IsRuntime)
      return false;

   if (R.isForRedeclaration())
      return false;

   if (R.getLookupKind() != Sema::LookupOrdinaryName)
      return false;

   if (!isa<FunctionDecl>(R.getSema().CurContext))
      return false;

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
                                                   /*IsDecltypeAuto*/false,
                                                   /*IsDependent*/false),
                                     /*TypeSourceInfo*/0, SC_None);

   if (Result) {
      // Annotate the decl to give a hint in cling.
      // FIXME: We should move this in cling, when we implement turning it on
      // and off.
      SourceRange invalidRange;
      Result->addAttr(new (C) AnnotateAttr(invalidRange, C, "__Auto", 0));

      R.addDecl(Result);
      // Say that we can handle the situation. Clang should try to recover
      return true;
   }
   // We cannot handle the situation. Give up.
   return false;
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

void TClingCallbacks::DeclDeserialized(const clang::Decl* D) {
   if (const RecordDecl* RD = dyn_cast<RecordDecl>(D)) {
      // FIXME: Our autoloading doesn't work (load the library) when the looked
      // up decl is found in the PCH/PCM. We have to do that extra step, which
      // loads the corresponding library when a decl was deserialized.
      //
      // Unfortunatelly we cannot do that with the current implementation,
      // because the library load will pull in the header files of the library
      // as well, even though they are in the PCH/PCM and available.
      (void)RD;//TCling__AutoLoadCallback(RD->getNameAsString().c_str());
   }
}

void TClingCallbacks::LibraryLoaded(const void* dyLibHandle,
                                    llvm::StringRef canonicalName) {
   TCling__LibraryLoadedRTTI(dyLibHandle, canonicalName);
}

void TClingCallbacks::LibraryUnloaded(const void* dyLibHandle,
                                      llvm::StringRef canonicalName) {
   TCling__LibraryUnloadedRTTI(dyLibHandle, canonicalName);
}

