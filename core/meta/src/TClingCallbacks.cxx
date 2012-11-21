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

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"

using namespace clang;
using namespace cling;

class TObject;

// Functions used to forward calls from code compiled with no-rtti to code 
// compiled with rtti.
extern "C" {
   void TCintWithCling__UpdateListsOnCommitted(const cling::Transaction&);
   void TCintWithCling__UpdateListsOnUnloaded(const cling::Transaction&); 
   TObject* TCintWithCling__GetObjectAddress(const char *Name, void *&LookupCtx);
   Decl* TCintWithCling__GetObjectDecl(TObject *obj);
   int TCintWithCling__AutoLoadCallback(const char* className);
}

TClingCallbacks::TClingCallbacks(cling::Interpreter* interp) 
   : InterpreterCallbacks(interp), fLastLookupCtx(0), fROOTSpecialNamespace(0),
     fFirstRun(true), fIsAutoloading(false), fIsAutoloadingRecursively(false) {
   const Decl* D = 0;
   m_Interpreter->declare("namespace __ROOT_SpecialObjects{}", &D);
   fROOTSpecialNamespace = dyn_cast<NamespaceDecl>(const_cast<Decl*>(D));
}

//pin the vtable here
TClingCallbacks::~TClingCallbacks() {}

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
bool TClingCallbacks::LookupObject(LookupResult &R, Scope *S) {

   Sema &SemaR = m_Interpreter->getSema();
   Preprocessor &PP = SemaR.getPreprocessor();
   DeclContext *CurDC = SemaR.CurContext;
   DeclarationName Name = R.getLookupName();

   // Try to autoload first if autoloading is enabled
   if (IsAutoloadingEnabled()) {
     // Avoid tail chasing.
     if (fIsAutoloadingRecursively)
       return false;
     fIsAutoloadingRecursively = true;

     // Save state of the PP
     Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);
     Parser& P = const_cast<Parser&>(m_Interpreter->getParser());
     Parser::ParserCurTokRestoreRAII savedCurToken(P);

     bool oldSuppressDiags = SemaR.getDiagnostics().getSuppressAllDiagnostics();
     SemaR.getDiagnostics().setSuppressAllDiagnostics();
      
     // We can't PushDeclContext, because we go up and the routine that pops 
     // the DeclContext assumes that we drill down always.
     // We have to be on the global context. At that point we are in a 
     // wrapper function so the parent context must be the global.
     Sema::ContextAndScopeRAII pushedSAndDC(SemaR, CurDC, S);

     bool lookupSuccess = false;
     if (TCintWithCling__AutoLoadCallback(Name.getAsString().c_str())) {
       pushedSAndDC.pop();
       cleanupRAII.pop();
       lookupSuccess = SemaR.LookupName(R, S);
     }

     SemaR.getDiagnostics().setSuppressAllDiagnostics(oldSuppressDiags);

     fIsAutoloadingRecursively = false;
   
     if (lookupSuccess)
       return true;
   }

   // If the autoload wasn't successful try ROOT specials.
   // Make sure that the failed lookup comes from the prompt.
   if(!CurDC || !CurDC->isFunctionOrMethod())
      return false;
   if (NamedDecl* ND = dyn_cast<NamedDecl>(CurDC))
      if (!m_Interpreter->isUniqueWrapper(ND->getNameAsString()))
         return false;

   TObject *obj = TCintWithCling__GetObjectAddress(Name.getAsString().c_str(), 
                                                   fLastLookupCtx);
   if (obj) {
      NamedDecl *ND = utils::Lookup::Named(&SemaR, Name, fROOTSpecialNamespace);
      if (ND) {
         TObject **address = (TObject**)m_Interpreter->getAddressOfGlobal(ND);
         *address = obj;
         R.addDecl(ND);
         //TODO: Check for same types.
      }
      else {
         // Save state of the PP
         Preprocessor::CleanupAndRestoreCacheRAII cleanupRAII(PP);

         const Decl *TD = TCintWithCling__GetObjectDecl(obj);
         ASTContext& C = SemaR.getASTContext();
         // We will declare the variable as pointer.
         QualType QT = C.getPointerType(C.getTypeDeclType(cast<TypeDecl>(TD)));
         
         VarDecl *VD = VarDecl::Create(C, fROOTSpecialNamespace, 
                                       SourceLocation(), SourceLocation(),
                                       Name.getAsIdentifierInfo(), QT,
                                       /*TypeSourceInfo*/0, SC_None, SC_None
                                       );
         // Register the decl in our hidden special namespace
         fROOTSpecialNamespace->addDecl(VD);

         cling::CompilationOptions CO;
         CO.DeclarationExtraction = 0;
         CO.ValuePrinting = CompilationOptions::VPDisabled;
         CO.ResultEvaluation = 0;
         CO.DynamicScoping = 0;
         CO.Debug = 0;
         CO.CodeGeneration = 1;

         cling::Transaction T(CO, /*llvm::Module=*/0);
         T.appendUnique(VD);
         T.setCompleted();

         m_Interpreter->codegen(&T);
         // initialize it directly in memory ugly:
         void** address = (void**)m_Interpreter->getAddressOfGlobal(VD);
         *address = obj;
         R.addDecl(VD);
      }
      return true;
   }

   return false; // give up.
}

// The callback is used to update the list of globals in ROOT.
//
void TClingCallbacks::TransactionCommitted(const Transaction &T) {
   if (!T.size())
      return;
   if (fFirstRun) {
      // Before setting up the callbacks register what cling have seen during init.
      const cling::Transaction* T = m_Interpreter->getFirstTransaction();
      while (T) {
         if (T->getState() == cling::Transaction::kCommitted)
            TCintWithCling__UpdateListsOnCommitted(*T);
         T = T->getNext();
      }

      fFirstRun = false;
   }

   TCintWithCling__UpdateListsOnCommitted(T);
}

// The callback is used to update the list of globals in ROOT.
//
void TClingCallbacks::TransactionUnloaded(const Transaction &T) {
   if (!T.size())
      return;

   TCintWithCling__UpdateListsOnUnloaded(T);
}
