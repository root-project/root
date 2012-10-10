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
}

TClingCallbacks::TClingCallbacks(cling::Interpreter* interp) 
   : InterpreterCallbacks(interp), fLastLookupCtx(0), fROOTSpecialNamespace(0) {
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
   // FIXME:
   // Disable the callback until we solve the issue with the parser lookup.
   // Here if we try to do parser lookup we end up in an invalid state.
   return false;


   Sema &SemaR = m_Interpreter->getSema();
   DeclContext *CurDC = SemaR.CurContext;

   // Make sure that the failed lookup comes from the prompt.
   if(!CurDC || !CurDC->isFunctionOrMethod())
      return false;
   if (NamedDecl* ND = dyn_cast<NamedDecl>(CurDC))
      if (!m_Interpreter->isUniqueWrapper(ND->getNameAsString()))
         return false;

   DeclarationName Name = R.getLookupName();
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
         const Decl *TD = TCintWithCling__GetObjectDecl(obj);
         QualType QT = SemaR.getASTContext().getTypeDeclType(cast<TypeDecl>(TD));
         VarDecl *VD = VarDecl::Create(SemaR.getASTContext(), 
                                       fROOTSpecialNamespace,
                                       SourceLocation(),
                                       SourceLocation(),
                                       Name.getAsIdentifierInfo(),
                                       QT,
                                       /*TypeSourceInfo*/0,
                                       SC_None,
                                       SC_None
                                       );
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

   TCintWithCling__UpdateListsOnCommitted(T);
}

// The callback is used to update the list of globals in ROOT.
//
void TClingCallbacks::TransactionUnloaded(const Transaction &T) {
   if (!T.size())
      return;

   TCintWithCling__UpdateListsOnUnloaded(T);
}
