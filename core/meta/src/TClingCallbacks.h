// @(#)root/core/meta:$Id$
// Author: Vassil Vassilev   7/10/2012

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "cling/Interpreter/InterpreterCallbacks.h"

namespace clang {
   class Decl;
   class LookupResult;
   class NamespaceDecl;
   class Scope;
}

namespace cling {
   class Interpreter;
   class Transaction;
}

// The callbacks are used to update the list of globals in ROOT.
//
class TClingCallbacks : public cling::InterpreterCallbacks {
private:
   void *fLastLookupCtx;
   clang::NamespaceDecl *fROOTSpecialNamespace;
   bool fFirstRun;
   bool fIsAutoloading;
   bool fIsAutoloadingRecursively;
public:
   TClingCallbacks(cling::Interpreter* interp);

   ~TClingCallbacks();

   void Initialize(clang::ASTContext& Ctx);

   void SetAutoloadingEnabled(bool val = true) { fIsAutoloading = val; }
   bool IsAutoloadingEnabled() { return fIsAutoloading; }

   virtual bool LookupObject(clang::LookupResult &R, clang::Scope *S);

   // The callback is used to update the list of globals in ROOT.
   //
   virtual void TransactionCommitted(const cling::Transaction &T);

   // The callback is used to update the list of globals in ROOT.
   //
   virtual void TransactionUnloaded(const cling::Transaction &T);

   // Used to inform client about a new decl read by the ASTReader.
   //
   virtual void DeclDeserialized(const clang::Decl* D);

private:
   bool tryAutoloadInternal(clang::LookupResult &R, clang::Scope *S);
   bool tryFindROOTSpecialInternal(clang::LookupResult &R, clang::Scope *S);
   bool tryResolveAtRuntimeInternal(clang::LookupResult &R, clang::Scope *S);
   bool shouldResolveAtRuntime(clang::LookupResult &R, clang::Scope *S);
   bool tryInjectImplicitAutoKeyword(clang::LookupResult &R, clang::Scope *S);
};
