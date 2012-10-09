// @(#)root/core/meta:$Id$
// Author: Vassil Vassilev   7/10/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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
public:
   TClingCallbacks(cling::Interpreter* interp, bool isEnabled = false);

   ~TClingCallbacks();

   virtual bool LookupObject(clang::LookupResult &R, clang::Scope *S);

   // The callback is used to update the list of globals in ROOT.
   //
   virtual void TransactionCommitted(const cling::Transaction &T);

   // The callback is used to update the list of globals in ROOT.
   //
   virtual void TransactionUnloaded(const cling::Transaction &T);
};
