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
   class TagDecl;
}

namespace cling {
   class Interpreter;
   class Transaction;
}

namespace llvm {
   class StringRef;
}

// The callbacks are used to update the list of globals in ROOT.
//
class TClingCallbacks : public cling::InterpreterCallbacks {
private:
   void *fLastLookupCtx;
   clang::NamespaceDecl *fROOTSpecialNamespace;
   clang::NamespaceDecl *fDeclContextToLookIn;
   bool fFirstRun;
   bool fIsAutoloading;
   bool fIsAutoloadingRecursively;
   bool fPPOldFlag;
   bool fPPChanged;
public:
   TClingCallbacks(cling::Interpreter* interp);

   ~TClingCallbacks();

   void Initialize(clang::ASTContext& Ctx);

   void SetAutoloadingEnabled(bool val = true) { fIsAutoloading = val; }
   bool IsAutoloadingEnabled() { return fIsAutoloading; }

   // Preprocessor callbacks used to handle special cases like for example: 
   // #include "myMacro.C+"
   //
   virtual bool FileNotFound(llvm::StringRef FileName, 
                             llvm::SmallVectorImpl<char>& RecoveryPath);

   virtual bool LookupObject(clang::LookupResult &R, clang::Scope *S);
   virtual bool LookupObject(const clang::DeclContext* DC, 
                             clang::DeclarationName Name);
   virtual bool LookupObject(clang::TagDecl* Tag);

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
   bool tryAutoloadInternal(llvm::StringRef Name, clang::LookupResult &R, 
                            clang::Scope *S);
   bool tryFindROOTSpecialInternal(clang::LookupResult &R, clang::Scope *S);
   bool tryResolveAtRuntimeInternal(clang::LookupResult &R, clang::Scope *S);
   bool shouldResolveAtRuntime(clang::LookupResult &R, clang::Scope *S);
   bool tryInjectImplicitAutoKeyword(clang::LookupResult &R, clang::Scope *S);
};
