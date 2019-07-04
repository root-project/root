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

#include <stack>

namespace clang {
   class Decl;
   class LookupResult;
   class NamespaceDecl;
   class Scope;
   class TagDecl;
   class Token;
   class FileEntry;
   class Module;
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
   bool fFirstRun;
   bool fIsAutoLoading;
   bool fIsAutoLoadingRecursively;
   bool fIsAutoParsingSuspended;
   bool fPPOldFlag;
   bool fPPChanged;
   bool fIsCodeGening;

public:
   TClingCallbacks(cling::Interpreter* interp, bool hasCodeGen);

   ~TClingCallbacks();

   void Initialize();

   void SetAutoLoadingEnabled(bool val = true) { fIsAutoLoading = val; }
   bool IsAutoLoadingEnabled() { return fIsAutoLoading; }

   void SetAutoParsingSuspended(bool val = true) { fIsAutoParsingSuspended = val; }
   bool IsAutoParsingSuspended() { return fIsAutoParsingSuspended; }

   virtual bool LibraryLoadingFailed(const std::string&, const std::string&, bool, bool);

   virtual void InclusionDirective(clang::SourceLocation /*HashLoc*/,
                                   const clang::Token &/*IncludeTok*/,
                                   llvm::StringRef FileName,
                                   bool /*IsAngled*/,
                                   clang::CharSourceRange /*FilenameRange*/,
                                   const clang::FileEntry * /*File*/,
                                   llvm::StringRef /*SearchPath*/,
                                   llvm::StringRef /*RelativePath*/,
                                   const clang::Module * /*Imported*/);

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

   // The callback is used to inform ROOT when cling started code generation.
   //
   virtual void TransactionCodeGenStarted(const cling::Transaction &T)
   {
      assert(!fIsCodeGening);
      fIsCodeGening = true;
   }

   // The callback is used to inform ROOT when cling finished code generation.
   //
   virtual void TransactionCodeGenFinished(const cling::Transaction &T)
   {
      assert(fIsCodeGening);
      fIsCodeGening = false;
   }

   // The callback is used to update the list of globals in ROOT.
   //
   virtual void TransactionUnloaded(const cling::Transaction &T);

   // The callback is used to clear the autoparsing caches.
   //
   virtual void TransactionRollback(const cling::Transaction &T);

   ///\brief A previous definition has been shadowed; invalidate TCling' stored
   /// data about the old (global) decl.
   virtual void DefinitionShadowed(const clang::NamedDecl *D);

   // Used to inform client about a new decl read by the ASTReader.
   //
   virtual void DeclDeserialized(const clang::Decl* D);

   virtual void LibraryLoaded(const void* dyLibHandle,
                              llvm::StringRef canonicalName);
   virtual void LibraryUnloaded(const void* dyLibHandle,
                                llvm::StringRef canonicalName);

   virtual void PrintStackTrace();

   virtual void *EnteringUserCode();
   virtual void ReturnedFromUserCode(void *stateInfo);
   virtual void *LockCompilationDuringUserCodeExecution();
   virtual void UnlockCompilationDuringUserCodeExecution(void *StateInfo);

private:
   bool tryAutoParseInternal(llvm::StringRef Name, clang::LookupResult &R,
                            clang::Scope *S, const clang::FileEntry* FE = 0);
   bool tryFindROOTSpecialInternal(clang::LookupResult &R, clang::Scope *S);
   bool tryResolveAtRuntimeInternal(clang::LookupResult &R, clang::Scope *S);
   bool shouldResolveAtRuntime(clang::LookupResult &R, clang::Scope *S);
   bool tryInjectImplicitAutoKeyword(clang::LookupResult &R, clang::Scope *S);
};
