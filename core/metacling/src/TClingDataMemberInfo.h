// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingDataMemberInfo
#define ROOT_TClingDataMemberInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingDataMemberInfo                                                 //
//                                                                      //
// Emulation of the CINT DataMemberInfo class.                          //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the data members of a class through the DataMemberInfo class.  This  //
// class provides the same functionality, using an interface as close   //
// as possible to DataMemberInfo but the data member metadata comes     //
// from the Clang C++ compiler, not CINT.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingClassInfo.h"

#include "TClingDeclInfo.h"

#include "cling/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/CompilerInstance.h"

#include <vector>
#include <string>

namespace clang {
   class Decl;
   class ValueDecl;
}

namespace ROOT {
   namespace TMetaUtils {
      class TNormalizedCtxt;
   }
}

class TClingClassInfo;

class TClingDataMemberInfo final : public TClingDeclInfo {

private:

   cling::Interpreter    *fInterp;    // Cling interpreter, we do *not* own.
   TClingClassInfo       *fClassInfo; // Class we are iterating over, we own.
   bool                   fFirstTime; // We need to skip the first increment to support the cint Next() semantics.
   clang::DeclContext::decl_iterator fIter; // Current decl.
   std::vector<clang::DeclContext::decl_iterator> fIterStack; // Recursion stack for traversing nested transparent scopes.
   std::string            fTitle; // The meta info for the member.

   llvm::SmallVector<clang::DeclContext *, 2>   fContexts; // Set of DeclContext that we will iterate over.

   unsigned int                                 fContextIdx; // Index in fContexts of DeclContext we are iterating over.
   mutable std::string fIoType;
   mutable std::string fIoName;
   union {
      float fFloat;
      double fDouble;
      long fLong;
   } fConstInitVal; // Result of VarDecl::evaluateValue()
   inline void CheckForIoTypeAndName () const;

public:

   ~TClingDataMemberInfo() { delete fClassInfo; }

   explicit TClingDataMemberInfo(cling::Interpreter *interp)
   : TClingDeclInfo(nullptr), fInterp(interp), fClassInfo(0), fFirstTime(true), fContextIdx(0U)
   {
      fClassInfo = new TClingClassInfo(fInterp);
      fIter = fInterp->getCI()->getASTContext().getTranslationUnitDecl()->decls_begin();
      // Move to first global variable.
      InternalNext();
   }

   TClingDataMemberInfo(cling::Interpreter *, TClingClassInfo *);

   // Takes concrete decl and disables the iterator.
   // ValueDecl is the common base between enum constant, var decl and field decl
   TClingDataMemberInfo(cling::Interpreter *, const clang::ValueDecl *, TClingClassInfo *);

   TClingDataMemberInfo(const TClingDataMemberInfo &rhs):
   TClingDeclInfo(rhs), fContextIdx(0)
   {
      fInterp = rhs.fInterp;
      fClassInfo = new TClingClassInfo(*rhs.fClassInfo);
      fFirstTime = rhs.fFirstTime;
      fIter = rhs.fIter;
      fIterStack = rhs.fIterStack;
      fContexts = rhs.fContexts;
      fContextIdx = rhs.fContextIdx;
   }

   TClingDataMemberInfo &operator=(const TClingDataMemberInfo &rhs)
   {
      if (this != &rhs) {
         fInterp = rhs.fInterp;
         delete fClassInfo;
         fClassInfo = new TClingClassInfo(*rhs.fClassInfo);
         fFirstTime = rhs.fFirstTime;
         fIter = rhs.fIter;
         fIterStack = rhs.fIterStack;
         fContexts = rhs.fContexts;
         fContextIdx = rhs.fContextIdx;
      }
      return *this;
   }

   typedef TDictionary::DeclId_t DeclId_t;

   int                ArrayDim() const;
   TClingClassInfo   *GetClassInfo() const { return fClassInfo; }
   const clang::Decl *GetDecl() const override {
     if (const clang::Decl* SingleDecl = TClingDeclInfo::GetDecl())
       return SingleDecl;
     return *fIter;
   }
   DeclId_t           GetDeclId() const;
   int                MaxIndex(int dim) const;
   int                InternalNext();
   bool               Next() { return InternalNext(); }
   long               Offset();
   long               Property() const;
   long               TypeProperty() const;
   int                TypeSize() const;
   const char        *TypeName() const;
   const char        *TypeTrueName(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const;
   const char        *Name() override;
   const char        *Title();
   llvm::StringRef    ValidArrayIndex() const;

};

#endif // ROOT_TClingDataMemberInfo
