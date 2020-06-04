// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingMethodInfo
#define ROOT_TClingMethodInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingMethodInfo                                                     //
//                                                                      //
// Emulation of the CINT MethodInfo class.                              //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a function through the MethodInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// MethodInfo but the typedef metadata comes from the Clang C++         //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingDeclInfo.h"

#include "TString.h"
#include "TDictionary.h"
#include <string>

#include "clang/AST/DeclBase.h"
#include "llvm/ADT/SmallVector.h"

namespace cling {
   class Interpreter;
}

namespace clang {
   class FunctionDecl;
   class CXXMethodDecl;
}

namespace ROOT {
   namespace TMetaUtils {
      class TNormalizedCtxt;
   }
}

class TClingClassInfo;
class TClingTypeInfo;

class TClingMethodInfo final : public TClingDeclInfo {
private:
   cling::Interpreter                          *fInterp; // Cling interpreter, we do *not* own.
   llvm::SmallVector<clang::DeclContext *, 2>   fContexts; // Set of DeclContext that we will iterate over.
   bool                                         fFirstTime; // Flag for first time incrementing iterator, cint semantics are weird.
   unsigned int                                 fContextIdx; // Index in fContexts of DeclContext we are iterating over.
   clang::DeclContext::decl_iterator            fIter; // Our iterator.
   std::string                                  fTitle; // The meta info for the method.
   const clang::FunctionDecl                   *fTemplateSpec; // an all-default-template-args function.
   llvm::SmallVector<clang::Decl *,4>           fDefDataSpecFuns; // decl_begin() will skip these special members, materialized from DefinitionData
   llvm::SmallVector<clang::Decl *,4>::const_iterator fDefDataSpecFunIter; // Iterator over fDefDataSpecFuns

   const clang::Decl* GetDeclSlow() const;

public:
   explicit TClingMethodInfo(cling::Interpreter *interp)
      : TClingDeclInfo(nullptr), fInterp(interp), fFirstTime(true), fContextIdx(0U), fTitle(""),
        fTemplateSpec(0) {}

   TClingMethodInfo(const TClingMethodInfo&);
   TClingMethodInfo& operator=(const TClingMethodInfo &in);

   // Takes concrete decl and disables the iterator.
   TClingMethodInfo(cling::Interpreter *, const clang::FunctionDecl *);
   TClingMethodInfo(cling::Interpreter *, TClingClassInfo *);

   ~TClingMethodInfo();

   const clang::FunctionDecl                   *GetMethodDecl() const;
   TDictionary::DeclId_t                        GetDeclId() const;
   cling::Interpreter                          *GetInterpreter() const { return fInterp; }
   void                                         CreateSignature(TString &signature) const;
   void                                         Init(const clang::FunctionDecl *);
   void                                        *InterfaceMethod(const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) const;

   const clang::Decl *GetDecl() const override {
     if (const clang::Decl* SingleDecl = TClingDeclInfo::GetDecl())
       return SingleDecl;
     return GetDeclSlow();
   }
   int                                          NArg() const;
   int                                          NDefaultArg() const;
   int                                          InternalNext();
   int                                          Next();
   long                                         Property() const;
   long                                         ExtraProperty() const;
   TClingTypeInfo                              *Type() const;
   std::string                                  GetMangledName() const;
   const char                                  *GetPrototype();
   const char                                  *Name() const override;
   const char                                  *TypeName() const;
   const char                                  *Title();
};

#endif // ROOT_TClingMethodInfo
