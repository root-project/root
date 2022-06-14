// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingMethodArgInfo
#define ROOT_TClingMethodArgInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingMethodArgInfo                                                  //
//                                                                      //
// Emulation of the CINT MethodInfo class.                              //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the arguments to a function through the MethodArgInfo class.  This   //
// class provides the same functionality, using an interface as close   //
// as possible to MethodArgInfo but the typedef metadata comes from     //
// the Clang C++ compiler, not CINT.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingDeclInfo.h"

#include "clang/AST/Decl.h"

namespace clang {
   class ParmVarDecl;
}

namespace cling {
   class Interpreter;
}

class TClingMethodInfo;
class TClingTypeInfo;

class TClingMethodArgInfo final : public TClingDeclInfo {

private:

   cling::Interpreter       *fInterp; // Cling interpreter, we do *not* own.
   int                       fIdx; // Iterator, current parameter index.

public:

   explicit TClingMethodArgInfo(cling::Interpreter *interp) : TClingDeclInfo(nullptr), fInterp(interp), fIdx(-1) {}
   TClingMethodArgInfo(cling::Interpreter *interp, const TClingMethodInfo *mi);

   // Add a covariant return type for handy use.
   const clang::ParmVarDecl* GetDecl() const override {
      if (const auto FD = llvm::cast_or_null<clang::FunctionDecl>(TClingDeclInfo::GetDecl()))
         return FD->getParamDecl(fIdx);
      return nullptr;
   }
   bool                   IsValid() const override;
   int                    Next();
   long                   Property() const;
   const char            *DefaultValue() const;
   const TClingTypeInfo  *Type() const;
   const char            *TypeName() const;

};

#endif // ROOT_TClingMethodArgInfo
