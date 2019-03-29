/// \file TClingDeclInfo.cxx
///
/// \brief The file contains a base class of TCling*Info classes.
///
/// \author Vassil Vassilev <vvasilev@cern.ch>
///
/// \date March, 2019
///
/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClingDeclInfo.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

using namespace clang;

// pin the vtable here.
TClingDeclInfo::~TClingDeclInfo() {}

const char* TClingDeclInfo::Name()
{
   if (!IsValid())
      return 0;

   if (!fNameCache.empty())
      return fNameCache.c_str();

   const Decl* D = GetDecl();
   if (!isa<NamedDecl>(D))
      return 0;

   const NamedDecl* ND = cast<NamedDecl>(D);
   clang::PrintingPolicy policy(ND->getASTContext().getPrintingPolicy());
   llvm::raw_string_ostream stream(fNameCache);
   ND->getNameForDiagnostic(stream, policy, /*Qualified=*/false);
   stream.flush();
   return fNameCache.c_str();
}
