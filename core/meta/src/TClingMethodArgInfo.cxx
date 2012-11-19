// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TClingMethodArgInfo.h"

#include "TClingMethodInfo.h"
#include "Property.h"
#include "TClingProperty.h"
#include "TClingTypeInfo.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

bool TClingMethodArgInfo::IsValid() const
{
   if (!fMethodInfo || !fMethodInfo->IsValid()) {
      return false;
   }
   int numParams = static_cast<int>(fMethodInfo->GetMethodDecl()->getNumParams());
   return (fIdx > -1) && (fIdx < numParams);
}

int TClingMethodArgInfo::Next()
{
   ++fIdx;
   return IsValid();
}

long TClingMethodArgInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   const clang::FunctionDecl *fd = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl *pvd = fd->getParamDecl(fIdx);
   if (pvd->hasDefaultArg() || pvd->hasInheritedDefaultArg()) {
      property |= G__BIT_ISDEFAULT;
   }
   clang::QualType qt = pvd->getOriginalType().getCanonicalType();
   if (qt.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   while (1) {
      if (qt->isArrayType()) {
         qt = llvm::cast<clang::ArrayType>(qt)->getElementType();
         continue;
      }
      else if (qt->isReferenceType()) {
         property |= G__BIT_ISREFERENCE;
         qt = llvm::cast<clang::ReferenceType>(qt)->getPointeeType();
         continue;
      }
      else if (qt->isPointerType()) {
         property |= G__BIT_ISPOINTER;
         if (qt.isConstQualified()) {
            property |= G__BIT_ISPCONSTANT;
         }
         qt = llvm::cast<clang::PointerType>(qt)->getPointeeType();
         continue;
      }
      else if (qt->isMemberPointerType()) {
         qt = llvm::cast<clang::MemberPointerType>(qt)->getPointeeType();
         continue;
      }
      break;
   }
   if (qt.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   return property;
}

const char *TClingMethodArgInfo::DefaultValue() const
{
   if (!IsValid()) {
      return 0;
   }
   const clang::FunctionDecl *fd = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl *pvd = fd->getParamDecl(fIdx);
   const clang::Expr *expr = pvd->getDefaultArg();
   clang::ASTContext &context = pvd->getASTContext();
   clang::PrintingPolicy policy(context.getPrintingPolicy());
   static std::string buf;
   buf.clear();
   llvm::raw_string_ostream out(buf);
   if (!expr) {
      // CINT returned NULL for non-defaulted args.
      return 0;
   }
   bool implicitInit = false;
   if (const clang::CXXConstructExpr *construct =
       llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
      implicitInit = (pvd->getInitStyle() == clang::VarDecl::CallInit) &&
         (construct->getNumArgs() == 0) &&
         !construct->isListInitialization();
   }
   if (!implicitInit) {
      if (pvd->getInitStyle() == clang::VarDecl::CallInit) {
         //out << "(";
      }
      else if (pvd->getInitStyle() == clang::VarDecl::CInit) {
         //out << " = ";
      }
      expr->printPretty(out, 0, policy, /*Indentation=*/0);
      if (pvd->getInitStyle() == clang::VarDecl::CallInit) {
         //out << ")";
      }
      out.flush();
   }
   return buf.c_str();
}

const char *TClingMethodArgInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   const clang::FunctionDecl *fd = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl *pvd = fd->getParamDecl(fIdx);
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy policy(pvd->getASTContext().getPrintingPolicy());
   pvd->getNameForDiagnostic(buf, policy, /*Qualified=*/true);
   return buf.c_str();
}

const TClingTypeInfo *TClingMethodArgInfo::Type() const
{
   static TClingTypeInfo ti(fInterp);
   if (!IsValid()) {
      return &ti;
   }
   const clang::FunctionDecl *fd = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl *pvd = fd->getParamDecl(fIdx);
   clang::QualType qt = pvd->getOriginalType();
   ti.Init(qt);
   return &ti;
}

const char *TClingMethodArgInfo::TypeName() const
{
   if (!IsValid()) {
      return 0;
   }
   return Type()->Name();
}

