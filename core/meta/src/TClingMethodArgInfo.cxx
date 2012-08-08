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

TClingMethodArgInfo::~TClingMethodArgInfo()
{
   delete fMethodArgInfo;
   fMethodArgInfo = 0;
   fInterp = 0;
   fMethodInfo = 0;
}

TClingMethodArgInfo::TClingMethodArgInfo(cling::Interpreter* interp)
   : fMethodArgInfo(0), fInterp(interp), fMethodInfo(0), fIdx(-1)
{
   fMethodArgInfo = new G__MethodArgInfo();
}

TClingMethodArgInfo::TClingMethodArgInfo(cling::Interpreter* interp,
      const tcling_MethodInfo* mi)
   : fMethodArgInfo(0), fInterp(interp), fMethodInfo(0), fIdx(-1)
{
   if (!mi) {
      fMethodArgInfo = new G__MethodArgInfo();
      return;
   }
   fMethodArgInfo = new G__MethodArgInfo(*mi->GetMethodInfo());
   fMethodInfo = mi;
}

TClingMethodArgInfo::TClingMethodArgInfo(const TClingMethodArgInfo& rhs)
   : fMethodArgInfo(0), fInterp(rhs.fInterp), fMethodInfo(rhs.fMethodInfo),
     fIdx(rhs.fIdx)
{
   fMethodArgInfo = new G__MethodArgInfo(*rhs.fMethodArgInfo);
}

TClingMethodArgInfo& TClingMethodArgInfo::operator=(
   const TClingMethodArgInfo& rhs)
{
   if (this != &rhs) {
      delete fMethodArgInfo;
      fMethodArgInfo = new G__MethodArgInfo(*rhs.fMethodArgInfo);
      fInterp = rhs.fInterp;
      fMethodInfo = rhs.fMethodInfo;
      fIdx = rhs.fIdx;
   }
   return *this;
}

bool TClingMethodArgInfo::IsValidClang() const
{
   if (!gAllowClang) {
      return false;
   }
   if (!fMethodInfo) {
      return false;
   }
   if (!fMethodInfo->IsValidClang()) {
      return false;
   }
   return (fIdx > -1) &&
          (fIdx < static_cast<int>(fMethodInfo->GetMethodDecl()->getNumParams()));
}

bool TClingMethodArgInfo::IsValidCint() const
{
   if (gAllowCint) {
      return fMethodArgInfo->IsValid();
   }
   return false;
}

bool TClingMethodArgInfo::IsValid() const
{
   return IsValidCint() || IsValidClang();
}

int TClingMethodArgInfo::Next()
{
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodArgInfo->Next();
      }
      return 0;
   }
   ++fIdx;
   return IsValidClang();
}

long TClingMethodArgInfo::Property() const
{
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodArgInfo->Property();
      }
      return 0L;
   }
   if (!gAllowClang) {
      return 0L;
   }
   long property = 0L;
   const clang::FunctionDecl* FD = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl* PVD = FD->getParamDecl(fIdx);
   if (PVD->hasDefaultArg() || PVD->hasInheritedDefaultArg()) {
      property |= G__BIT_ISDEFAULT;
   }
   clang::QualType QT = PVD->getOriginalType().getCanonicalType();
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   while (1) {
      if (QT->isArrayType()) {
         QT = llvm::cast<clang::ArrayType>(QT)->getElementType();
         continue;
      }
      else if (QT->isReferenceType()) {
         property |= G__BIT_ISREFERENCE;
         QT = llvm::cast<clang::ReferenceType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isPointerType()) {
         property |= G__BIT_ISPOINTER;
         if (QT.isConstQualified()) {
            property |= G__BIT_ISPCONSTANT;
         }
         QT = llvm::cast<clang::PointerType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isMemberPointerType()) {
         QT = llvm::cast<clang::MemberPointerType>(QT)->getPointeeType();
         continue;
      }
      break;
   }
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         long cint_property = fMethodArgInfo->Property();
         if (property != cint_property) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingMethodArgInfo::Property: "
                       "%s  cint: 0x%lx  clang: 0x%lx\n", fMethodInfo->Name(),
                       cint_property, property);
            }
         }
      }
   }
   return property;
}

const char* TClingMethodArgInfo::DefaultValue() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodArgInfo->DefaultValue();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   const clang::FunctionDecl* FD = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl* PVD = FD->getParamDecl(fIdx);
   const clang::Expr* expr = PVD->getDefaultArg();
   clang::ASTContext& Context = PVD->getASTContext();
   clang::PrintingPolicy Policy(Context.getPrintingPolicy());
   static std::string buf;
   buf.clear();
   llvm::raw_string_ostream Out(buf);
   if (expr) {
      bool ImplicitInit = false;
      if (const clang::CXXConstructExpr* Construct =
               llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
         ImplicitInit = (PVD->getInitStyle() == clang::VarDecl::CallInit) &&
                        (Construct->getNumArgs() == 0) &&
                        !Construct->isListInitialization();
      }
      if (!ImplicitInit) {
         if (PVD->getInitStyle() == clang::VarDecl::CallInit) {
            //Out << "(";
         }
         else if (PVD->getInitStyle() == clang::VarDecl::CInit) {
            //Out << " = ";
         }
         expr->printPretty(Out, Context, 0, Policy, /*Indentation=*/0);
         if (PVD->getInitStyle() == clang::VarDecl::CallInit) {
            //Out << ")";
         }
         Out.flush();
      }
   }
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_val = fMethodArgInfo->DefaultValue();
         if (buf != cint_val) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingMethodArgInfo::DefaultValue: "
                       "cint: %s  clang: %s\n", cint_val, buf.c_str());
            }
         }
      }
   }
   return buf.c_str();
}

const char* TClingMethodArgInfo::Name() const
{
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodArgInfo->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   const clang::FunctionDecl* FD = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl* PVD = FD->getParamDecl(fIdx);
   static std::string buf;
   buf.clear();
   clang::PrintingPolicy Policy(PVD->getASTContext().getPrintingPolicy());
   PVD->getNameForDiagnostic(buf, Policy, /*Qualified=*/true);
   if (gAllowCint) {
      if (IsValidCint()) {
         const char* cint_name = fMethodArgInfo->Name();
         if (buf != cint_name) {
            if (gDebug > 0) {
               fprintf(stderr,
                       "VALIDITY: TClingMethodArgInfo::Name: "
                       "cint: %s  clang: %s\n", cint_name, buf.c_str());
            }
         }
      }
   }
   return buf.c_str();
}

const tcling_TypeInfo* TClingMethodArgInfo::Type() const
{
   static tcling_TypeInfo ti(fInterp);
   if (!IsValidClang()) {
      if (gAllowCint) {
         ti.Init(fMethodArgInfo->Type()->Name());
      }
      return &ti;
   }
   if (!gAllowClang) {
      return &ti;
   }
   const clang::FunctionDecl* FD = fMethodInfo->GetMethodDecl();
   const clang::ParmVarDecl* PVD = FD->getParamDecl(fIdx);
   clang::QualType QT = PVD->getOriginalType();
   ti.Init(QT);
   return &ti;
}

const char* TClingMethodArgInfo::TypeName() const
{
   if (!IsValidClang()) {
      if (gAllowCint) {
         return fMethodArgInfo->Type()->Name();
      }
      return 0;
   }
   if (!gAllowClang) {
      return 0;
   }
   return Type()->Name();
}

