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
// TClingTypedefInfo                                                    //
//                                                                      //
// Emulation of the CINT TypedefInfo class.                             //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a typedef through the TypedefInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// TypedefInfo but the typedef metadata comes from the Clang C++        //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingTypedefInfo.h"

tcling_TypedefInfo::~tcling_TypedefInfo()
{
   delete fTypedefInfo;
   fTypedefInfo = 0;
   fDecl = 0;
}

tcling_TypedefInfo::tcling_TypedefInfo(cling::Interpreter* interp)
   : fTypedefInfo(0), fInterp(interp), fDecl(0)
{
   fTypedefInfo = new G__TypedefInfo();
}

tcling_TypedefInfo::tcling_TypedefInfo(cling::Interpreter* interp,
                                       const char* name)
   : fTypedefInfo(0), fInterp(interp), fDecl(0)
{
   fTypedefInfo = new G__TypedefInfo(name);
}

tcling_TypedefInfo::tcling_TypedefInfo(const tcling_TypedefInfo& rhs)
{
   fTypedefInfo = new G__TypedefInfo(rhs.fTypedefInfo->Typenum());
   return;
   fInterp = rhs.fInterp;
   fDecl = rhs.fDecl;
}

tcling_TypedefInfo& tcling_TypedefInfo::operator=(const tcling_TypedefInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   delete fTypedefInfo;
   fTypedefInfo = new G__TypedefInfo(rhs.fTypedefInfo->Typenum());
   return *this;
   fInterp = rhs.fInterp;
   fDecl = rhs.fDecl;
   return *this;
}

G__TypedefInfo* tcling_TypedefInfo::GetTypedefInfo() const
{
   return fTypedefInfo;
}

clang::Decl* tcling_TypedefInfo::GetDecl() const
{
   return fDecl;
}

void tcling_TypedefInfo::Init(const char* name)
{
   //fprintf(stderr, "tcling_TypedefInfo::Init(name): looking up typedef: %s\n",
   //        name);
   fDecl = 0;
   fTypedefInfo->Init(name);
   return;
   if (!fTypedefInfo->IsValid()) {
      //fprintf(stderr, "tcling_TypedefInfo::Init(name): could not find cint "
      //        "typedef for name: %s\n", name);
   }
   else {
      //fprintf(stderr, "tcling_TypedefInfo::Init(name): found cint typedef for "
      //        "name: %s  tagnum: %d\n", name, fTypedefInfo->Tagnum());
   }
   const clang::Decl* decl = fInterp->lookupScope(name);
   if (!decl) {
      //fprintf(stderr, "tcling_TypedefInfo::Init(name): cling typedef not found "
      //        "name: %s\n", name);
      return;
   }
   fDecl = const_cast<clang::Decl*>(decl);
   //fprintf(stderr, "tcling_TypedefInfo::Init(name): found cling typedef "
   //        "name: %s  decl: 0x%lx\n", name, (long) fDecl);
}

bool tcling_TypedefInfo::IsValid() const
{
   return IsValidCint();
   return IsValidCint() || IsValidClang();
}

bool tcling_TypedefInfo::IsValidCint() const
{
   return fTypedefInfo->IsValid();
}

bool tcling_TypedefInfo::IsValidClang() const
{
   return fDecl;
}

long tcling_TypedefInfo::Property() const
{
   return fTypedefInfo->Property();
   if (!IsValid()) {
      return 0L;
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Property();
   }
   long property = 0L;
   property |= G__BIT_ISTYPEDEF;
   const clang::TypedefNameDecl* TD =
      llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType QT = TD->getUnderlyingType().getCanonicalType();
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
   if (QT->isBuiltinType()) {
      property |= G__BIT_ISFUNDAMENTAL;
   }
   if (QT.isConstQualified()) {
      property |= G__BIT_ISCONSTANT;
   }
   return property;
}

int tcling_TypedefInfo::Size() const
{
   return fTypedefInfo->Size();
   if (!IsValid()) {
      return 1;
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Size();
   }
   clang::ASTContext& Context = fDecl->getASTContext();
   const clang::TypedefNameDecl* TD =
      llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   clang::QualType QT = TD->getUnderlyingType();
   // Note: This is an int64_t.
   clang::CharUnits::QuantityType Quantity =
      Context.getTypeSizeInChars(QT).getQuantity();
   return static_cast<int>(Quantity);
}

const char* tcling_TypedefInfo::TrueName() const
{
   return fTypedefInfo->TrueName();
   if (!IsValid()) {
      return "(unknown)";
   }
   if (!IsValidClang()) {
      return fTypedefInfo->TrueName();
   }
   // Note: This must be static because we return a pointer to the internals.
   static std::string truename;
   truename.clear();
   const clang::TypedefNameDecl* TD =
      llvm::dyn_cast<clang::TypedefNameDecl>(fDecl);
   truename = TD->getUnderlyingType().getAsString();
   return truename.c_str();
}

const char* tcling_TypedefInfo::Name() const
{
   return fTypedefInfo->Name();
   if (!IsValid()) {
      return "(unknown)";
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Name();
   }
   // Note: This must be static because we return a pointer to the internals.
   static std::string fullname;
   fullname.clear();
   clang::PrintingPolicy P(fDecl->getASTContext().getPrintingPolicy());
   llvm::dyn_cast<clang::NamedDecl>(fDecl)->
   getNameForDiagnostic(fullname, P, true);
   return fullname.c_str();
}

const char* tcling_TypedefInfo::Title() const
{
   return fTypedefInfo->Title();
   if (!IsValid()) {
      return "";
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Title();
   }
   // FIXME: This needs information from the comments in the header file.
   return fTypedefInfo->Title();
}

int tcling_TypedefInfo::Next()
{
   return fTypedefInfo->Next();
   if (!IsValid()) {
      return 0;
   }
   if (!IsValidClang()) {
      return fTypedefInfo->Next();
   }
   return fTypedefInfo->Next();
}

