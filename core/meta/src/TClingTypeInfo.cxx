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
// TClingTypeInfo                                                       //
//                                                                      //
// Emulation of the CINT TypeInfo class.                                //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a type through the TypeInfo class.  This class provides the same     //
// functionality, using an interface as close as possible to TypeInfo   //
// but the type metadata comes from the Clang C++ compiler, not CINT.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingTypeInfo.h"

//______________________________________________________________________________



tcling_TypeInfo::~tcling_TypeInfo()
{
   delete fTypeInfo;
   fTypeInfo = 0;
   delete fClassInfo;
   fClassInfo = 0;
   fDecl = 0;
}

tcling_TypeInfo::tcling_TypeInfo(cling::Interpreter* interp)
   : fTypeInfo(0), fClassInfo(0), fInterp(interp), fDecl(0)
{
   fTypeInfo = new G__TypeInfo;
   fClassInfo = new G__ClassInfo;
}

tcling_TypeInfo::tcling_TypeInfo(cling::Interpreter* interp, const char* name)
   : fTypeInfo(0), fClassInfo(0), fInterp(interp), fDecl(0)
{
   fTypeInfo = new G__TypeInfo(name);
   int tagnum = fTypeInfo->Tagnum();
   if (tagnum == -1) {
      fClassInfo = new G__ClassInfo;
      return;
   }
   fClassInfo = new G__ClassInfo(tagnum);
   return;
   //fprintf(stderr, "tcling_TypeInfo(name): looking up cling class: %s  "
   //        "tagnum: %d\n", name, tagnum);
   const clang::Decl* decl = fInterp->lookupScope(name);
   if (!decl) {
      //fprintf(stderr, "tcling_TypeInfo(name): cling class not found: %s  "
      //        "tagnum: %d\n", name, tagnum);
      return;
   }
   fDecl = const_cast<clang::Decl*>(decl);
   //fprintf(stderr, "tcling_TypeInfo(name): cling class found: %s  "
   //        "tagnum: %d  Decl: 0x%lx\n", name, tagnum, (long) fDecl);
}

tcling_TypeInfo::tcling_TypeInfo(const tcling_TypeInfo& rhs)
{
   fTypeInfo = new G__TypeInfo(*rhs.fTypeInfo);
   fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
   fInterp = rhs.fInterp;
   fDecl = rhs.fDecl;
}

tcling_TypeInfo& tcling_TypeInfo::operator=(const tcling_TypeInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   delete fTypeInfo;
   fTypeInfo = new G__TypeInfo(*rhs.fTypeInfo);
   delete fClassInfo;
   fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
   fInterp = rhs.fInterp;
   fDecl = rhs.fDecl;
   return *this;
}

G__TypeInfo* tcling_TypeInfo::GetTypeInfo() const
{
   return fTypeInfo;
}

G__ClassInfo* tcling_TypeInfo::GetClassInfo() const
{
   return fClassInfo;
}

clang::Decl* tcling_TypeInfo::GetDecl() const
{
   return fDecl;
}

void tcling_TypeInfo::Init(const char* name)
{
   fTypeInfo->Init(name);
   int tagnum = fTypeInfo->Tagnum();
   if (tagnum == -1) {
      fClassInfo = new G__ClassInfo;
      fDecl = 0;
      return;
   }
   fClassInfo  = new G__ClassInfo(tagnum);
   return;
   const char* fullname = fClassInfo->Fullname();
   //fprintf(stderr, "tcling_TypeInfo::Init(name): looking up cling class: %s  "
   //        "tagnum: %d\n", fullname, tagnum);
   const clang::Decl* decl = fInterp->lookupScope(fullname);
   if (!decl) {
      //fprintf(stderr, "tcling_TypeInfo::Init(name): cling class not found: %s  "
      //        "tagnum: %d\n", fullname, tagnum);
      return;
   }
   fDecl = const_cast<clang::Decl*>(decl);
   //fprintf(stderr, "tcling_TypeInfo::Init(name): cling class found: %s  "
   //        "tagnum: %d  Decl: 0x%lx\n", fullname, tagnum, (long) fDecl);
}

bool tcling_TypeInfo::IsValid() const
{
   return fTypeInfo->IsValid();
}

const char* tcling_TypeInfo::Name() const
{
   return fTypeInfo->Name();
}

long tcling_TypeInfo::Property() const
{
   return fTypeInfo->Property();
}

int tcling_TypeInfo::RefType() const
{
   return fTypeInfo->Reftype();
}

int tcling_TypeInfo::Size() const
{
   return fTypeInfo->Size();
}

const char* tcling_TypeInfo::TrueName() const
{
   return fTypeInfo->TrueName();
}

