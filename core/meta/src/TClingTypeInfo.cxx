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



TClingTypeInfo::~TClingTypeInfo()
{
   delete fTypeInfo;
   fTypeInfo = 0;
   delete fClassInfo;
   fClassInfo = 0;
   fDecl = 0;
}

TClingTypeInfo::TClingTypeInfo(cling::Interpreter* interp)
   : fTypeInfo(0), fClassInfo(0), fInterp(interp), fDecl(0)
{
   fTypeInfo = new G__TypeInfo;
   fClassInfo = new G__ClassInfo;
}

TClingTypeInfo::TClingTypeInfo(cling::Interpreter* interp, const char* name)
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
   //fprintf(stderr, "TClingTypeInfo(name): looking up cling class: %s  "
   //        "tagnum: %d\n", name, tagnum);
   const clang::Decl* decl = fInterp->lookupScope(name);
   if (!decl) {
      //fprintf(stderr, "TClingTypeInfo(name): cling class not found: %s  "
      //        "tagnum: %d\n", name, tagnum);
      return;
   }
   fDecl = const_cast<clang::Decl*>(decl);
   //fprintf(stderr, "TClingTypeInfo(name): cling class found: %s  "
   //        "tagnum: %d  Decl: 0x%lx\n", name, tagnum, (long) fDecl);
}

TClingTypeInfo::TClingTypeInfo(const TClingTypeInfo& rhs)
{
   fTypeInfo = new G__TypeInfo(*rhs.fTypeInfo);
   fClassInfo = new G__ClassInfo(rhs.fClassInfo->Tagnum());
   fInterp = rhs.fInterp;
   fDecl = rhs.fDecl;
}

TClingTypeInfo& TClingTypeInfo::operator=(const TClingTypeInfo& rhs)
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

G__TypeInfo* TClingTypeInfo::GetTypeInfo() const
{
   return fTypeInfo;
}

G__ClassInfo* TClingTypeInfo::GetClassInfo() const
{
   return fClassInfo;
}

clang::Decl* TClingTypeInfo::GetDecl() const
{
   return fDecl;
}

void TClingTypeInfo::Init(const char* name)
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
   //fprintf(stderr, "TClingTypeInfo::Init(name): looking up cling class: %s  "
   //        "tagnum: %d\n", fullname, tagnum);
   const clang::Decl* decl = fInterp->lookupScope(fullname);
   if (!decl) {
      //fprintf(stderr, "TClingTypeInfo::Init(name): cling class not found: %s  "
      //        "tagnum: %d\n", fullname, tagnum);
      return;
   }
   fDecl = const_cast<clang::Decl*>(decl);
   //fprintf(stderr, "TClingTypeInfo::Init(name): cling class found: %s  "
   //        "tagnum: %d  Decl: 0x%lx\n", fullname, tagnum, (long) fDecl);
}

bool TClingTypeInfo::IsValid() const
{
   return fTypeInfo->IsValid();
}

const char* TClingTypeInfo::Name() const
{
   return fTypeInfo->Name();
}

long TClingTypeInfo::Property() const
{
   return fTypeInfo->Property();
}

int TClingTypeInfo::RefType() const
{
   return fTypeInfo->Reftype();
}

int TClingTypeInfo::Size() const
{
   return fTypeInfo->Size();
}

const char* TClingTypeInfo::TrueName() const
{
   return fTypeInfo->TrueName();
}

