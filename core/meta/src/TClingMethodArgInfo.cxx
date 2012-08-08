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

tcling_MethodArgInfo::~tcling_MethodArgInfo()
{
   delete fMethodArgInfo;
   fMethodArgInfo = 0;
}

tcling_MethodArgInfo::tcling_MethodArgInfo(cling::Interpreter* interp)
   : fMethodArgInfo(0), fInterp(interp)
{
   fMethodArgInfo = new G__MethodArgInfo();
}

tcling_MethodArgInfo::tcling_MethodArgInfo(cling::Interpreter* interp,
                                          tcling_MethodInfo* tcling_method_info)
   : fMethodArgInfo(0), fInterp(interp)
{
   if (!tcling_method_info || !tcling_method_info->IsValid()) {
      fMethodArgInfo = new G__MethodArgInfo();
      return;
   }
   fMethodArgInfo = new G__MethodArgInfo(*tcling_method_info->GetMethodInfo());
}

tcling_MethodArgInfo::tcling_MethodArgInfo(const tcling_MethodArgInfo& rhs)
   : fMethodArgInfo(0), fInterp(rhs.fInterp)
{
   if (!rhs.IsValid()) {
      fMethodArgInfo = new G__MethodArgInfo();
      return;
   }
   fMethodArgInfo = new G__MethodArgInfo(*rhs.fMethodArgInfo);
}

tcling_MethodArgInfo& tcling_MethodArgInfo::operator=(const tcling_MethodArgInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fMethodArgInfo;
      fMethodArgInfo = new G__MethodArgInfo();
      fInterp = rhs.fInterp;
   }
   else {
      delete fMethodArgInfo;
      fMethodArgInfo = new G__MethodArgInfo(*rhs.fMethodArgInfo);
      fInterp = rhs.fInterp;
   }
   return *this;
}

bool tcling_MethodArgInfo::IsValid() const
{
   return fMethodArgInfo->IsValid();
}

int tcling_MethodArgInfo::Next() const
{
   return fMethodArgInfo->Next();
}

long tcling_MethodArgInfo::Property() const
{
   return fMethodArgInfo->Property();
}

const char* tcling_MethodArgInfo::DefaultValue() const
{
   return fMethodArgInfo->DefaultValue();
}

const char* tcling_MethodArgInfo::Name() const
{
   return fMethodArgInfo->Name();
}

const char* tcling_MethodArgInfo::TypeName() const
{
   return fMethodArgInfo->Type()->Name();
}

