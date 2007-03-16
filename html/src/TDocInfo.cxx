// @(#)root/html:$Name:  $:$Id: TDocInfo.cxx,v 1.1 2007/02/07 20:40:38 brun Exp $
// Author: Axel Naumann 2007-01-09

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TDocInfo.h"

#include "TClass.h"
#include "TSystem.h"

ClassImp(TClassDocInfo);

const char* TClassDocInfo::GetName() const
{
   // Get the class name, or (UNKNOWN) is no TClass object was found.

   return fClass ? fClass->GetName() : "(UNKNOWN)";
}

ULong_t TClassDocInfo::Hash() const
{
   // Forward to TClass::Hash(), return -1 if no TClass object was found.
   return fClass ? fClass->Hash() : (ULong_t)-1;
}

Int_t TClassDocInfo::Compare(const TObject* obj) const
{
   // Compare two TClassDocInfo objects; used for sorting.
   return fClass ? fClass->Compare(obj) : obj < this;
}


ClassImp(TModuleDocInfo);

void TModuleDocInfo::SetSourceDir(const char* dir) {
   fSourceDir = dir;
   gSystem->ExpandPathName(fSourceDir);
}
