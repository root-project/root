// @(#)root/meta:$Id$
// Author: Fons Rademakers   08/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBaseClass.h"
#include "TClass.h"
#include "TInterpreter.h"
#include <limits.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Each class (see TClass) has a linked list of its base class(es).    //
//  This class describes one single base class.                         //
//  The base class info is obtained via the CINT api.                   //
//     see class TCling.                                                 //
//                                                                      //
//  The base class information is used a.o. in to find all inherited    //
//  methods.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


ClassImp(TBaseClass)

//______________________________________________________________________________
TBaseClass::TBaseClass(BaseClassInfo_t *info, TClass *cl) :
   TDictionary(), fInfo(info), fClass(cl), fDelta(INT_MAX)
{
   // Default TBaseClass ctor. TBaseClasses are constructed in TClass
   // via a call to TCling::CreateListOfBaseClasses().

   if (fInfo) SetName(gCling->BaseClassInfo_FullName(fInfo));
}

//______________________________________________________________________________
TBaseClass::~TBaseClass()
{
   // TBaseClass dtor deletes adopted CINT BaseClassInfo object.

   gCling->BaseClassInfo_Delete(fInfo);
}

//______________________________________________________________________________
void TBaseClass::Browse(TBrowser *b)
{
   // Called by the browser, to browse a baseclass.

   TClass *c = GetClassPointer();
   if (c) c->Browse(b);
}

//______________________________________________________________________________
TClass *TBaseClass::GetClassPointer(Bool_t load)
{
   // Get pointer to the base class TClass.

   if (!fClassPtr) {
      if (fInfo) fClassPtr = TClass::GetClass(gCling->BaseClassInfo_ClassInfo(fInfo),load);
      else fClassPtr = TClass::GetClass(fName, load);
   }
   return fClassPtr;
}

//______________________________________________________________________________
Int_t TBaseClass::GetDelta()
{
   // Get offset from "this" to part of base class.

   // Initialized to INT_MAX to signal that it's unset; -1 is a valid value
   // meaning "cannot calculate base offset".
   if (fDelta == INT_MAX)
      fDelta = (Int_t)gCling->BaseClassInfo_Offset(fInfo);
   return fDelta;
}

//______________________________________________________________________________
const char *TBaseClass::GetTitle() const
{
   // Get base class description (comment).

   TClass *c = ((TBaseClass *)this)->GetClassPointer();
   return c ? c->GetTitle() : "";
}

//______________________________________________________________________________
ROOT::ESTLType TBaseClass::IsSTLContainer()
{
   // Return which type (if any) of STL container the data member is.

   if (!fInfo) return ROOT::kNotSTL;
   const char *type = gCling->BaseClassInfo_TmpltName(fInfo);
   if (!type) return ROOT::kNotSTL;

   if (!strcmp(type, "vector"))   return ROOT::kSTLvector;
   if (!strcmp(type, "list"))     return ROOT::kSTLlist;
   if (!strcmp(type, "deque"))    return ROOT::kSTLdeque;
   if (!strcmp(type, "map"))      return ROOT::kSTLmap;
   if (!strcmp(type, "multimap")) return ROOT::kSTLmultimap;
   if (!strcmp(type, "set"))      return ROOT::kSTLset;
   if (!strcmp(type, "multiset")) return ROOT::kSTLmultiset;
   return ROOT::kNotSTL;
}

//______________________________________________________________________________
Long_t TBaseClass::Property() const
{
   // Get property description word. For meaning of bits see EProperty.
   if (fProperty == -1)
      fProperty = gCling->BaseClassInfo_Property(fInfo);
   return fProperty;
}
