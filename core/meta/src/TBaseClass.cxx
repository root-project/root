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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Each class (see TClass) has a linked list of its base class(es).    //
//  This class describes one single base class.                         //
//  The base class info is obtained via the CINT api.                   //
//     see class TCint.                                                 //
//                                                                      //
//  The base class information is used a.o. in to find all inherited    //
//  methods.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


ClassImp(TBaseClass)

//______________________________________________________________________________
TBaseClass::TBaseClass(BaseClassInfo_t *info, TClass *cl) : TDictionary()
{
   // Default TBaseClass ctor. TBaseClasses are constructed in TClass
   // via a call to TCint::CreateListOfBaseClasses().

   fInfo     = info;
   fClass    = cl;
   fClassPtr = 0;
   if (fInfo) SetName(gCint->BaseClassInfo_FullName(fInfo));
}

//______________________________________________________________________________
TBaseClass::~TBaseClass()
{
   // TBaseClass dtor deletes adopted CINT BaseClassInfo object.

   gCint->BaseClassInfo_Delete(fInfo);
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

   if (!fClassPtr) fClassPtr = TClass::GetClass(fName, load);
   return fClassPtr;
}

//______________________________________________________________________________
Int_t TBaseClass::GetDelta() const
{
   // Get offset from "this" to part of base class.

   return (Int_t)gCint->BaseClassInfo_Offset(fInfo);
}

//______________________________________________________________________________
const char *TBaseClass::GetTitle() const
{
   // Get base class description (comment).

   TClass *c = ((TBaseClass *)this)->GetClassPointer();
   return c ? c->GetTitle() : "";
}

//______________________________________________________________________________
int TBaseClass::IsSTLContainer()
{
   // Return which type (if any) of STL container the data member is.

   if (!fInfo) return kNone;
   const char *type = gCint->BaseClassInfo_TmpltName(fInfo);
   if (!type) return kNone;

   if (!strcmp(type, "vector"))   return kVector;
   if (!strcmp(type, "list"))     return kList;
   if (!strcmp(type, "deque"))    return kDeque;
   if (!strcmp(type, "map"))      return kMap;
   if (!strcmp(type, "multimap")) return kMultimap;
   if (!strcmp(type, "set"))      return kSet;
   if (!strcmp(type, "multiset")) return kMultiset;
   return kNone;
}

//______________________________________________________________________________
Long_t TBaseClass::Property() const
{
   // Get property description word. For meaning of bits see EProperty.

   return gCint->BaseClassInfo_Property(fInfo);
}
