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

#include "TVirtualMutex.h" // For R__LOCKGUARD

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
   TDictionary(), fInfo(info), fClass(cl), fDelta(INT_MAX),
   fProperty(-1), fSTLType(-1)
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
   if (fDelta == INT_MAX) {
      R__LOCKGUARD(gInterpreterMutex);
      if (Property() & kIsVirtualBase)
         fDelta = -1;
      else if (fInfo)
         fDelta = (Int_t)gCling->BaseClassInfo_Offset(fInfo);
   }
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

   // fSTLType is -1 if not yet evaulated.
   // fSTLType is -2 if no fInfo was available.

   if (fSTLType < 0) {
      if (!fInfo) {
         fSTLType = -2;
      } else {
         const char *type = gCling->BaseClassInfo_TmpltName(fInfo);
         if (!type)                                    fSTLType = ROOT::kNotSTL;
         else if (!strcmp(type, "vector"))             fSTLType = ROOT::kSTLvector;
         else if (!strcmp(type, "list"))               fSTLType = ROOT::kSTLlist;
         else if (!strcmp(type, "forward_list"))       fSTLType = ROOT::kSTLforwardlist;
         else if (!strcmp(type, "deque"))              fSTLType = ROOT::kSTLdeque;
         else if (!strcmp(type, "map"))                fSTLType = ROOT::kSTLmap;
         else if (!strcmp(type, "multimap"))           fSTLType = ROOT::kSTLmultimap;
         else if (!strcmp(type, "set"))                fSTLType = ROOT::kSTLset;
         else if (!strcmp(type, "multiset"))           fSTLType = ROOT::kSTLmultiset;
         else if (!strcmp(type, "unordered_set"))      fSTLType = ROOT::kSTLunorderedset;
         else if (!strcmp(type, "unordered_multiset")) fSTLType = ROOT::kSTLunorderedmultiset;
         else if (!strcmp(type, "unordered_map"))      fSTLType = ROOT::kSTLunorderedmap;
         else if (!strcmp(type, "unordered_multimap")) fSTLType = ROOT::kSTLunorderedmultimap;
         else                                          fSTLType = ROOT::kNotSTL;
      }
   }
   if (fSTLType == -2) return ROOT::kNotSTL;
   return (ROOT::ESTLType) fSTLType;
}

//______________________________________________________________________________
Long_t TBaseClass::Property() const
{
   // Get property description word. For meaning of bits see EProperty.
   if (fProperty == -1 && fInfo) {
      R__LOCKGUARD(gInterpreterMutex);
      fProperty = gCling->BaseClassInfo_Property(fInfo);
   }
   return fProperty;
}

//______________________________________________________________________________
void TBaseClass::Streamer(TBuffer& b) {
   // Stream an object of TBaseClass. Triggers the calculation of the
   // cache variables to store them.
   if (b.IsReading()) {
      b.ReadClassBuffer(Class(), this);
   } else {
      // Writing.
      // Calculate cache properties first.
      GetDelta();
      Property();
      IsSTLContainer();
      b.WriteClassBuffer(Class(), this);
   }
}
