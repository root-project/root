// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorList                                                        //
//                                                                      //
// A TList derived class that makes sure that objects added to it       //
// are not linked to the currently open file (like histograms,          //
// eventlists and trees). Also it makes sure the name of the added      //
// object is unique. This class is used in the TSelector for the        //
// output list.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSelectorList.h"
#include "TMethodCall.h"


ClassImp(TSelectorList)

//______________________________________________________________________________
Bool_t TSelectorList::UnsetDirectory(TObject *obj)
{
   // If the class of obj has the SetDirectory(TDirectory*) method
   // call it to unset the directory assiciation. The objects in the
   // selector list or owned by the list and not by the directory that
   // was active when they were created. Returns true in case of success.

   if (!obj || !obj->IsA())
      return kFALSE;

   TMethodCall callEnv;
   callEnv.InitWithPrototype(obj->IsA(), "SetDirectory", "TDirectory*");
   if (!callEnv.IsValid())
      return kFALSE;

   callEnv.SetParam((Long_t) 0);
   callEnv.Execute(obj);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSelectorList::CheckDuplicateName(TObject *obj)
{
   // Check for duplicate object names in the list. If an object with
   // the same name is added then the merge function will fail that will
   // look up objects in different output lists by name. Returns true
   // in case name is unique.

   if (!obj)
      return kFALSE;

   TObject *org = FindObject(obj->GetName());
   if (org == obj) {
      Error("CheckDuplicateName","object with name: %s already in the list",obj->GetName());
      return kFALSE;
   }

   if (org) {
      Error("CheckDuplicateName","an object with the same name: %s is already in the list",obj->GetName());
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TSelectorList::AddFirst(TObject *obj)
{
   // Add at the start of the list

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddFirst(obj);
}

//______________________________________________________________________________
void TSelectorList::AddFirst(TObject *obj, Option_t *opt)
{
   // Add at the start of the list

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddFirst(obj, opt);
}

//______________________________________________________________________________
void TSelectorList::AddLast(TObject *obj)
{
   // Add at the end of the list

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddLast(obj);
}

//______________________________________________________________________________
void TSelectorList::AddLast(TObject *obj, Option_t *opt)
{
   // Add at the end of the list

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddLast(obj, opt);
}

//______________________________________________________________________________
void TSelectorList::AddAt(TObject *obj, Int_t idx)
{
   // Add to the list.

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddAt(obj, idx);
}

//______________________________________________________________________________
void TSelectorList::AddAfter(const TObject *after, TObject *obj)
{
   // Add to the list.

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddAfter(after, obj);
}

//______________________________________________________________________________
void TSelectorList::AddAfter(TObjLink *after, TObject *obj)
{
   // Add to the list.

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddAfter(after, obj);
}

//______________________________________________________________________________
void TSelectorList::AddBefore(const TObject *before, TObject *obj)
{
   // Add to the list.

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddBefore(before, obj);
}

//______________________________________________________________________________
void TSelectorList::AddBefore(TObjLink *before, TObject *obj)
{
   // Add to the list.

   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      TList::AddBefore(before, obj);
}
