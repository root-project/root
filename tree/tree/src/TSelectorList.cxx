// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSelectorList
\ingroup tree

A TList derived class that makes sure that objects added to it
are not linked to the currently open file (like histograms,
eventlists and trees). Also it makes sure the name of the added
object is unique. This class is used in the TSelector for the
output list.
*/

#include "TSelectorList.h"
#include "TMethodCall.h"

ClassImp(TSelectorList);

////////////////////////////////////////////////////////////////////////////////
/// If the class of obj has the SetDirectory(TDirectory*) method
/// call it to unset the directory association. The objects in the
/// selector list or owned by the list and not by the directory that
/// was active when they were created. Returns true in case of success.

Bool_t TSelectorList::UnsetDirectory(TObject *obj)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Check for duplicate object names in the list. If an object with
/// the same name is added then the merge function will fail that will
/// look up objects in different output lists by name. Returns true
/// in case name is unique.

Bool_t TSelectorList::CheckDuplicateName(TObject *obj)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Add at the start of the list

void TSelectorList::AddFirst(TObject *obj)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddFirst(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add at the start of the list

void TSelectorList::AddFirst(TObject *obj, Option_t *opt)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddFirst(obj, opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Add at the end of the list

void TSelectorList::AddLast(TObject *obj)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddLast(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add at the end of the list

void TSelectorList::AddLast(TObject *obj, Option_t *opt)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddLast(obj, opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Add to the list.

void TSelectorList::AddAt(TObject *obj, Int_t idx)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddAt(obj, idx);
}

////////////////////////////////////////////////////////////////////////////////
/// Add to the list.

void TSelectorList::AddAfter(const TObject *after, TObject *obj)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddAfter(after, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add to the list.

void TSelectorList::AddAfter(TObjLink *after, TObject *obj)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddAfter(after, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add to the list.

void TSelectorList::AddBefore(const TObject *before, TObject *obj)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddBefore(before, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add to the list.

void TSelectorList::AddBefore(TObjLink *before, TObject *obj)
{
   UnsetDirectory(obj);
   if (CheckDuplicateName(obj))
      THashList::AddBefore(before, obj);
}
