// @(#)root/cont:$Id$
// Author: Fons Rademakers   14/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TSortedList
\ingroup Containers
A sorted doubly linked list. All sortable classes inheriting from
TObject can be inserted in a TSortedList.
*/

#include "TSortedList.h"


ClassImp(TSortedList);

////////////////////////////////////////////////////////////////////////////////
/// Add object in sorted list. Uses object Compare() member to find right
/// position.

void TSortedList::Add(TObject *obj)
{
   if (IsArgNull("Add", obj)) return;

   if (!obj->IsSortable()) {
      Error("Add", "object must be sortable");
      return;
   }

   if (!fFirst) {
      TList::AddLast(obj);
      return;
   }

   auto lnk = fFirst;

   while (lnk) {
      Int_t cmp = lnk->GetObject()->Compare(obj);
      if ((IsAscending() && cmp > 0) || (!IsAscending() && cmp < 0)) {
         if (lnk->Prev()) {
            NewLink(obj, lnk->PrevSP());
            fSize++;
            return;
         } else {
            TList::AddFirst(obj);
            return;
         }
      }
      lnk = lnk->NextSP();
   }
   TList::AddLast(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object in sorted list. Uses object Compare() member to find right
/// position and also store option. See TList::Add for explanation of
/// usage of option.

void TSortedList::Add(TObject *obj, Option_t *opt)
{
   if (IsArgNull("Add", obj)) return;

   if (!obj->IsSortable()) {
      Error("Add", "object must be sortable");
      return;
   }

   if (!fFirst) {
      TList::Add(obj, opt);
      return;
   }

   auto lnk = fFirst;

   while (lnk) {
      Int_t cmp = lnk->GetObject()->Compare(obj);
      if ((IsAscending() && cmp > 0) || (!IsAscending() && cmp < 0)) {
         if (lnk->Prev()) {
            NewOptLink(obj, opt, lnk);
            fSize++;
            return;
         } else {
            TList::AddFirst(obj, opt);
            return;
         }
      }
      lnk = lnk->NextSP();
   }
   TList::Add(obj, opt);
}
