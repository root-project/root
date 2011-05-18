// @(#)root/cont:$Id$
// Author: Fons Rademakers   14/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSortedList
#define ROOT_TSortedList


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSortedList                                                          //
//                                                                      //
// A sorted doubly linked list. All sortable classes inheriting from    //
// TObject can be inserted in a TSortedList.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TList
#include "TList.h"
#endif


class TSortedList : public TList {

public:
   TSortedList(Bool_t order = kSortAscending) { fAscending = order; }

   void      Add(TObject *obj);
   void      Add(TObject *obj, Option_t *opt);

   Bool_t    IsSorted() const { return kTRUE; }

   //---- the following methods are overridden to preserve sorting order
   void      AddFirst(TObject *obj) { Add(obj); }
   void      AddFirst(TObject *obj, Option_t *opt) { Add(obj, opt); }
   void      AddLast(TObject *obj) { Add(obj); }
   void      AddLast(TObject *obj, Option_t *opt) { Add(obj, opt); }
   void      AddAt(TObject *obj, Int_t) { Add(obj); }
   void      AddAfter(const TObject *, TObject *obj) { Add(obj); }
   void      AddAfter(TObjLink *, TObject *obj) { Add(obj); }
   void      AddBefore(const TObject *, TObject *obj) { Add(obj); }
   void      AddBefore(TObjLink *, TObject *obj) { Add(obj); }
   void      Sort(Bool_t = kSortAscending) { }

   ClassDef(TSortedList,0)  //A sorted list
};

#endif

