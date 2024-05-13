// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THashList
#define ROOT_THashList


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THashList                                                            //
//                                                                      //
// THashList implements a hybrid collection class consisting of a       //
// hash table and a list to store TObject's. The hash table is used for //
// quick access and lookup of objects while the list allows the objects //
// to be ordered. The hash value is calculated using the value returned //
// by the TObject's Hash() function. Each class inheriting from TObject //
// can override Hash() as it sees fit.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"

class THashTable;


class THashList : public TList {

protected:
   THashTable   *fTable;    //Hashtable used for quick lookup of objects

private:
   THashList(const THashList&) = delete;
   THashList& operator=(const THashList&) = delete;

public:
   THashList(Int_t capacity=TCollection::kInitHashTableCapacity, Int_t rehash=0);
   THashList(TObject *parent, Int_t capacity=TCollection::kInitHashTableCapacity, Int_t rehash=0);
   virtual    ~THashList();
   Float_t    AverageCollisions() const;
   void       Clear(Option_t *option="") override;
   void       Delete(Option_t *option="") override;

   TObject   *FindObject(const char *name) const override;
   TObject   *FindObject(const TObject *obj) const override;

   const TList *GetListForObject(const char *name) const;
   const TList *GetListForObject(const TObject *obj) const;

   void       AddFirst(TObject *obj) override;
   void       AddFirst(TObject *obj, Option_t *opt) override;
   void       AddLast(TObject *obj) override;
   void       AddLast(TObject *obj, Option_t *opt) override;
   void       AddAt(TObject *obj, Int_t idx) override;
   void       AddAfter(const TObject *after, TObject *obj) override;
   void       AddAfter(TObjLink *after, TObject *obj) override;
   void       AddBefore(const TObject *before, TObject *obj) override;
   void       AddBefore(TObjLink *before, TObject *obj) override;
   void       RecursiveRemove(TObject *obj) override;
   void       Rehash(Int_t newCapacity);
   TObject   *Remove(TObject *obj) override;
   TObject   *Remove(TObjLink *lnk) override;
   bool       UseRWLock(Bool_t enable = true) override;

   ClassDefOverride(THashList,0)  //Doubly linked list with hashtable for lookup
};

#endif
