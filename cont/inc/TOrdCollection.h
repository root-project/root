// @(#)root/cont:$Name$:$Id$
// Author: Fons Rademakers   13/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOrdCollection
#define ROOT_TOrdCollection


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOrdCollection                                                       //
//                                                                      //
// Ordered collection.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif

class TOrdCollectionIter;


class TOrdCollection : public TSeqCollection {

friend class  TOrdCollectionIter;

private:
   TObject  **fCont;
   Int_t      fCapacity;
   Int_t      fGapStart;
   Int_t      fGapSize;

   Int_t      PhysIndex(Int_t idx) const;
   Int_t      LogIndex(Int_t idx) const;
   void       MoveGapTo(Int_t newGapStart);
   Bool_t     IllegalIndex(const char *method, Int_t idx) const;
   void       Init(Int_t capacity);
   Bool_t     LowWaterMark() const;
   void       SetCapacity(Int_t newCapacity);

public:
   enum { kDefaultCapacity = 1, kMinExpand = 8, kShrinkFactor = 2 };

   TOrdCollection(Int_t capacity = kDefaultCapacity);
   ~TOrdCollection();
   void          Clear(Option_t *option="");
   void          Delete(Option_t *option="");
   Int_t         IndexOf(TObject *obj) const; // returns -1 if not found
   TIterator    *MakeIterator(Bool_t dir = kIterForward) const;

   void          AddFirst(TObject *obj);
   void          AddLast(TObject *obj);
   void          AddAt(TObject *obj, Int_t idx);
   void          AddAfter(TObject *after, TObject *obj);
   void          AddBefore(TObject *before, TObject *obj);
   void          PutAt(TObject *obj, Int_t idx);
   TObject      *RemoveAt(Int_t idx);
   TObject      *Remove(TObject *obj);

   TObject      *At(Int_t idx) const;
   TObject      *Before(TObject *obj) const;
   TObject      *After(TObject *obj) const;
   TObject      *First() const;
   TObject      *Last() const;

   void          Sort();
   Int_t         BinarySearch(TObject *obj);

   ClassDef(TOrdCollection,0)  //An ordered collection
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOrdCollectionIter                                                   //
//                                                                      //
// Iterator of ordered collection.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TOrdCollectionIter : public TIterator {

private:
   const TOrdCollection  *fCol;       //collection being iterated
   Int_t                  fCursor;    //current position in collection
   Bool_t                 fDirection; //iteration direction

   TOrdCollectionIter() : fCol(0) { }

public:
   TOrdCollectionIter(const TOrdCollection *col, Bool_t dir = kIterForward);
   TOrdCollectionIter(const TOrdCollectionIter &iter);
   ~TOrdCollectionIter() { }
   TIterator          &operator=(const TIterator &rhs);
   TOrdCollectionIter &operator=(const TOrdCollectionIter &rhs);

   const TCollection *GetCollection() const { return fCol; }
   TObject           *Next();
   void              Reset();

   ClassDef(TOrdCollectionIter,0)  //Ordered collection iterator
};


//---- inlines -----------------------------------------------------------------

inline Bool_t TOrdCollection::LowWaterMark() const
{
   return (fSize < (fCapacity / 4) && fSize > TCollection::kInitCapacity);
}

inline Int_t TOrdCollection::PhysIndex(Int_t idx) const
   { return (idx < fGapStart) ? idx : idx + fGapSize; }

inline Int_t TOrdCollection::LogIndex(Int_t idx) const
   { return (idx < fGapStart) ? idx : idx - fGapSize; }

#endif

