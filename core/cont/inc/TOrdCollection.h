// @(#)root/cont:$Id$
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

#include "TSeqCollection.h"

#include <iterator>


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

   TOrdCollection(const TOrdCollection&) = delete;
   TOrdCollection& operator=(const TOrdCollection&) = delete;

public:
   enum { kDefaultCapacity = 1, kMinExpand = 8, kShrinkFactor = 2 };

   typedef TOrdCollectionIter Iterator_t;

   TOrdCollection(Int_t capacity = kDefaultCapacity);
   ~TOrdCollection();
   void          Clear(Option_t *option="") override;
   void          Delete(Option_t *option="") override;
   TObject     **GetObjectRef(const TObject *obj) const override;
   Int_t         IndexOf(const TObject *obj) const override;
   TIterator    *MakeIterator(Bool_t dir = kIterForward) const override;

   void          AddFirst(TObject *obj) override;
   void          AddLast(TObject *obj) override;
   void          AddAt(TObject *obj, Int_t idx) override;
   void          AddAfter(const TObject *after, TObject *obj) override;
   void          AddBefore(const TObject *before, TObject *obj) override;
   void          PutAt(TObject *obj, Int_t idx);
   TObject      *RemoveAt(Int_t idx) override;
   TObject      *Remove(TObject *obj) override;

   TObject      *At(Int_t idx) const override;
   TObject      *Before(const TObject *obj) const override;
   TObject      *After(const TObject *obj) const override;
   TObject      *First() const override;
   TObject      *Last() const override;

   void          Sort();
   Int_t         BinarySearch(TObject *obj);

   ClassDefOverride(TOrdCollection,0)  //An ordered collection
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
   Int_t                  fCurCursor; //current position in collection
   Int_t                  fCursor;    //next position in collection
   Bool_t                 fDirection; //iteration direction

   TOrdCollectionIter() : fCol(nullptr), fCurCursor(0), fCursor(0), fDirection(kIterForward) { }

public:
   using iterator_category = std::bidirectional_iterator_tag;
   using value_type = TObject *;
   using difference_type = std::ptrdiff_t;
   using pointer = TObject **;
   using const_pointer = const TObject **;
   using reference = const TObject *&;

   TOrdCollectionIter(const TOrdCollection *col, Bool_t dir = kIterForward);
   TOrdCollectionIter(const TOrdCollectionIter &iter);
   ~TOrdCollectionIter() { }
   TIterator          &operator=(const TIterator &rhs) override;
   TOrdCollectionIter &operator=(const TOrdCollectionIter &rhs);

   const TCollection *GetCollection() const override { return fCol; }
   TObject           *Next() override;
   void               Reset() override;
   Bool_t             operator!=(const TIterator &aIter) const override;
   Bool_t             operator!=(const TOrdCollectionIter &aIter) const;
   TObject           *operator*() const override;

   ClassDefOverride(TOrdCollectionIter,0)  //Ordered collection iterator
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
