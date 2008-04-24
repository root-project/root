// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOrdCollection                                                       //
//                                                                      //
// Ordered collection. An ordered collection has TList insertion        //
// semantics but is implemented using an array of TObject*'s. It uses   //
// less space than a TList (since there is no need for the prev and     //
// next pointers), but it is more costly to insert objects (since it    //
// has to create a gap by copying object pointers). TOrdCollection      //
// is better than TList when objects are only added at the end of the   //
// collection since no copying needs to be done.                        //
//Begin_Html
/*
<img src=gif/tordcollection.gif>
*/
//End_Html
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TOrdCollection.h"
#include "TError.h"


ClassImp(TOrdCollection)

//______________________________________________________________________________
TOrdCollection::TOrdCollection(Int_t capacity)
{
   // Create an ordered collection.

   if (capacity < 0) {
      Warning("TOrdCollection", "capacity (%d) < 0", capacity);
      capacity = kDefaultCapacity;
   } else if (capacity == 0)
      capacity = kDefaultCapacity;
   Init(capacity);
}

//______________________________________________________________________________
TOrdCollection::~TOrdCollection()
{
   // Delete the collection. Objects are not deleted unless the TOrdCollection
   // is the owner (set via SetOwner()).

   if (IsOwner())
      Delete();

   TStorage::Dealloc(fCont);
   fCont = 0;
   fSize = 0;
}

//______________________________________________________________________________
void TOrdCollection::AddAt(TObject *obj, Int_t idx)
{
   // Insert object at position idx in the collection.

   Int_t physIdx;

   if (idx > fSize) idx = fSize;

   if (fGapSize <= 0)
      SetCapacity(GrowBy(TMath::Max(fCapacity, (int)kMinExpand)));

   if (idx == fGapStart) {
      physIdx = fGapStart;
      fGapStart++;
   } else {
      physIdx = PhysIndex(idx);
      if (physIdx < fGapStart) {
         MoveGapTo(physIdx);
         physIdx = fGapStart;
         fGapStart++;
      } else {
         MoveGapTo(physIdx - fGapSize);
         physIdx = fGapStart + fGapSize - 1;
      }
   }
   R__ASSERT(physIdx >= 0 && physIdx < fCapacity);
   fCont[physIdx] = obj;
   fGapSize--;
   fSize++;
   Changed();
}

//______________________________________________________________________________
void TOrdCollection::AddFirst(TObject *obj)
{
   // Insert object at beginning of collection.

   AddAt(obj, 0);
}

//______________________________________________________________________________
void TOrdCollection::AddLast(TObject *obj)
{
   // Add object at the end of the collection.

   AddAt(obj, fSize);
}

//______________________________________________________________________________
void TOrdCollection::AddBefore(const TObject *before, TObject *obj)
{
   // Insert object before object before in the collection.

   if (!before)
      AddFirst(obj);
   else {
      Int_t idx = IndexOf(before);
      if (idx == -1) {
         Error("AddBefore", "before not found, object not added");
         return;
      }
      if (idx == 0) {
         AddFirst(obj);
         return;
      }
      AddAt(obj, idx);
   }
}

//______________________________________________________________________________
void TOrdCollection::AddAfter(const TObject *after, TObject *obj)
{
   // Insert object after object after in the collection.

   if (!after)
      AddLast(obj);
   else {
      Int_t idx = IndexOf(after);
      if (idx == -1) {
         Error("AddAfter", "after not found, object not added");
         return;
      }
      AddAt(obj, idx+1);
   }
}

//______________________________________________________________________________
TObject *TOrdCollection::After(const TObject *obj) const
{
   // Return the object after object obj. Returns 0 if obj is last
   // in collection.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj);
   if (idx == -1 || idx == fSize-1) return 0;

   return At(idx+1);
}

//______________________________________________________________________________
TObject *TOrdCollection::At(Int_t idx) const
{
   // Returns the object at position idx. Returns 0 if idx is out of range.

   if (IllegalIndex("At", idx)) return 0;
   return fCont[PhysIndex(idx)];
}

//______________________________________________________________________________
TObject *TOrdCollection::Before(const TObject *obj) const
{
   // Returns the object before object obj. Returns 0 if obj is first
   // in collection.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj);
   if (idx == -1 || idx == 0) return 0;

   return At(idx-1);
}

//______________________________________________________________________________
void TOrdCollection::Clear(Option_t *)
{
   // Remove all objects from the collection. Does not delete the objects
   // unless the TOrdCollection is the owner (set via SetOwner()).

   if (IsOwner())
      Delete();
   else {
      TStorage::Dealloc(fCont);
      fCont = 0;
      Init(fCapacity);
      fSize = 0;
   }
}

//______________________________________________________________________________
void TOrdCollection::Delete(Option_t *)
{
   // Remove all objects from the collection AND delete all heap based objects.

   for (Int_t i = 0; i < fSize; i++) {
      TObject *obj = At(i);
      if (obj && obj->IsOnHeap())
         TCollection::GarbageCollect(obj);
   }
   TStorage::Dealloc(fCont);
   fCont = 0;
   Init(fCapacity);
   fSize = 0;
}

//______________________________________________________________________________
TObject *TOrdCollection::First() const
{
   // Return the first object in the collection. Returns 0 when collection
   // is empty.

   return At(0);
}

//______________________________________________________________________________
TObject **TOrdCollection::GetObjectRef(const TObject *obj) const
{
   // return address of pointer obj
   Int_t index = IndexOf(obj);
   return &fCont[index];
}

//______________________________________________________________________________
TObject *TOrdCollection::Last() const
{
   // Return the last object in the collection. Returns 0 when collection
   // is empty.

   return At(fSize-1);
}

//______________________________________________________________________________
Bool_t TOrdCollection::IllegalIndex(const char *method, Int_t idx) const
{
   // Return true when index out of bounds and print error.

   if (idx < 0 || idx >= fSize) {
      Error(method, "index error (= %d) < 0 or > Size() (= %d)", idx, fSize);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
Int_t TOrdCollection::IndexOf(const TObject *obj) const
{
   // Return index of object in collection. Returns -1 when object not found.
   // Uses member IsEqual() to find object.

   for (Int_t i = 0; i < GetSize(); i++)
      if (fCont[PhysIndex(i)]->IsEqual(obj))
         return i;

   return -1;
}

//______________________________________________________________________________
void TOrdCollection::Init(Int_t capacity)
{
   // Initialize ordered collection.

   fCapacity = capacity;
   fCont = (TObject**) TStorage::Alloc(fCapacity*sizeof(TObject*)); //new TObject* [fCapacity];
   memset(fCont, 0, fCapacity*sizeof(TObject*));
   fGapStart = 0;
   fGapSize  = capacity;
   Changed();
}

//______________________________________________________________________________
TIterator *TOrdCollection::MakeIterator(Bool_t dir) const
{
   // Return an ordered collection iterator.

   return new TOrdCollectionIter(this, dir);
}

//______________________________________________________________________________
void TOrdCollection::MoveGapTo(Int_t start)
{
   // Move gap to new position. Gap needs to be moved when objects are
   // inserted not at the end.

   Int_t i;

   R__ASSERT(start + fGapSize - 1 < fCapacity);

   if (fGapSize <= 0) {
      fGapStart = start;
      return;
   }
   if (start < fGapStart) {
      for (i = fGapStart - 1; i >= start; i--)
         fCont[i + fGapSize] = fCont[i];
   } else if (start > fGapStart) {
      Int_t stop = start + fGapSize;
      for (i = fGapStart + fGapSize; i < stop; i++)
         fCont[i - fGapSize] = fCont[i];
   }
   fGapStart = start;
   for (i = fGapStart; i < fGapStart + fGapSize; i++)
      fCont[i] = 0;
}

//______________________________________________________________________________
void TOrdCollection::PutAt(TObject *obj, Int_t idx)
{
   // Put object at index idx. Overwrites what was at idx before.

   if (IllegalIndex("PutAt", idx)) return;

   Int_t phx = PhysIndex(idx);
   R__ASSERT(phx >= 0 && phx < fCapacity);
   fCont[phx] = obj;
   Changed();
}

//______________________________________________________________________________
TObject *TOrdCollection::RemoveAt(Int_t idx)
{
   // Remove object at index idx.

   Int_t physIdx;

   if (idx == fGapStart - 1 || idx == fGapStart) {
      if (idx == fGapStart)
         physIdx = fGapStart + fGapSize;        // at right boundary
      else
         physIdx = --fGapStart;                 // at left boundary
   } else {
      physIdx = PhysIndex(idx);
      if (physIdx < fGapStart) {
         MoveGapTo(physIdx + 1);
         physIdx = --fGapStart;
      } else {
         MoveGapTo(physIdx - fGapSize);
         physIdx = fGapStart + fGapSize;
      }
   }
   R__ASSERT(physIdx >= 0 && physIdx < fCapacity);
   TObject *obj = fCont[physIdx];
   fCont[physIdx] = 0;
   fGapSize++;
   fSize--;
   Changed();

   if (LowWaterMark()) {
      Int_t newCapacity = TMath::Max(int(fCapacity / kShrinkFactor), 1);
      if (fCapacity > newCapacity)
         SetCapacity(newCapacity);
   }
   return obj;
}

//______________________________________________________________________________
TObject *TOrdCollection::Remove(TObject *obj)
{
   // Remove object from collection.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj);
   if (idx == -1) return 0;

   return RemoveAt(idx);
}

//______________________________________________________________________________
void TOrdCollection::SetCapacity(Int_t newCapacity)
{
   // Set/change ordered collection capacity.

   R__ASSERT(newCapacity > 0);
   R__ASSERT(fSize <= newCapacity);

   if (fCapacity == newCapacity) return;

   Int_t newGapSize = newCapacity - fSize;
   MoveGapTo(fCapacity - fGapSize);
   fCont = (TObject**) TStorage::ReAlloc(fCont, newCapacity*sizeof(TObject*),
                                         fCapacity*sizeof(TObject*));
   fGapSize  = newGapSize;
   fCapacity = newCapacity;
}

//______________________________________________________________________________
void TOrdCollection::Sort()
{
   // If objects in collection are sortable (i.e. IsSortable() returns true
   // for all objects) then sort collection.

   if (fSize <= 0 || fSorted) return;
   if (!At(0)->IsSortable()) {
      Error("Sort", "objects in collection are not sortable");
      return;
   }

   MoveGapTo(fCapacity - fGapSize);
   QSort(fCont, 0, fSize);

   fSorted = kTRUE;
}

//______________________________________________________________________________
Int_t TOrdCollection::BinarySearch(TObject *obj)
{
   // Find object using a binary search. Collection must first have been
   // sorted.

   Int_t result;

   if (!obj) return -1;

   if (!fSorted) {
      Error("BinarySearch", "collection must first be sorted");
      return -1;
   }

   MoveGapTo(fCapacity - fGapSize);

   Int_t base = 0;
   Int_t last = base + GetSize() - 1;
   while (last >= base) {
      Int_t position = (base + last) / 2;
      TObject *obj2 = fCont[position];
      if ((obj2 == 0) || (result = obj->Compare(obj2)) == 0)
         return LogIndex(position);
      if (result < 0)
         last = position - 1;
      else
         base = position + 1;
   }
   return -1;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOrdCollectionIter                                                   //
//                                                                      //
// Iterator of ordered collection.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TOrdCollectionIter)

//______________________________________________________________________________
TOrdCollectionIter::TOrdCollectionIter(const TOrdCollection *col, Bool_t dir): fCol(col), fDirection(dir)
{
   // Create collection iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.

   Reset();
}

//______________________________________________________________________________
TOrdCollectionIter::TOrdCollectionIter(const TOrdCollectionIter &iter) : TIterator(iter)
{
   // Copy ctor.

   fCol       = iter.fCol;
   fDirection = iter.fDirection;
   fCursor    = iter.fCursor;
   fCurCursor = iter.fCurCursor;
}

//______________________________________________________________________________
TIterator &TOrdCollectionIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TOrdCollectionIter::Class()) {
      const TOrdCollectionIter &rhs1 = (const TOrdCollectionIter &)rhs;
      fCol       = rhs1.fCol;
      fDirection = rhs1.fDirection;
      fCursor    = rhs1.fCursor;
      fCurCursor = rhs1.fCurCursor;
   }
   return *this;
}

//______________________________________________________________________________
TOrdCollectionIter &TOrdCollectionIter::operator=(const TOrdCollectionIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fCol       = rhs.fCol;
      fDirection = rhs.fDirection;
      fCursor    = rhs.fCursor;
      fCurCursor = rhs.fCurCursor;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TOrdCollectionIter::Next()
{
   // Return next object in collection. Returns 0 when no more objects in
   // collection.

   fCurCursor = fCursor;
   if (fDirection == kIterForward) {
      if (fCursor < fCol->GetSize())
         return fCol->At(fCursor++);
   } else {
      if (fCursor >= 0)
         return fCol->At(fCursor--);
   }
   return 0;
}

//______________________________________________________________________________
void TOrdCollectionIter::Reset()
{
   // Reset collection iterator.

   if (fDirection == kIterForward)
      fCursor = 0;
   else
      fCursor = fCol->GetSize() - 1;
   
   fCurCursor = fCursor;
}

//______________________________________________________________________________
bool TOrdCollectionIter::operator!=(const TIterator &aIter) const
{
   // This operator compares two TIterator objects.

   if (nullptr == (&aIter))
      return (fCurCursor < fCol->GetSize());

   if (aIter.IsA() == TOrdCollectionIter::Class()) {
      const TOrdCollectionIter &iter(dynamic_cast<const TOrdCollectionIter &>(aIter));
      return (fCurCursor != iter.fCurCursor);
   }
   return false; // for base class we don't implement a comparison
}

//______________________________________________________________________________
bool TOrdCollectionIter::operator!=(const TOrdCollectionIter &aIter) const
{
   // This operator compares two TOrdCollectionIter objects.

   if (nullptr == (&aIter))
      return (fCurCursor < fCol->GetSize());

   return (fCurCursor != aIter.fCurCursor);
}

//______________________________________________________________________________
TObject *TOrdCollectionIter::operator*() const
{
   // Return current object or nullptr.

   return (((fCurCursor >= 0) && (fCurCursor < fCol->GetSize())) ?
           fCol->At(fCurCursor) : nullptr);
}
