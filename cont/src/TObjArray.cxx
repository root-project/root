// @(#)root/cont:$Name:  $:$Id: TObjArray.cxx,v 1.9 2001/04/04 14:07:26 brun Exp $
// Author: Fons Rademakers   11/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjArray                                                            //
//                                                                      //
// An array of TObjects. The array expands automatically when           //
// objects are added (shrinking can be done by hand using Expand(),     //
// how nice to have meaningful names -:)).                              //
// Use operator[] to have "real" array behaviour.                       //
//Begin_Html
/*
<img src=gif/tobjarray.gif>
*/
//End_Html
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjArray.h"
#include "TMath.h"
#include "TError.h"
#include "TClass.h"
#include "TROOT.h"

ClassImp(TObjArray)

//______________________________________________________________________________
TObjArray::TObjArray(Int_t s, Int_t lowerBound)
{
   // Create an object array. Using s one can set the array size (default is
   // kInitCapacity=16) and lowerBound can be used to set the array lowerbound
   // index (default is 0).

   if (s < 0) {
      Warning("TObjArray", "size (%d) < 0", s);
      s = TCollection::kInitCapacity;
   } else if (s == 0)
      s = TCollection::kInitCapacity;
   fCont = 0;
   Init(s, lowerBound);
}

//______________________________________________________________________________
TObjArray::TObjArray(const TObjArray &a)
{
   // Create a copy of TObjArray a. Note, does not copy the kIsOwner flag.

   fCont = 0;
   Init(a.fSize, a.fLowerBound);

   for (Int_t i = 0; i < fSize; i++)
      fCont[i] = a.fCont[i];

   fLast = a.fLast;
   fName = a.fName;
}

//______________________________________________________________________________
TObjArray::~TObjArray()
{
   // Delete an array. Objects are not deleted unless the TObjArray is the
   // owner (set via SetOwner()).

   if (IsOwner())
      Delete();

   delete [] fCont;
   fCont = 0;
   fSize = 0;
}

//______________________________________________________________________________
void TObjArray::AddFirst(TObject *obj)
{
   // Add object in the first slot of the array. This will overwrite the
   // first element that might have been there. To have insertion semantics
   // use either a TList or a TOrdCollection.

   fCont[0] = obj;
   Changed();
}

//______________________________________________________________________________
void TObjArray::AddLast(TObject *obj)
{
   // Add object in the next empty slot in the array. Expand the array
   // if necessary.

   AddAtAndExpand(obj, GetAbsLast()+1+fLowerBound);
}

//______________________________________________________________________________
void TObjArray::AddBefore(TObject *before, TObject *obj)
{
   // Add object in the slot before object before. If before=0 add object
   // in the first slot. Note that this will overwrite any object that
   // might have already been in this slot. For insertion semantics use
   // either a TList or a TOrdCollection.

   if (!before)
      AddFirst(obj);
   else {
      Int_t idx = IndexOf(before) - fLowerBound;
      if (idx == -1) {
         Error("AddBefore", "before not found, object not added");
         return;
      }
      if (idx == 0) {
         Error("AddBefore", "cannot add before lowerbound (%d)", fLowerBound);
         return;
      }
      AddAt(obj, idx+fLowerBound-1);
   }
}

//______________________________________________________________________________
void TObjArray::AddAfter(TObject *after, TObject *obj)
{
   // Add object in the slot after object after. If after=0 add object in
   // the last empty slot. Note that this will overwrite any object that
   // might have already been in this slot. For insertion semantics use
   // either a TList or a TOrdCollection.

   if (!after)
      AddLast(obj);
   else {
      Int_t idx = IndexOf(after) - fLowerBound;
      if (idx == -1) {
         Error("AddAfter", "after not found, object not added");
         return;
      }
      AddAtAndExpand(obj, idx+fLowerBound+1);
   }
}

//______________________________________________________________________________
void TObjArray::AddAtAndExpand(TObject *obj, Int_t idx)
{
   // Add object at position idx. If idx is larger than the current size
   // of the array, expand the array (double its size).

   if (idx < fLowerBound) {
      Error("AddAt", "out of bounds at %d in %x", idx, this);
      return;
   }
   if (idx-fLowerBound >= fSize)
      Expand(TMath::Max(idx-fLowerBound+1, GrowBy(fSize)));
   fCont[idx-fLowerBound] = obj;
   fLast = TMath::Max(idx-fLowerBound, GetAbsLast());
   Changed();
}

//______________________________________________________________________________
void TObjArray::AddAt(TObject *obj, Int_t idx)
{
   // Add object at position ids. Give an error when idx is out of bounds
   // (i.e. the array is not expanded).

   if (!BoundsOk("AddAt", idx)) return;

   fCont[idx-fLowerBound] = obj;
   fLast = TMath::Max(idx-fLowerBound, GetAbsLast());
   Changed();
}

//______________________________________________________________________________
Int_t  TObjArray::AddAtFree(TObject *obj)
{
   // Return the position of the new object.
   // Find the first empty cell or AddLast if there is no empty cell

   if (Last()) {    // <---------- This is to take in account "empty" TObjArray's
       Int_t i;
       for (i = 0; i < fSize; i++)
          if (!fCont[i]) {         // Add object at position i
             fCont[i] = obj;
             fLast = TMath::Max(i, GetAbsLast());
             Changed();
             return i+fLowerBound;
          }
   }
   AddLast(obj);
   return GetLast();
}

//______________________________________________________________________________
TObject *TObjArray::After(TObject *obj) const
{
   // Return the object after obj. Returns 0 if obj is last object.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj) - fLowerBound;
   if (idx == -1 || idx == fSize-1) return 0;

   return fCont[idx+1];
}

//______________________________________________________________________________
TObject *TObjArray::At(Int_t i) const
{
   // Return the object at position i. Returns 0 if i is out of bounds.

   if (BoundsOk("At", i))
      return fCont[i-fLowerBound];
   return 0;
}

//______________________________________________________________________________
TObject *TObjArray::Before(TObject *obj) const
{
   // Return the object before obj. Returns 0 if obj is first object.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj) - fLowerBound;
   if (idx == -1 || idx == 0) return 0;

   return fCont[idx-1];
}

//______________________________________________________________________________
void TObjArray::Clear(Option_t *)
{
   // Remove all objects from the array. Does not delete the objects
   // unless the TObjArray is the owner (set via SetOwner()).

   if (IsOwner())
      Delete();
   else
      Init(fSize, fLowerBound);
}

//______________________________________________________________________________
void TObjArray::Compress()
{
   // Remove empty slots from array.

   Int_t j = 0;

   for (Int_t i = 0; i < fSize; i++) {
      if (fCont[i]) {
         fCont[j] = fCont[i];
         j++;
      }
   }

   fLast = j - 1;

   for ( ; j < fSize; j++)
      fCont[j] = 0;
}

//______________________________________________________________________________
void TObjArray::Delete(Option_t *)
{
   // Remove all objects from the array AND delete all heap based objects.

   for (Int_t i = 0; i < fSize; i++)
      if (fCont[i] && fCont[i]->IsOnHeap()) {
         TCollection::GarbageCollect(fCont[i]);
         fCont[i] = 0;
      }

   Init(fSize, fLowerBound);
}

//______________________________________________________________________________
void TObjArray::Expand(Int_t newSize)
{
   // Expand or shrink the array to newSize elements.

   if (newSize < 0) {
      Error ("Expand", "newSize must be positive (%d)", newSize);
      return;
   }
   if (newSize == fSize)
      return;
   if (newSize < fSize) {
      // if the array is shrinked check whether there are nonempty entries
      for (Int_t j = newSize; j < fSize; j++)
         if (fCont[j]) {
            Error ("Expand", "expand would cut off nonempty entries at %d", j);
            return;
         }
   }
   fCont = (TObject**) TStorage::ReAlloc(fCont, newSize * sizeof(TObject*),
                                         fSize * sizeof(TObject*));
   fSize = newSize;
}

//_______________________________________________________________________
void TObjArray::Streamer(TBuffer &b)
{
   // Stream all objects in the array to or from the I/O buffer.

   UInt_t R__s, R__c;
   Int_t nobjects;
   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 2)
         TObject::Streamer(b);
      if (v > 1)
         fName.Streamer(b);
      b >> nobjects;
      b >> fLowerBound;
      if (nobjects >= fSize) Expand(nobjects);
      fLast = -1;
      TObject *obj;
      for (Int_t i = 0; i < nobjects; i++) {
         b >> obj;
         if (obj) {
            fCont[i] = obj;
            fLast = i;
         }
      }
      Changed();
      b.CheckByteCount(R__s, R__c,TObjArray::IsA());
   } else {
      R__c = b.WriteVersion(TObjArray::IsA(), kTRUE);
      TObject::Streamer(b);
      fName.Streamer(b);
      nobjects = GetLast()+1;
      b << nobjects;
      b << fLowerBound;

      for (Int_t i = 0; i < nobjects; i++) {
         b << fCont[i];
      }
      b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TObject *TObjArray::First() const
{
   // Return the object in the first slot.

   return fCont[0];
}

//______________________________________________________________________________
TObject *TObjArray::Last() const
{
   // Return the object in the last filled slot. Returns 0 if no entries.

   if (fLast == -1)
      return 0;
   else
      return fCont[GetAbsLast()];
}

//______________________________________________________________________________
Int_t TObjArray::GetEntries() const
{
   // Return the number of objects in array (i.e. number of non-empty slots).
   // Attention: use this method ONLY if you want to know the number of
   // non-empty slots. This function loops over the complete array and
   // is therefore very slow when applied in a loop. Most of the time you
   // better use GetLast()+1.

   Int_t cnt = 0;

   for (Int_t i = 0; i < fSize; i++)
      if (fCont[i]) cnt++;

   return cnt;
}

//______________________________________________________________________________
Int_t TObjArray::GetAbsLast() const
{
   // Return absolute index to last object in array. Returns -1 in case
   // array is empty.

   // For efficiency we need sometimes to update fLast so we have
   // to cast const away. Ugly, but making GetAbsLast() not const breaks
   // many other const functions.
   if (fLast == -2) {
      for (Int_t i = fSize-1; i >= 0; i--)
         if (fCont[i]) {
            ((TObjArray*)this)->fLast = i;
            return fLast;
         }
      ((TObjArray*)this)->fLast = -1;
   }
   return fLast;
}

//______________________________________________________________________________
Int_t TObjArray::GetLast() const
{
   // Return index of last object in array. Returns lowerBound-1 in case
   // array is empty.

   return fLowerBound+GetAbsLast();
}

//______________________________________________________________________________
TObject **TObjArray::GetObjectRef(TObject *obj) const
{
   // Return address of pointer obj.

   Int_t index = IndexOf(obj);
   return &fCont[index];
}

//______________________________________________________________________________
Int_t TObjArray::IndexOf(const TObject *obj) const
{
   // obj != 0 Return index of object in array.
   //          Returns lowerBound-1 in case array doesn't contain the obj.
   //
   // obj == 0 Return the index of the first empty slot.
   //          Returns lowerBound-1 in case array doesn't contain any empty slot.

   Int_t i;
   if (obj) {
     for (i = 0; i < fSize; i++)
        if (fCont[i] && fCont[i]->IsEqual(obj))
           return i+fLowerBound;
   } else {    // Look for the first empty slot
     for (i = 0; i < fSize; i++)
        if (!fCont[i])
           return i+fLowerBound;
   }

   return fLowerBound-1;
}

//______________________________________________________________________________
void TObjArray::Init(Int_t s, Int_t lowerBound)
{
   // Initialize a TObjArray.

   if (fCont && fSize != s) {
      delete [] fCont;
      fCont = 0;
   }

   fSize = s;

   if (!fCont)
      fCont = new TObject* [fSize];
   memset(fCont, 0, fSize*sizeof(TObject*));
   fLowerBound = lowerBound;
   fLast = -1;
   Changed();
}

//______________________________________________________________________________
TIterator *TObjArray::MakeIterator(Bool_t dir) const
{
   // Returns an array iterator.

   return new TObjArrayIter(this, dir);
}

//______________________________________________________________________________
Bool_t TObjArray::OutOfBoundsError(const char *where, Int_t i) const
{
   // Generate an out-of-bounds error. Always returns false.

   Error(where, "index %d out of bounds (size: %d, this: 0x%08x)", i, fSize, this);
   return kFALSE;
}

//______________________________________________________________________________
TObject *TObjArray::RemoveAt(Int_t idx)
{
   // Remove object at index idx.

   if (!BoundsOk("RemoveAt", idx)) return 0;

   int i = idx-fLowerBound;

   TObject *obj = 0;
   if (fCont[i]) {
      obj = fCont[i];
      fCont[i] = 0;
      // recalculate array size
      if (i == fLast)
         do { fLast--; } while (fLast >= 0 && fCont[fLast] == 0);
      Changed();
   }
   return obj;
}

//______________________________________________________________________________
TObject *TObjArray::Remove(TObject *obj)
{
   // Remove object from array.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj) - fLowerBound;

   if (idx == -1) return 0;

   TObject *ob = fCont[idx];
   fCont[idx] = 0;
   // recalculate array size
   if (idx == fLast)
      do { fLast--; } while (fLast >= 0 && fCont[fLast] == 0);
   Changed();
   return ob;
}

//______________________________________________________________________________
void TObjArray::SetLast(Int_t last)
{
   // Set index of last object in array, effectively truncating the
   // array. Use carefully since whenever last position has to be
   // recalculated, e.g. after a Remove() or Sort() it will be reset
   // to the last non-empty slot.

   if (BoundsOk("SetLast", last))
      fLast = last - fLowerBound;
}

//______________________________________________________________________________
void TObjArray::Sort(Int_t upto)
{
   // If objects in array are sortable (i.e. IsSortable() returns true
   // for all objects) then sort array.

   if (GetAbsLast() == -1 || fSorted) return;
   for (Int_t i = 0; i < fSize; i++)
      if (fCont[i]) {
         if (!fCont[i]->IsSortable()) {
            Error("Sort", "objects in array are not sortable");
            return;
         }
      }

   QSort(fCont, 0, TMath::Min(fSize, upto-fLowerBound));

   fLast   = -2;
   fSorted = kTRUE;
}

//______________________________________________________________________________
Int_t TObjArray::BinarySearch(TObject *op, Int_t upto)
{
   // Find object using a binary search. Array must first have been sorted.
   // Search can be limited by setting upto to desired index.

   Int_t   base, position, last, result = 0;
   TObject *op2;

   if (!op) return -1;

   if (!fSorted) {
      Error("BinarySearch", "array must first be sorted");
      return -1;
   }

   base = 0;
   last = TMath::Min(fSize, upto-fLowerBound) - 1;

   while (last >= base) {
      position = (base+last) / 2;
      op2 = fCont[position];
      if (op2 && (result = op->Compare(op2)) == 0)
         return position + fLowerBound;
      if (!op2 || result < 0)
         last = position-1;
      else
         base = position+1;
   }
   return -1;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjArrayIter                                                        //
//                                                                      //
// Iterator of object array.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TObjArrayIter)

//______________________________________________________________________________
TObjArrayIter::TObjArrayIter(const TObjArray *arr, Bool_t dir)
{
   // Create array iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.

   fArray     = arr;
   fDirection = dir;
   Reset();
}

//______________________________________________________________________________
TObjArrayIter::TObjArrayIter(const TObjArrayIter &iter)
{
   // Copy ctor.

   fArray     = iter.fArray;
   fDirection = iter.fDirection;
   fCursor    = iter.fCursor;
}

//______________________________________________________________________________
TIterator &TObjArrayIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TObjArrayIter::Class()) {
      const TObjArrayIter &rhs1 = (const TObjArrayIter &)rhs;
      fArray     = rhs1.fArray;
      fDirection = rhs1.fDirection;
      fCursor    = rhs1.fCursor;
   }
   return *this;
}

//______________________________________________________________________________
TObjArrayIter &TObjArrayIter::operator=(const TObjArrayIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fArray     = rhs.fArray;
      fDirection = rhs.fDirection;
      fCursor    = rhs.fCursor;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TObjArrayIter::Next()
{
   // Return next object in array. Returns 0 when no more objects in array.

   if (fDirection == kIterForward) {
      for ( ; fCursor < fArray->Capacity() && fArray->fCont[fCursor] == 0;
              fCursor++) { }

      if (fCursor < fArray->Capacity())
         return fArray->fCont[fCursor++];
   } else {
      for ( ; fCursor >= 0 && fArray->fCont[fCursor] == 0;
              fCursor--) { }

      if (fCursor >= 0)
         return fArray->fCont[fCursor--];
   }
   return 0;
}

//______________________________________________________________________________
void TObjArrayIter::Reset()
{
   // Reset array iterator.

   if (fDirection == kIterForward)
      fCursor = 0;
   else
      fCursor = fArray->Capacity() - 1;
}

