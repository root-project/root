// @(#)root/cont:$Name:  $:$Id: TRefArray.cxx,v 1.6 2002/02/02 11:57:10 brun Exp $
// Author: Rene Brun  02/10/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRefArray                                                            //
//                                                                      //
// An array of references to TObjects. The array expands automatically  //
// when  objects are added (shrinking can be done by hand using Expand()//
//                                                                      //
// The TRefArray can be filled with:                                    //
//     array.Add(obj)                                                   //
//     array.AddAt(obj,i)                                               //
//     but not array[i] = obj  !!!                                      //
//                                                                      //
// The array elements can be retrieved with:                            //
//     TObject *obj = array.At(i);                                      //
//                                                                      //
// When a TRefArray is Streamed, only the pointer unique id is written, //
// not the referenced object. TRefArray may be assigned to different    //
// branches of one Tree or several Trees.                               //
// The branch containing the TRefArray can be read before or after the  //
// array (eg TClonesArray, STL vector,..) of the referenced objects.    //
//                                                                      //
// See an example in $ROOTSYS/test/Event.h                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRefArray.h"
#include "TError.h"
#include "TFile.h"

ClassImp(TRefArray)

//______________________________________________________________________________
TRefArray::TRefArray(Int_t s, Int_t lowerBound)
{
   // Create an object array. Using s one can set the array size (default is
   // kInitCapacity=16) and lowerBound can be used to set the array lowerbound
   // index (default is 0).

   if (s < 0) {
      Warning("TRefArray", "size (%d) < 0", s);
      s = TCollection::kInitCapacity;
   } else if (s == 0)
      s = TCollection::kInitCapacity;
   fPID  = TProcessID::GetSessionProcessID();
   fUIDs = 0;
   Init(s, lowerBound);
}

//______________________________________________________________________________
TRefArray::TRefArray(const TRefArray &a)
{
   // Create a copy of TRefArray a. Note, does not copy the kIsOwner flag.

   fPID  = TProcessID::GetSessionProcessID();
   fUIDs = 0;
   Init(a.fSize, a.fLowerBound);

   for (Int_t i = 0; i < fSize; i++)
      fUIDs[i] = a.fUIDs[i];

   fLast = a.fLast;
   fName = a.fName;
}

//______________________________________________________________________________
TRefArray::~TRefArray()
{
   // Delete an array. Objects are not deleted unless the TRefArray is the
   // owner (set via SetOwner()).

   if (fUIDs) delete [] fUIDs;
   fPID  = 0;
   fUIDs = 0;
   fSize = 0;
}

//______________________________________________________________________________
void TRefArray::AddFirst(TObject *obj)
{
   // Add object in the first slot of the array. This will overwrite the
   // first element that might have been there. To have insertion semantics
   // use either a TList or a TOrdCollection.

   if (!obj) return;
   fUIDs[0] = TProcessID::AssignID(obj);
   Changed();
}

//______________________________________________________________________________
void TRefArray::AddLast(TObject *obj)
{
   // Add object in the next empty slot in the array. Expand the array
   // if necessary.

   AddAtAndExpand(obj, GetAbsLast()+1+fLowerBound);
}

//______________________________________________________________________________
void TRefArray::AddBefore(TObject *before, TObject *obj)
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
void TRefArray::AddAfter(TObject *after, TObject *obj)
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
void TRefArray::AddAtAndExpand(TObject *obj, Int_t idx)
{
   // Add object at position idx. If idx is larger than the current size
   // of the array, expand the array (double its size).

   if (!obj) return;
   if (idx < fLowerBound) {
      Error("AddAt", "out of bounds at %d in %x", idx, this);
      return;
   }
   if (idx-fLowerBound >= fSize)
      Expand(TMath::Max(idx-fLowerBound+1, GrowBy(fSize)));

   fUIDs[idx-fLowerBound] = TProcessID::AssignID(obj);
   fLast = TMath::Max(idx-fLowerBound, GetAbsLast());
   Changed();
}

//______________________________________________________________________________
void TRefArray::AddAt(TObject *obj, Int_t idx)
{
   // Add object at position ids. Give an error when idx is out of bounds
   // (i.e. the array is not expanded).

   if (!obj) return;
   if (!BoundsOk("AddAt", idx)) return;

   fUIDs[idx-fLowerBound] = TProcessID::AssignID(obj);
   fLast = TMath::Max(idx-fLowerBound, GetAbsLast());
   Changed();
}

//______________________________________________________________________________
Int_t  TRefArray::AddAtFree(TObject *obj)
{
   // Return the position of the new object.
   // Find the first empty cell or AddLast if there is no empty cell

   if (!obj) return 0;
   if (Last()) {    // <---------- This is to take in account "empty" TRefArray's
       Int_t i;
       for (i = 0; i < fSize; i++)
          if (!fUIDs[i]) {         // Add object at position i
             fUIDs[i] = TProcessID::AssignID(obj);
             fLast = TMath::Max(i, GetAbsLast());
             Changed();
             return i+fLowerBound;
          }
   }
   AddLast(obj);
   return GetLast();
}

//______________________________________________________________________________
TObject *TRefArray::After(TObject *obj) const
{
   // Return the object after obj. Returns 0 if obj is last object.

   if (!obj || !fPID) return 0;

   Int_t idx = IndexOf(obj) - fLowerBound;
   if (idx == -1 || idx == fSize-1) return 0;

   return fPID->GetObjectWithID(fUIDs[idx+1]);
}

//______________________________________________________________________________
TObject *TRefArray::Before(TObject *obj) const
{
   // Return the object before obj. Returns 0 if obj is first object.

   if (!obj || !fPID) return 0;

   Int_t idx = IndexOf(obj) - fLowerBound;
   if (idx == -1 || idx == 0) return 0;

   return fPID->GetObjectWithID(fUIDs[idx+1]);
}

//______________________________________________________________________________
void TRefArray::Clear(Option_t *)
{
   // Remove all objects from the array. Does not delete the objects
   // unless the TRefArray is the owner (set via SetOwner()).

   fLast = - 1;

   for (Int_t j=0 ; j < fSize; j++) fUIDs[j] = 0;
   
   Changed();
}

//______________________________________________________________________________
void TRefArray::Compress()
{
   // Remove empty slots from array.

   Int_t j = 0;

   for (Int_t i = 0; i < fSize; i++) {
      if (fUIDs[i]) {
         fUIDs[j] = fUIDs[i];
         j++;
      }
   }

   fLast = j - 1;

   for ( ; j < fSize; j++) fUIDs[j] = 0;
}

//______________________________________________________________________________
void TRefArray::Delete(Option_t *)
{
   // Remove all objects from the array AND delete all heap based objects.

   fLast = - 1;

   fSize = 0;
   if (fUIDs) {delete [] fUIDs; fUIDs = 0;}
      
   Changed();
}

//______________________________________________________________________________
void TRefArray::Expand(Int_t newSize)
{
   // Expand or shrink the array to newSize elements.

   if (newSize < 0) {
      Error ("Expand", "newSize must be positive (%d)", newSize);
      return;
   }
   if (newSize == fSize) return;
   UInt_t *temp = fUIDs;
   if (newSize != 0) {
      fUIDs = new UInt_t[newSize];
      if (newSize < fSize) memcpy(fUIDs,temp, newSize*sizeof(UInt_t));
      else {
         memcpy(fUIDs,temp,fSize*sizeof(UInt_t));
         memset(&fUIDs[fSize],0,(newSize-fSize)*sizeof(UInt_t));
      }
   } else {
     fUIDs = 0;
   }
   if (fSize) delete [] temp;
   fSize = newSize;
}

//_______________________________________________________________________
void TRefArray::Streamer(TBuffer &R__b)
{
   // Stream all objects in the array to or from the I/O buffer.

   UInt_t R__s, R__c;
   Int_t nobjects;
   UShort_t pidf;
   TFile *file = (TFile*)R__b.GetParent();
   if (R__b.IsReading()) {
      R__b.ReadVersion(&R__s, &R__c);
      TObject::Streamer(R__b);
      fName.Streamer(R__b);
      R__b >> nobjects;
      R__b >> fLowerBound;
      if (nobjects >= fSize) Expand(nobjects);
      fLast = -1;
      R__b >> pidf;
      fPID = TProcessID::ReadProcessID(pidf,file);
      for (Int_t i = 0; i < nobjects; i++) {
          R__b >> fUIDs[i];
          if (fUIDs[i] != 0) fLast = i;
      }
      Changed();
      R__b.CheckByteCount(R__s, R__c,TRefArray::IsA());
   } else {
      R__c = R__b.WriteVersion(TRefArray::IsA(), kTRUE);
      TObject::Streamer(R__b);
      fName.Streamer(R__b);
      nobjects = GetLast()+1;
      R__b << nobjects;
      R__b << fLowerBound;
      pidf = TProcessID::WriteProcessID(fPID,file);
      R__b << pidf;
      for (Int_t i = 0; i < nobjects; i++) {
          R__b << fUIDs[i];
      }
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
TObject *TRefArray::First() const
{
   // Return the object in the first slot.

   return fPID->GetObjectWithID(fUIDs[0]);
}

//______________________________________________________________________________
TObject *TRefArray::Last() const
{
   // Return the object in the last filled slot. Returns 0 if no entries.

   if (fLast == -1)
      return 0;
   else
      return fPID->GetObjectWithID(fUIDs[GetAbsLast()]);
}

//______________________________________________________________________________
Int_t TRefArray::GetEntries() const
{
   // Return the number of objects in array (i.e. number of non-empty slots).
   // Attention: use this method ONLY if you want to know the number of
   // non-empty slots. This function loops over the complete array and
   // is therefore very slow when applied in a loop. Most of the time you
   // better use GetLast()+1.

   Int_t cnt = 0;

   for (Int_t i = 0; i < fSize; i++)
      if (fUIDs[i]) cnt++;

   return cnt;
}

//______________________________________________________________________________
Int_t TRefArray::GetAbsLast() const
{
   // Return absolute index to last object in array. Returns -1 in case
   // array is empty.

   // For efficiency we need sometimes to update fLast so we have
   // to cast const away. Ugly, but making GetAbsLast() not const breaks
   // many other const functions.
   if (fLast == -2) {
      for (Int_t i = fSize-1; i >= 0; i--)
         if (fUIDs[i]) {
            ((TRefArray*)this)->fLast = i;
            return fLast;
         }
      ((TRefArray*)this)->fLast = -1;
   }
   return fLast;
}

//______________________________________________________________________________
Int_t TRefArray::GetLast() const
{
   // Return index of last object in array. Returns lowerBound-1 in case
   // array is empty.

   return fLowerBound+GetAbsLast();
}

//______________________________________________________________________________
TObject **TRefArray::GetObjectRef(TObject *obj) const
{
   // Return address of pointer obj.

   //Int_t index = IndexOf(obj);
   //return &fCont[index];
   return 0;
}

//______________________________________________________________________________
Int_t TRefArray::IndexOf(const TObject *obj) const
{
   // obj != 0 Return index of object in array.
   //          Returns lowerBound-1 in case array doesn't contain the obj.
   //
   // obj == 0 Return the index of the first empty slot.
   //          Returns lowerBound-1 in case array doesn't contain any empty slot.

/*
   Int_t i;
   if (obj) {
     for (i = 0; i < fSize; i++)
        if (fUIDs[i] && fUIDs[i]->IsEqual(obj))
           return i+fLowerBound;
   } else {    // Look for the first empty slot
     for (i = 0; i < fSize; i++)
        if (!fUIDs[i])
           return i+fLowerBound;
   }
*/
   return fLowerBound-1;
}

//______________________________________________________________________________
void TRefArray::Init(Int_t s, Int_t lowerBound)
{
   // Initialize a TRefArray.

   if (fUIDs && fSize != s) {
      delete [] fUIDs;
      fUIDs = 0;
   }

   fSize = s;

   fUIDs = new UInt_t[fSize];
   for (Int_t i=0;i<s;i++) fUIDs[i] = 0;
   fLowerBound = lowerBound;
   fLast = -1;
   Changed();
}

//______________________________________________________________________________
TIterator *TRefArray::MakeIterator(Bool_t dir) const
{
   // Returns an array iterator.

   return new TRefArrayIter(this, dir);
}

//______________________________________________________________________________
Bool_t TRefArray::OutOfBoundsError(const char *where, Int_t i) const
{
   // Generate an out-of-bounds error. Always returns false.

   Error(where, "index %d out of bounds (size: %d, this: 0x%08x)", i, fSize, this);
   return kFALSE;
}

//______________________________________________________________________________
TObject *TRefArray::RemoveAt(Int_t idx)
{
   // Remove object at index idx.

   if (!BoundsOk("RemoveAt", idx)) return 0;

   int i = idx-fLowerBound;

   TObject *obj = 0;
   if (fUIDs[i]) {
      obj = fPID->GetObjectWithID(fUIDs[i]);
      fUIDs[i] = 0;
      // recalculate array size
      if (i == fLast)
         do { fLast--; } while (fLast >= 0 && fUIDs[fLast] == 0);
      Changed();
   }

   return obj;
}

//______________________________________________________________________________
TObject *TRefArray::Remove(TObject *obj)
{
   // Remove object from array.

   if (!obj) return 0;

   Int_t idx = IndexOf(obj) - fLowerBound;

   if (idx == -1) return 0;

   TObject *ob = fPID->GetObjectWithID(fUIDs[idx]);
   fUIDs[idx] = 0;
   // recalculate array size
   if (idx == fLast)
      do { fLast--; } while (fLast >= 0 && fUIDs[fLast] == 0);
   Changed();
   return ob;
}

//______________________________________________________________________________
void TRefArray::SetLast(Int_t last)
{
   // Set index of last object in array, effectively truncating the
   // array. Use carefully since whenever last position has to be
   // recalculated, e.g. after a Remove() or Sort() it will be reset
   // to the last non-empty slot. If last is -2 this will force the
   // recalculation of the last used slot.

   if (last == -2)
      fLast = -2;
   else if (BoundsOk("SetLast", last))
      fLast = last - fLowerBound;
}

//______________________________________________________________________________
void TRefArray::Sort(Int_t upto)
{
   // If objects in array are sortable (i.e. IsSortable() returns true
   // for all objects) then sort array.

/*
   if (GetAbsLast() == -1 || fSorted) return;
   for (Int_t i = 0; i < fSize; i++)
      if (fUIDs[i]) {
         if (!fUIDs[i]->IsSortable()) {
            Error("Sort", "objects in array are not sortable");
            return;
         }
      }

   QSort(fUIDs, 0, TMath::Min(fSize, upto-fLowerBound));

   fLast   = -2;
   fSorted = kTRUE;
*/
}

//______________________________________________________________________________
Int_t TRefArray::BinarySearch(TObject *op, Int_t upto)
{
   // Find object using a binary search. Array must first have been sorted.
   // Search can be limited by setting upto to desired index.

/*
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
      //position = (base+last) / 2;
      //op2 = fCont[position];
      //if (op2 && (result = op->Compare(op2)) == 0)
      //   return position + fLowerBound;
      //if (!op2 || result < 0)
      //   last = position-1;
      //else
      //   base = position+1;
   }
*/
   return -1;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRefArrayIter                                                        //
//                                                                      //
// Iterator of object array.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TRefArrayIter)

//______________________________________________________________________________
TRefArrayIter::TRefArrayIter(const TRefArray *arr, Bool_t dir)
{
   // Create array iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.

   fArray     = arr;
   fDirection = dir;
   Reset();
}

//______________________________________________________________________________
TRefArrayIter::TRefArrayIter(const TRefArrayIter &iter)
{
   // Copy ctor.

   fArray     = iter.fArray;
   fDirection = iter.fDirection;
   fCursor    = iter.fCursor;
}

//______________________________________________________________________________
TIterator &TRefArrayIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TRefArrayIter::Class()) {
      const TRefArrayIter &rhs1 = (const TRefArrayIter &)rhs;
      fArray     = rhs1.fArray;
      fDirection = rhs1.fDirection;
      fCursor    = rhs1.fCursor;
   }
   return *this;
}

//______________________________________________________________________________
TRefArrayIter &TRefArrayIter::operator=(const TRefArrayIter &rhs)
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
TObject *TRefArrayIter::Next()
{
   // Return next object in array. Returns 0 when no more objects in array.

/*
   if (fDirection == kIterForward) {
      for ( ; fCursor < fArray->Capacity() && fArray->fCont[fCursor] == 0;
              fCursor++) { }

      if (fCursor < fArray->Capacity())
         return fArray->fUIDs[fCursor++];
   } else {
      for ( ; fCursor >= 0 && fArray->fUIDs[fCursor] == 0;
              fCursor--) { }

      if (fCursor >= 0)
         return fArray->fUIDs[fCursor--];
   }
*/
   return 0;
}

//______________________________________________________________________________
void TRefArrayIter::Reset()
{
   // Reset array iterator.

   if (fDirection == kIterForward)
      fCursor = 0;
   else
      fCursor = fArray->Capacity() - 1;
}

