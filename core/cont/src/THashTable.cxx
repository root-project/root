// @(#)root/cont:$Id$
// Author: Fons Rademakers   27/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THashTable                                                           //
//                                                                      //
// THashTable implements a hash table to store TObject's. The hash      //
// value is calculated using the value returned by the TObject's        //
// Hash() function. Each class inheriting from TObject can override     //
// Hash() as it sees fit.                                               //
// THashTable does not preserve the insertion order of the objects.     //
// If the insertion order is important AND fast retrieval is needed     //
// use THashList instead.                                               //
//Begin_Html
/*
<img src=gif/thashtable.gif>
*/
//End_Html
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THashTable.h"
#include "TObjectTable.h"
#include "TList.h"
#include "TError.h"


ClassImp(THashTable)

//______________________________________________________________________________
THashTable::THashTable(Int_t capacity, Int_t rehashlevel)
{
   // Create a THashTable object. Capacity is the initial hashtable capacity
   // (i.e. number of slots), by default kInitHashTableCapacity = 17, and
   // rehashlevel is the value at which a rehash will be triggered. I.e. when
   // the average size of the linked lists at a slot becomes longer than
   // rehashlevel then the hashtable will be resized and refilled to reduce
   // the collision rate to about 1. The higher the collision rate, i.e. the
   // longer the linked lists, the longer lookup will take. If rehashlevel=0
   // the table will NOT automatically be rehashed. Use Rehash() for manual
   // rehashing.

   if (capacity < 0) {
      Warning("THashTable", "capacity (%d) < 0", capacity);
      capacity = TCollection::kInitHashTableCapacity;
   } else if (capacity == 0)
      capacity = TCollection::kInitHashTableCapacity;

   fSize = (Int_t)TMath::NextPrime(TMath::Max(capacity,(int)TCollection::kInitHashTableCapacity));
   fCont = new TList* [fSize];
   memset(fCont, 0, fSize*sizeof(TList*));

   fEntries   = 0;
   fUsedSlots = 0;
   if (rehashlevel < 2) rehashlevel = 0;
   fRehashLevel = rehashlevel;
}

//______________________________________________________________________________
THashTable::~THashTable()
{
   // Delete a hashtable. Objects are not deleted unless the THashTable is the
   // owner (set via SetOwner()).

   if (fCont) Clear();
   delete [] fCont;
   fCont = 0;
   fSize = 0;
}

//______________________________________________________________________________
void THashTable::Add(TObject *obj)
{
   // Add object to the hash table. Its position in the table will be
   // determined by the value returned by its Hash() function.

   if (IsArgNull("Add", obj)) return;

   Int_t slot = GetHashValue(obj);
   if (!fCont[slot]) {
      fCont[slot] = new TList;
      fUsedSlots++;
   }
   fCont[slot]->Add(obj);
   fEntries++;

   if (fRehashLevel && AverageCollisions() > fRehashLevel)
      Rehash(fEntries);
}

//______________________________________________________________________________
void THashTable::AddAll(const TCollection *col)
{
   // Add all objects from collection col to this collection.
   // Implemented for more efficient rehashing.

   // Hashing after AddAll can be much more expensive than
   // hashing before, as we need to add more elements.
   // We assume an ideal hash, i.e. fUsedSlots==fSize.
   Int_t sumEntries=fEntries+col->GetEntries();
   Bool_t rehashBefore=fRehashLevel && (sumEntries > fSize*fRehashLevel);
   if (rehashBefore)
      Rehash(sumEntries);

   // prevent Add from Rehashing
   Int_t saveRehashLevel=fRehashLevel;
   fRehashLevel=0;

   TCollection::AddAll(col);

   fRehashLevel=saveRehashLevel;
   // If we didn't Rehash before, we might have to do it
   // now, due to a non-perfect hash function.
   if (!rehashBefore && fRehashLevel && AverageCollisions() > fRehashLevel)
      Rehash(fEntries);
}

//______________________________________________________________________________
void THashTable::Clear(Option_t *option)
{
   // Remove all objects from the table. Does not delete the objects
   // unless the THashTable is the owner (set via SetOwner()).

   for (int i = 0; i < fSize; i++) {
      // option "nodelete" is passed when Clear is called from
      // THashList::Clear() or THashList::Delete() or Rehash().
      if (fCont[i]) {
         if (IsOwner())
            fCont[i]->SetOwner();
         fCont[i]->Clear(option);
      }
      SafeDelete(fCont[i]);
   }

   fEntries   = 0;
   fUsedSlots = 0;
}

//______________________________________________________________________________
Int_t THashTable::Collisions(const char *name) const
{
   // Returns the number of collisions for an object with a certain name
   // (i.e. number of objects in same slot in the hash table, i.e. length
   // of linked list).

   Int_t slot = GetHashValue(name);
   if (fCont[slot]) return fCont[slot]->GetSize();
   return 0;
}

//______________________________________________________________________________
Int_t THashTable::Collisions(TObject *obj) const
{
   // Returns the number of collisions for an object (i.e. number of objects
   // in same slot in the hash table, i.e. length of linked list).

   if (IsArgNull("Collisions", obj)) return 0;

   Int_t slot = GetHashValue(obj);
   if (fCont[slot]) return fCont[slot]->GetSize();
   return 0;
}

//______________________________________________________________________________
void THashTable::Delete(Option_t *)
{
   // Remove all objects from the table AND delete all heap based objects.

   for (int i = 0; i < fSize; i++)
      if (fCont[i]) {
         fCont[i]->Delete();
         SafeDelete(fCont[i]);
      }

   fEntries   = 0;
   fUsedSlots = 0;
}

//______________________________________________________________________________
TObject *THashTable::FindObject(const char *name) const
{
   // Find object using its name. Uses the hash value returned by the
   // TString::Hash() after converting name to a TString.

   Int_t slot = GetHashValue(name);
   if (fCont[slot]) return fCont[slot]->FindObject(name);
   return 0;
}

//______________________________________________________________________________
TObject *THashTable::FindObject(const TObject *obj) const
{
   // Find object using its hash value (returned by its Hash() member).

   if (IsArgNull("FindObject", obj)) return 0;

   Int_t slot = GetHashValue(obj);
   if (fCont[slot]) return fCont[slot]->FindObject(obj);
   return 0;
}

//______________________________________________________________________________
TList *THashTable::GetListForObject(const char *name) const
{
   // Return the TList corresponding to object's name based hash value.
   // One can iterate this list "manually" to find, e.g. objects with
   // the same name.

   return fCont[GetHashValue(name)];
}

//______________________________________________________________________________
TList *THashTable::GetListForObject(const TObject *obj) const
{
   // Return the TList corresponding to object's hash value.
   // One can iterate this list "manually" to find, e.g. identical
   // objects.

   if (IsArgNull("GetListForObject", obj)) return 0;
   return fCont[GetHashValue(obj)];
}

//______________________________________________________________________________
TObject **THashTable::GetObjectRef(const TObject *obj) const
{
   // Return address of pointer to obj

   if (IsArgNull("GetObjectRef", obj)) return 0;

   Int_t slot = GetHashValue(obj);
   if (fCont[slot]) return fCont[slot]->GetObjectRef(obj);
   return 0;
}

//______________________________________________________________________________
TIterator *THashTable::MakeIterator(Bool_t dir) const
{
   // Returns a hash table iterator.

   return new THashTableIter(this, dir);
}

//______________________________________________________________________________
void THashTable::Rehash(Int_t newCapacity, Bool_t checkObjValidity)
{
   // Rehash the hashtable. If the collision rate becomes too high (i.e.
   // the average size of the linked lists become too long) then lookup
   // efficiency decreases since relatively long lists have to be searched
   // every time. To improve performance rehash the hashtable. This resizes
   // the table to newCapacity slots and refills the table. Use
   // AverageCollisions() to check if you need to rehash. Set checkObjValidity
   // to kFALSE if you know that all objects in the table are still valid
   // (i.e. have not been deleted from the system in the meanwhile).

   THashTable *ht = new THashTable(newCapacity);

   TIter next(this);
   TObject *obj;

   if (checkObjValidity && TObject::GetObjectStat() && gObjectTable) {
      while ((obj = next()))
         if (gObjectTable->PtrIsValid(obj)) ht->Add(obj);
   } else {
      while ((obj = next()))
         ht->Add(obj);
   }

   Clear("nodelete");
   delete [] fCont;
   fCont = ht->fCont;
   ht->fCont = 0;

   fSize      = ht->fSize;     // idem
   fEntries   = ht->fEntries;
   fUsedSlots = ht->fUsedSlots;

   // this should not happen, but it will prevent an endless loop
   // in case of a very bad hash function
   if (fRehashLevel && AverageCollisions() > fRehashLevel)
      fRehashLevel = (int)AverageCollisions() + 1;

   delete ht;
}

//______________________________________________________________________________
TObject *THashTable::Remove(TObject *obj)
{
   // Remove object from the hashtable.

   Int_t slot = GetHashValue(obj);
   if (fCont[slot]) {
      TObject *ob = fCont[slot]->Remove(obj);
      if (ob) {
         fEntries--;
         if (fCont[slot]->GetSize() == 0) {
            SafeDelete(fCont[slot]);
            fUsedSlots--;
         }
         return ob;
      }
   }
   return 0;
}

//______________________________________________________________________________
TObject *THashTable::RemoveSlow(TObject *obj)
{
   // Remove object from the hashtable without using the hash value.

   for (int i = 0; i < fSize; i++) {
      if (fCont[i]) {
         TObject *ob = fCont[i]->Remove(obj);
         if (ob) {
            fEntries--;
            if (fCont[i]->GetSize() == 0) {
               SafeDelete(fCont[i]);
               fUsedSlots--;
            }
            return ob;
         }
      }
   }
   return 0;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THashTableIter                                                       //
//                                                                      //
// Iterator of hash table.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(THashTableIter)

//______________________________________________________________________________
THashTableIter::THashTableIter(const THashTable *ht, Bool_t dir)
{
   // Create a hashtable iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.

   fTable      = ht;
   fDirection  = dir;
   fListCursor = 0;
   Reset();
}

//______________________________________________________________________________
THashTableIter::THashTableIter(const THashTableIter &iter) : TIterator(iter)
{
   // Copy ctor.

   fTable      = iter.fTable;
   fDirection  = iter.fDirection;
   fCursor     = iter.fCursor;
   fListCursor = 0;
   if (iter.fListCursor) {
      fListCursor = (TListIter *)iter.fListCursor->GetCollection()->MakeIterator();
      if (fListCursor)
         fListCursor->operator=(*iter.fListCursor);
   }
}

//______________________________________________________________________________
TIterator &THashTableIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == THashTableIter::Class()) {
      const THashTableIter &rhs1 = (const THashTableIter &)rhs;
      fTable     = rhs1.fTable;
      fDirection = rhs1.fDirection;
      fCursor    = rhs1.fCursor;
      if (rhs1.fListCursor) {
         fListCursor = (TListIter *)rhs1.fListCursor->GetCollection()->MakeIterator();
         if (fListCursor)
            fListCursor->operator=(*rhs1.fListCursor);
      }
   }
   return *this;
}

//______________________________________________________________________________
THashTableIter &THashTableIter::operator=(const THashTableIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fTable     = rhs.fTable;
      fDirection = rhs.fDirection;
      fCursor    = rhs.fCursor;
      if (rhs.fListCursor) {
         fListCursor = (TListIter *)rhs.fListCursor->GetCollection()->MakeIterator();
         if (fListCursor)
            fListCursor->operator=(*rhs.fListCursor);
      }
   }
   return *this;
}

//______________________________________________________________________________
THashTableIter::~THashTableIter()
{
   // Delete hashtable iterator.

   delete fListCursor;
}

//______________________________________________________________________________
TObject *THashTableIter::Next()
{
   // Return next object in hashtable. Returns 0 when no more objects in table.

   while (kTRUE) {
      if (!fListCursor) {
         int slot = NextSlot();
         if (slot == -1) return 0;
         fListCursor = new TListIter(fTable->fCont[slot], fDirection);
      }

      TObject *obj = fListCursor->Next();
      if (obj) return obj;

      SafeDelete(fListCursor);
   }
}

//______________________________________________________________________________
Int_t THashTableIter::NextSlot()
{
   // Returns index of next slot in table containing list to be iterated.

   if (fDirection == kIterForward) {
      for ( ; fCursor < fTable->Capacity() && fTable->fCont[fCursor] == 0;
              fCursor++) { }

      if (fCursor < fTable->Capacity())
         return fCursor++;

   } else {
      for ( ; fCursor >= 0 && fTable->fCont[fCursor] == 0;
              fCursor--) { }

      if (fCursor >= 0)
         return fCursor--;
   }
   return -1;
}

//______________________________________________________________________________
void THashTableIter::Reset()
{
   // Reset the hashtable iterator. Either to beginning or end, depending on
   // the initial iteration direction.

   if (fDirection == kIterForward)
      fCursor = 0;
   else
      fCursor = fTable->Capacity() - 1;
   SafeDelete(fListCursor);
}

//______________________________________________________________________________
bool THashTableIter::operator!=(const TIterator &aIter) const
{
   // This operator compares two TIterator objects.

   if (nullptr == (&aIter))
      return fListCursor;

   if (aIter.IsA() == THashTableIter::Class()) {
      const THashTableIter &iter(dynamic_cast<const THashTableIter &>(aIter));
      return (fListCursor != iter.fListCursor);
   }
   return false; // for base class we don't implement a comparison
}

//______________________________________________________________________________
bool THashTableIter::operator!=(const THashTableIter &aIter) const
{
   // This operator compares two THashTableIter objects.

   if (nullptr == (&aIter))
      return fListCursor;

   return (fListCursor != aIter.fListCursor);
}

//______________________________________________________________________________
TObject *THashTableIter::operator*() const
{
   // Return pointer to current object or nullptr.

   return (fListCursor ? fListCursor->operator*() : nullptr);
}
