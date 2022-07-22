// @(#)root/cont:$Id$
// Author: Fons Rademakers   26/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TExMap

This class stores a (key,value) pair using an external hash.
The (key,value) are Long64_t's and therefore can contain object
pointers or any longs. The map uses an open addressing hashing
method (linear probing).
*/

#include "TExMap.h"
#include "TBuffer.h"
#include "TError.h"
#include "TMathBase.h"
#include <string.h>


ClassImp(TExMap);

////////////////////////////////////////////////////////////////////////////////
/// Create a TExMap.

TExMap::TExMap(Int_t mapSize)
{
   // needed for automatic resizing to guarantee that one slot is always empty
   if (mapSize < 4) mapSize = 5;

   switch (mapSize) {
      // Avoid calling NextPrime for the common case:
      case   5: fSize = 5; break;
      case 503: fSize = 503; break;
      default:
         fSize  = (Int_t)TMath::NextPrime(mapSize);
   }
   fTable = new Assoc_t [fSize];

   memset(fTable,0,sizeof(Assoc_t)*fSize);
   fTally = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TExMap::TExMap(const TExMap &map) : TObject(map)
{
   fSize  = map.fSize;
   fTally = map.fTally;
   fTable = new Assoc_t [fSize];
   memcpy(fTable, map.fTable, fSize*sizeof(Assoc_t));
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TExMap& TExMap::operator=(const TExMap &map)
{
   if (this != &map) {
      TObject::operator=(map);
      fSize  = map.fSize;
      fTally = map.fTally;
      delete [] fTable;
      fTable = new Assoc_t [fSize];
      memcpy(fTable, map.fTable, fSize*sizeof(Assoc_t));
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete TExMap.

TExMap::~TExMap()
{
   delete [] fTable; fTable = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an (key,value) pair to the table. The key should be unique.

void TExMap::Add(ULong64_t hash, Long64_t key, Long64_t value)
{
   if (!fTable) return;

   Int_t slot = FindElement(hash, key);
   if (!fTable[slot].InUse()) {
      fTable[slot].SetHash(hash);
      fTable[slot].fKey = key;
      fTable[slot].fValue = value;
      fTally++;
      if (HighWaterMark())
         Expand(2 * fSize);
   } else
      Error("Add", "key %lld is not unique", key);
}

////////////////////////////////////////////////////////////////////////////////
/// Add an (key,value) pair to the table. The key should be unique.
/// If the 'slot' is open, use it to store the value,
/// otherwise revert to Add(hash,key,value)
/// This is usually used in conjunction with GetValue with 3 parameters:
/// ~~~ {.cpp}
/// if ((idx = (ULong64_t)fMap->GetValue(hash, key, slot)) != 0) {
///    ...
/// } else {
///    fMap->AddAt(slot,hash,key,value);
/// }
/// ~~~

void TExMap::AddAt(UInt_t slot, ULong64_t hash, Long64_t key, Long64_t value)
{
   if (!fTable) return;

   if (!fTable[slot].InUse()) {
      fTable[slot].SetHash(hash);
      fTable[slot].fKey = key;
      fTable[slot].fValue = value;
      fTally++;
      if (HighWaterMark())
         Expand(2 * fSize);
   } else {
      Add(hash,key,value);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return a reference to the value belonging to the key with the
/// specified hash value. If the key does not exist it will be added.
/// NOTE: the reference will be invalidated an Expand() triggered by
/// an Add() or another operator() call.

Long64_t &TExMap::operator()(ULong64_t hash, Long64_t key)
{
   static Long64_t err;
   if (!fTable) {
      Error("operator()", "fTable==0, should never happen");
      return err;
   }

   Int_t slot = FindElement(hash, key);
   if (!fTable[slot].InUse()) {
      fTable[slot].SetHash(hash);
      fTable[slot].fKey = key;
      fTable[slot].fValue = 0;
      fTally++;
      if (HighWaterMark()) {
         Expand(2 * fSize);
         slot = FindElement(hash, key);
      }
   }
   return fTable[slot].fValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all entries stored in the TExMap.

void TExMap::Delete(Option_t *)
{
   memset(fTable,0,sizeof(Assoc_t)*fSize);
   fTally = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the value belonging to specified key and hash value. If key not
/// found return 0.

Long64_t TExMap::GetValue(ULong64_t hash, Long64_t key)
{
   if (!fTable) return 0;

   hash |= 0x1;
   Int_t slot = Int_t(hash % fSize);
   Int_t firstSlot = slot;
   do {
      if (!fTable[slot].InUse()) return 0;
      if (key == fTable[slot].fKey) return fTable[slot].fValue;
      if (++slot == fSize) slot = 0;
   } while (firstSlot != slot);

   Error("GetValue", "table full");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the value belonging to specified key and hash value. If key not
/// found return 0.
/// In 'slot', return the index of the slot used or the first empty slot.
/// (to be used with AddAt).

Long64_t TExMap::GetValue(ULong64_t hash, Long64_t key, UInt_t &slot)
{
   if (!fTable) { slot = 0; return 0; }

   hash |= 0x1;
   slot = Int_t(hash % fSize);
   UInt_t firstSlot = slot;
   do {
      if (!fTable[slot].InUse()) return 0;
      if (key == fTable[slot].fKey) return fTable[slot].fValue;
      if (++slot == (UInt_t)fSize) slot = 0;
   } while (firstSlot != slot);

   Error("GetValue", "table full");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove entry with specified key from the TExMap.

void TExMap::Remove(ULong64_t hash, Long64_t key)
{
   if (!fTable)
      return;

   Int_t i = FindElement(hash, key);
   if (!fTable[i].InUse()) {
      Error("Remove", "key %lld not found at %d", key, i);
      return;
   }

   fTable[i].Clear();
   FixCollisions(i);
   fTally--;
}

////////////////////////////////////////////////////////////////////////////////
/// Find an entry with specified hash and key in the TExMap.
/// Returns the slot of the key or the next empty slot.

Int_t TExMap::FindElement(ULong64_t hash, Long64_t key)
{
   if (!fTable) return 0;

   hash |= 0x1;
   Int_t slot = Int_t(hash % fSize);
   Int_t firstSlot = slot;
   do {
      if (!fTable[slot].InUse()) return slot;
      if (key == fTable[slot].fKey) return slot;
      if (++slot == fSize) slot = 0;
   } while (firstSlot != slot);

   Error("FindElement", "table full");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Rehash the map in case an entry has been removed.

void TExMap::FixCollisions(Int_t index)
{
   Int_t oldIndex, nextIndex;
   Assoc_t nextObject;

   for (oldIndex = index+1; ;oldIndex++) {
      if (oldIndex >= fSize)
         oldIndex = 0;
      nextObject = fTable[oldIndex];
      if (!nextObject.InUse())
         break;
      nextIndex = FindElement(nextObject.GetHash(), nextObject.fKey);
      if (nextIndex != oldIndex) {
         fTable[nextIndex] = nextObject;
         fTable[oldIndex].Clear();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Expand the TExMap.

void TExMap::Expand(Int_t newSize)
{
   Int_t i;
   Assoc_t *oldTable = fTable;
   Int_t oldsize = fSize;
   newSize = (Int_t)TMath::NextPrime(newSize);
   fTable  = new Assoc_t [newSize];

   for (i = newSize; --i >= 0;) {
      fTable[i].Clear();
   }

   fSize = newSize;
   for (i = 0; i < oldsize; i++)
      if (oldTable[i].InUse()) {
         Int_t slot = FindElement(oldTable[i].GetHash(), oldTable[i].fKey);
         if (!fTable[slot].InUse())
            fTable[slot] = oldTable[i];
         else
            Error("Expand", "slot %d not empty (should never happen)", slot);
      }
   delete [] oldTable;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream all objects in the collection to or from the I/O buffer.

void TExMap::Streamer(TBuffer &b)
{
   Int_t i;
   UInt_t R__s, R__c;

   if (b.IsReading()) {
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      TObject::Streamer(b);

      if (R__v >= 3) {
         // new custom streamer with slots indices stored (Long64_t version).
         Int_t size, tally;
         b >> size;
         Expand(size);
         b >> tally;
         Int_t slot;
         ULong64_t hash;
         Long64_t key, value;
         for (i = 0; i < tally; ++i) {
            b >> slot;
            b >> hash;
            b >> key;
            b >> value;
            Assoc_t *assoc = fTable + slot;
            assoc->SetHash(hash);
            assoc->fKey = key;
            assoc->fValue = value;
         }
         fTally = tally;
      } else if (R__v >= 2) {
         // new custom streamer with slots indices stored.
         Int_t size, tally;
         b >> size;
         Expand(size);
         b >> tally;
         Int_t slot;
         ULong_t hash;
         Long_t key, value;
         for (i = 0; i < tally; ++i) {
            b >> slot;
            b >> hash;
            b >> key;
            b >> value;
            Assoc_t* assoc = fTable + slot;
            assoc->SetHash(hash);
            assoc->fKey = key;
            assoc->fValue = value;
         }
         fTally = tally;
      } else {
         // old custom streamer that only allows slow dynamic rebuild of TExMap:
         Int_t n;
         b >> n;
         ULong_t hash;
         Long_t key, value;
         for (i = 0; i < n; i++) {
            b >> hash;
            b >> key;
            b >> value;
            Add(hash, key, value);
         }
      }
      b.CheckByteCount(R__s, R__c, TExMap::IsA());
   } else {
      R__c = b.WriteVersion(TExMap::IsA(), kTRUE);
      // new custom streamer stores slots indices
      TObject::Streamer(b);
      b << fSize;
      b << fTally;

      for (i=0; i < fSize; i++) {
         if (!fTable[i].InUse()) continue;
         b << i;
         b << fTable[i].GetHash();
         b << fTable[i].fKey;
         b << fTable[i].fValue;
      }
      b.SetByteCount(R__c, kTRUE);
   }
}


ClassImp(TExMapIter);

////////////////////////////////////////////////////////////////////////////////
/// Create TExMap iterator.

TExMapIter::TExMapIter(const TExMap *map) : fMap(map), fCursor(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Overloaded assignment operator.

TExMapIter &TExMapIter::operator=(const TExMapIter &rhs)
{
   if (this != &rhs) {
      fMap    = rhs.fMap;
      fCursor = rhs.fCursor;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Get next entry from TExMap. Returns kFALSE at end of map.

Bool_t TExMapIter::Next(ULong64_t &hash, Long64_t &key, Long64_t &value)
{
   while (fCursor < fMap->fSize && !fMap->fTable[fCursor].InUse())
      fCursor++;

   if (fCursor == fMap->fSize)
      return kFALSE;

   hash  = fMap->fTable[fCursor].GetHash();
   key   = fMap->fTable[fCursor].fKey;
   value = fMap->fTable[fCursor].fValue;
   fCursor++;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get next entry from TExMap. Returns kFALSE at end of map.

Bool_t TExMapIter::Next(Long64_t &key, Long64_t &value)
{
   ULong64_t hash;
   return Next(hash, key, value);
}
