// @(#)root/cont:$Name:  $:$Id: TExMap.cxx,v 1.3 2001/05/21 12:44:00 rdm Exp $
// Author: Fons Rademakers   26/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TExMap                                                               //
//                                                                      //
// This class stores a (key,value) pair using an external hash.         //
// The (key,value) are Long_t's and therefore can contain object        //
// pointers or any longs. The map uses an open addressing hashing       //
// method (linear probing).                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TExMap.h"
#include "TMath.h"


ClassImp(TExMap)

//______________________________________________________________________________
TExMap::TExMap(Int_t mapSize)
{
   // Create a TExMap.

   fSize  = (Int_t)TMath::NextPrime(mapSize);
   fTable = new Assoc_t* [fSize];
   memset(fTable, 0, fSize*sizeof(Assoc_t*));
   fTally = 0;
}

//______________________________________________________________________________
TExMap::TExMap(const TExMap &map) : TObject(map)
{
   // Copy constructor.

   fSize  = map.fSize;
   fTally = map.fTally;
   fTable = new Assoc_t* [fSize];
   memset(fTable, 0, fSize*sizeof(Assoc_t*));
   for (Int_t i = 0; i < fSize; i++)
      if (map.fTable[i])
         fTable[i] = new Assoc_t(map.fTable[i]->fHash, map.fTable[i]->fKey,
                                 map.fTable[i]->fValue);
}

//______________________________________________________________________________
TExMap::~TExMap()
{
   // Delete TExMap.

   for (Int_t i = 0; i < fSize; i++)
      delete fTable[i];
   delete [] fTable; fTable = 0;
}

//______________________________________________________________________________
void TExMap::Add(ULong_t hash, Long_t key, Long_t value)
{
   // Add an (key,value) pair to the table. The key should be unique.

   if (!fTable)
      return;

   Int_t slot = FindElement(hash, key);
   if (fTable[slot] == 0) {
      fTable[slot] = new Assoc_t(hash, key, value);
      fTally++;
      if (HighWaterMark())
         Expand(2 * fSize);
   } else
      Error("Add", "key %ld is not unique", key);
}

//______________________________________________________________________________
Long_t &TExMap::operator()(ULong_t hash, Long_t key)
{
   // Return a reference to the value belonging to the key with the
   // specified hash value. If the key does not exist it will be added.

   static Long_t err;
   if (!fTable) {
      Error("operator()", "fTable==0, should never happen");
      return err;
   }

   Int_t slot = FindElement(hash, key);
   if (fTable[slot] == 0) {
      fTable[slot] = new Assoc_t(hash, key, 0);
      fTally++;
      if (HighWaterMark()) {
         Expand(2 * fSize);
         slot = FindElement(hash, key);
      }
   }
   return fTable[slot]->fValue;
}

//______________________________________________________________________________
void TExMap::Delete(Option_t *)
{
   // Delete all entries stored in the TExMap.

   for (int i = 0; i < fSize; i++) {
      if (fTable[i]) {
         delete fTable[i];
         fTable[i] = 0;
      }
   }
   fTally = 0;
}

//______________________________________________________________________________
Long_t TExMap::GetValue(ULong_t hash, Long_t key)
{
   // Return the value belonging to specified key and hash value. If key not
   // found return 0.

   if (!fTable) return 0;

   Int_t slot = Int_t(hash % fSize);
   for (int n = 0; n < fSize; n++) {
      if (!fTable[slot]) return 0;
      if (key == fTable[slot]->fKey) return fTable[slot]->fValue;
      if (++slot == fSize) slot = 0;
   }

   if (fTable[slot])
      return fTable[slot]->fValue;
   return 0;
}

//______________________________________________________________________________
void TExMap::Remove(ULong_t hash, Long_t key)
{
   // Remove entry with specified key from the TExMap.

   if (!fTable)
      return;

   Int_t i = FindElement(hash, key);
   if (fTable[i] == 0) {
      Warning("Remove", "key %ld not found at %d", key, i);
      for (int j = 0; j < fSize; j++) {
         if (fTable[j] && fTable[j]->fKey == key) {
            Error("Remove", "%ld found at %d !!!", key, j);
            i = j;
         }
      }
   }

   if (fTable[i]) {
      delete fTable[i];
      fTable[i] = 0;
      FixCollisions(i);
      fTally--;
   }
}

//______________________________________________________________________________
Int_t TExMap::FindElement(ULong_t hash, Long_t key)
{
   // Find an entry with specified hash and key in the TExMap.
   // Returns the slot of the key or the next empty slot.

   if (!fTable) return 0;

   Int_t slot = Int_t(hash % fSize);
   for (int n = 0; n < fSize; n++) {
      if (!fTable[slot]) return slot;
      if (key == fTable[slot]->fKey) return slot;
      if (++slot == fSize) slot = 0;
   }
   return slot;
}

//______________________________________________________________________________
void TExMap::FixCollisions(Int_t index)
{
   // Rehash the map in case an entry has been removed.

   Int_t oldIndex, nextIndex;
   Assoc_t *nextObject;

   for (oldIndex = index+1; ;oldIndex++) {
      if (oldIndex >= fSize)
         oldIndex = 0;
      nextObject = fTable[oldIndex];
      if (nextObject == 0)
         break;
      nextIndex = FindElement(nextObject->fHash, nextObject->fKey);
      if (nextIndex != oldIndex) {
         fTable[nextIndex] = nextObject;
         fTable[oldIndex] = 0;
      }
   }
}

//______________________________________________________________________________
void TExMap::Expand(Int_t newSize)
{
   // Expand the TExMap.

   Assoc_t **oldTable = fTable, *op;
   Int_t oldsize = fSize;
   newSize = (Int_t)TMath::NextPrime(newSize);
   fTable  = new Assoc_t* [newSize];
   memset(fTable, 0, newSize*sizeof(Assoc_t*));
   fSize   = newSize;
   for (int i = 0; i < oldsize; i++)
      if ((op = oldTable[i])) {
         Int_t slot = FindElement(op->fHash, op->fKey);
         if (fTable[slot] == 0)
            fTable[slot] = op;
         else
            Error("Expand", "slot %d not empty (should never happen)", slot);
      }
   delete [] oldTable;
}


ClassImp(TExMapIter)

//______________________________________________________________________________
TExMapIter::TExMapIter(const TExMap *map) : fMap(map), fCursor(0)
{
   // Create TExMap iterator.
}

//______________________________________________________________________________
Bool_t TExMapIter::Next(ULong_t &hash, Long_t &key, Long_t &value)
{
   // Get next entry from TExMap. Returns kFALSE at end of map.

   while (fCursor < fMap->fSize && !fMap->fTable[fCursor])
      fCursor++;

   if (fCursor == fMap->fSize)
      return kFALSE;

   hash  = fMap->fTable[fCursor]->fHash;
   key   = fMap->fTable[fCursor]->fKey;
   value = fMap->fTable[fCursor]->fValue;
   fCursor++;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TExMapIter::Next(Long_t &key, Long_t &value)
{
   // Get next entry from TExMap. Returns kFALSE at end of map.

   ULong_t hash;
   return Next(hash, key, value);
}

