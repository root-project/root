// @(#)root/cont:$Id$
// Author: Fons Rademakers   12/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMap                                                                 //
//                                                                      //
// TMap implements an associative array of (key,value) pairs using a    //
// THashTable for efficient retrieval (therefore TMap does not conserve //
// the order of the entries). The hash value is calculated              //
// using the value returned by the keys Hash() function and the         //
// key comparison is done via the IsEqual() function.                   //
// Both key and value must inherit from TObject.                        //
//Begin_Html
/*
<img src=gif/tmap.gif>
*/
//End_Html
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMap.h"
#include "THashTable.h"
#include "TROOT.h"
#include "TBrowser.h"
#include "TRegexp.h"

ClassImp(TMap)

//______________________________________________________________________________
TMap::TMap(Int_t capacity, Int_t rehashlevel)
{
   // TMap ctor. See THashTable for a description of the arguments.

   fSize  = 0;
   fTable = new THashTable(capacity, rehashlevel);
}

//______________________________________________________________________________
TMap::~TMap()
{
   // TMap dtor. Objects are not deleted unless the TMap is the
   // owner (set via SetOwner()).

   Clear();
   delete fTable;
}

//______________________________________________________________________________
void TMap::Add(TObject *)
{
   // This function may not be used (but we need to provide it since it is
   // a pure virtual in TCollection). Use Add(key,value) instead.

   MayNotUse("Add(TObject *obj)");
}

//______________________________________________________________________________
void TMap::Add(TObject *key, TObject *value)
{
   // Add a (key,value) pair to the map.

   if (IsArgNull("Add", key)) return;

   fTable->Add(new TPair(key, value));
   fSize++;
}

//______________________________________________________________________________
Float_t TMap::AverageCollisions() const
{
   // Return the ratio of entries vs occupied slots.

   return fTable->AverageCollisions();
}

//______________________________________________________________________________
Int_t TMap::Capacity() const
{
   // Return number of slots in the hashtable. Use GetSize() to get the
   // number of objects stored in the TMap.

   return fTable->Capacity();
}

//______________________________________________________________________________
void TMap::Clear(Option_t *option)
{
   // Remove all (key,value) pairs from the map. The keys/values are
   // deleted depending on the state of key-ownership (SetOwner()) and
   // value-ownership (SetOwnerValue()).
   //
   // To delete these objects regardless of the ownership state use:
   //  - Delete()       to delete only keys;
   //  - DeleteValues() to delete only values;
   //  - DeleteAll()    to delete both keys and values.

   if (IsOwner() && IsOwnerValue())
      DeleteAll();
   else if (IsOwner())
      Delete();
   else if (IsOwnerValue())
      DeleteValues();
   else {
      fTable->Delete(option);    // delete the TPair's
      fSize = 0;
   }
}

//______________________________________________________________________________
Int_t TMap::Collisions(const char *keyname) const
{
   // Returns the number of collisions for a key with a certain name
   // (i.e. number of objects in same slot in the hash table, i.e. length
   // of linked list).

   return fTable->Collisions(keyname);
}

//______________________________________________________________________________
Int_t TMap::Collisions(TObject *key) const
{
   // Returns the number of collisions for a key (i.e. number of objects
   // in same slot in the hash table, i.e. length of linked list).

   return fTable->Collisions(key);
}

//______________________________________________________________________________
void TMap::Delete(Option_t *option)
{
   // Remove all (key,value) pairs from the map AND delete the keys
   // when they are allocated on the heap.

   TIter next(fTable);
   TPair *a;

   while ((a = (TPair *)next()))
      if (a->Key() && a->Key()->IsOnHeap())
         TCollection::GarbageCollect(a->Key());

   fTable->Delete(option);   // delete the TPair's
   fSize = 0;
}

//______________________________________________________________________________
void TMap::DeleteValues()
{
   // Remove all (key,value) pairs from the map AND delete the values
   // when they are allocated on the heap.

   TIter next(fTable);
   TPair *a;

   while ((a = (TPair *)next()))
      if (a->Value() && a->Value()->IsOnHeap())
         TCollection::GarbageCollect(a->Value());

   fTable->Delete();   // delete the TPair's
   fSize = 0;
}

//______________________________________________________________________________
void TMap::DeleteAll()
{
   // Remove all (key,value) pairs from the map AND delete the keys AND
   // values when they are allocated on the heap.

   TIter next(fTable);
   TPair *a;

   while ((a = (TPair *)next())) {
      if (a->Key()   && a->Key()->IsOnHeap())
         TCollection::GarbageCollect(a->Key());
      if (a->Value() && a->Value()->IsOnHeap())
         TCollection::GarbageCollect(a->Value());
   }

   fTable->Delete();   // delete the TPair's
   fSize = 0;
}

//______________________________________________________________________________
Bool_t TMap::DeleteEntry(TObject *key)
{
   // Remove (key,value) pair with key from the map. Returns true
   // if the key was found and removed, false otherwise.
   // The key and value objects are deleted if map is the owner
   // of keys and values respectively.

   if (!key) return kFALSE;

   TPair *a;
   if ((a = (TPair *)fTable->FindObject(key))) {
      if (fTable->Remove(key)) {
         if (IsOwner() && a->Key() && a->Key()->IsOnHeap())
            TCollection::GarbageCollect(a->Key());
         if (IsOwnerValue() && a->Value() && a->Value()->IsOnHeap())
            TCollection::GarbageCollect(a->Value());
         delete a;
         fSize--;
         return kTRUE;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
TObject *TMap::FindObject(const char *keyname) const
{
   // Check if a (key,value) pair exists with keyname as name of the key.
   // Returns a TPair* (need to downcast from TObject). Use Key() and
   // Value() to get the pointers to the key and value, respectively.
   // Returns 0 if not found.

   return fTable->FindObject(keyname);
}

//______________________________________________________________________________
TObject *TMap::FindObject(const TObject *key) const
{
   // Check if a (key,value) pair exists with key as key.
   // Returns a TPair* (need to downcast from TObject). Use Key() and
   // Value() to get the pointers to the key and value, respectively.
   // Returns 0 if not found.

   if (IsArgNull("FindObject", key)) return 0;

   return fTable->FindObject(key);
}

//______________________________________________________________________________
TObject *TMap::GetValue(const char *keyname) const
{
   // Returns a pointer to the value associated with keyname as name of the key.

   TPair *a = (TPair *)fTable->FindObject(keyname);
   if (a) return a->Value();
   return 0;
}

//______________________________________________________________________________
TObject *TMap::GetValue(const TObject *key) const
{
   // Returns a pointer to the value associated with key.

   if (IsArgNull("GetValue", key)) return 0;

   TPair *a = (TPair *)fTable->FindObject(key);
   if (a) return a->Value();
   return 0;
}

//______________________________________________________________________________
TIterator *TMap::MakeIterator(Bool_t dir) const
{
   // Create an iterator for TMap.

   return new TMapIter(this, dir);
}

//______________________________________________________________________________
void TMap::PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const
{
   // Print the collection entry.

   TObject* val = GetValue(entry);

   TROOT::IndentLevel();
   printf("Key:   ");
   entry->Print();
   TROOT::IndentLevel();
   if (TStorage::IsOnHeap(val)) {
      printf("Value: ");
      TCollection* coll = dynamic_cast<TCollection*>(val);
      if (coll) {
         coll->Print(option, recurse);
      } else {
         val->Print(option);
      }
   } else {
      printf("Value: 0x%lx\n", (ULong_t) val);
   }
}

//______________________________________________________________________________
void TMap::Rehash(Int_t newCapacity, Bool_t checkObjValidity)
{
   // Rehash the underlaying THashTable (see THashTable::Rehash()).

   fTable->Rehash(newCapacity, checkObjValidity);
}

//______________________________________________________________________________
TObject *TMap::Remove(TObject *key)
{
   // Remove the (key,value) pair with key from the map. Returns the
   // key object or 0 in case key was not found. If map is the owner
   // of values, the value is deleted.

   if (!key) return 0;

   TPair *a;
   if ((a = (TPair *)fTable->FindObject(key))) {
      if (fTable->Remove(key)) {
         if (IsOwnerValue() && a->Value() && a->Value()->IsOnHeap())
            TCollection::GarbageCollect(a->Value());
         TObject *kobj = a->Key();
         delete a;
         fSize--;
         return kobj;
      }
   }
   return 0;
}

//______________________________________________________________________________
TPair *TMap::RemoveEntry(TObject *key)
{
   // Remove (key,value) pair with key from the map. Returns the
   // pair object or 0 in case the key was not found.
   // It is caller's responsibility to delete the pair and, eventually,
   // the key and value objects.

   if (!key) return 0;

   TPair *a;
   if ((a = (TPair *)fTable->FindObject(key))) {
      if (fTable->Remove(key)) {
         fSize--;
         return a;
      }
   }
   return 0;
}

//______________________________________________________________________________
void TMap::SetOwnerValue(Bool_t enable)
{
   // Set whether this map is the owner (enable==true)
   // of its values.  If it is the owner of its contents,
   // these objects will be deleted whenever the collection itself
   // is deleted. The objects might also be deleted or destructed when Clear
   // is called (depending on the collection).

   if (enable)
      SetBit(kIsOwnerValue);
   else
      ResetBit(kIsOwnerValue);
}

//______________________________________________________________________________
void TMap::SetOwnerKeyValue(Bool_t ownkeys, Bool_t ownvals)
{
   // Set ownership for keys and values.

   SetOwner(ownkeys);
   SetOwnerValue(ownvals);
}

//______________________________________________________________________________
void TMap::Streamer(TBuffer &b)
{
   // Stream all key/value pairs in the map to or from the I/O buffer.

   TObject *obj=0;
   UInt_t R__s, R__c;

   if (b.IsReading()) {
      Int_t    nobjects;
      TObject *value=0;

      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 2)
         TObject::Streamer(b);
      if (v > 1)
         fName.Streamer(b);
      b >> nobjects;
      for (Int_t i = 0; i < nobjects; i++) {
         b >> obj;
         b >> value;
         if (obj) Add(obj, value);
      }
      b.CheckByteCount(R__s, R__c,TMap::IsA());
   } else {
      R__c = b.WriteVersion(TMap::IsA(), kTRUE);
      TObject::Streamer(b);
      fName.Streamer(b);
      b << GetSize();
      TIter next(fTable);
      TPair *a;
      while ((a = (TPair*) next())) {
         b << a->Key();
         b << a->Value();
      }
      b.SetByteCount(R__c, kTRUE);
   }
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPair                                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TPair::Browse(TBrowser *b)
{
   // Browse the pair.

   if (b) {
      if (fKey)   b->Add(fKey);
      if (fValue) b->Add(fValue);
   } else {
      if (fKey)   fKey->Browse(b);
      if (fValue) fValue->Browse(b);
   }
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMapIter                                                             //
//                                                                      //
// Iterator of map.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TMapIter)

//______________________________________________________________________________
TMapIter::TMapIter(const TMap *m, Bool_t dir)
{
   // Create a map iterator. Use dir to specify the desired iteration direction.

   fMap        = m;
   fDirection  = dir;
   fCursor     = 0;
}

//______________________________________________________________________________
TMapIter::TMapIter(const TMapIter &iter) : TIterator(iter)
{
   // Copy ctor.

   fMap       = iter.fMap;
   fDirection = iter.fDirection;
   fCursor    = 0;
   if (iter.fCursor) {
      fCursor = (THashTableIter *)iter.fCursor->GetCollection()->MakeIterator();
      if (fCursor)
         fCursor->operator=(*iter.fCursor);
   }
}

//______________________________________________________________________________
TIterator &TMapIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

   if (this != &rhs && rhs.IsA() == TMapIter::Class()) {
      const TMapIter &rhs1 = (const TMapIter &)rhs;
      fMap       = rhs1.fMap;
      fDirection = rhs1.fDirection;
      if (rhs1.fCursor) {
         fCursor = (THashTableIter *)rhs1.fCursor->GetCollection()->MakeIterator();
         if (fCursor)
            fCursor->operator=(*rhs1.fCursor);
      }
   }
   return *this;
}

//______________________________________________________________________________
TMapIter &TMapIter::operator=(const TMapIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fMap       = rhs.fMap;
      fDirection = rhs.fDirection;
      if (rhs.fCursor) {
         fCursor = (THashTableIter *)rhs.fCursor->GetCollection()->MakeIterator();
         if (fCursor)
            fCursor->operator=(*rhs.fCursor);
      }
   }
   return *this;
}

//______________________________________________________________________________
TMapIter::~TMapIter()
{
   // Map iterator dtor.

   Reset();
}

//______________________________________________________________________________
TObject *TMapIter::Next()
{
   // Returns the next key from a map. Use TMap::GetValue() to get the value
   // associated with the key. Returns 0 when no more items in map.

   if (!fCursor)
      fCursor = new THashTableIter(fMap->fTable, fDirection);

   TPair *a = (TPair *)fCursor->Next();
   if (a) return a->Key();
   return 0;
}

//______________________________________________________________________________
void TMapIter::Reset()
{
   // Reset the map iterator.

   SafeDelete(fCursor);
}

//______________________________________________________________________________
bool TMapIter::operator!=(const TIterator &aIter) const
{
   // This operator compares two TIterator objects.

   if (nullptr == (&aIter))
      return fCursor->operator*();

   if (aIter.IsA() == TMapIter::Class()) {
      const TMapIter &iter(dynamic_cast<const TMapIter &>(aIter));
      return (fCursor->operator*() != iter.fCursor->operator*());
   }
   return false; // for base class we don't implement a comparison
}

//______________________________________________________________________________
bool TMapIter::operator!=(const TMapIter &aIter) const
{
   // This operator compares two TMapIter objects.

   if (nullptr == (&aIter))
      return fCursor->operator*();

   return (fCursor->operator*() != aIter.fCursor->operator*());
}

//______________________________________________________________________________
TObject *TMapIter::operator*() const
{
   // Return pointer to current object (a TPair) or nullptr.

   return (fCursor ? fCursor->operator*() : nullptr);
}
