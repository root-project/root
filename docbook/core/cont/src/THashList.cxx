// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
//Begin_Html
/*
<img src=gif/thashlist.gif>
*/
//End_Html
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THashList.h"
#include "THashTable.h"


ClassImp(THashList)

//______________________________________________________________________________
THashList::THashList(Int_t capacity, Int_t rehash)
{
   // Create a THashList object. Capacity is the initial hashtable capacity
   // (i.e. number of slots), by default kInitHashTableCapacity = 17, and
   // rehash is the value at which a rehash will be triggered. I.e. when the
   // average size of the linked lists at a slot becomes longer than rehash
   // then the hashtable will be resized and refilled to reduce the collision
   // rate to about 1. The higher the collision rate, i.e. the longer the
   // linked lists, the longer lookup will take. If rehash=0 the table will
   // NOT automatically be rehashed. Use Rehash() for manual rehashing.
   // WARNING !!!
   // If the name of an object in the HashList is modified, The hashlist
   // must be Rehashed

   fTable = new THashTable(capacity, rehash);
}

//______________________________________________________________________________
THashList::THashList(TObject *, Int_t capacity, Int_t rehash)
{
   // For backward compatibility only. Use other ctor.

   fTable = new THashTable(capacity, rehash);
}

//______________________________________________________________________________
THashList::~THashList()
{
   // Delete a hashlist. Objects are not deleted unless the THashList is the
   // owner (set via SetOwner()).

   Clear();
   SafeDelete(fTable);
}

//______________________________________________________________________________
void THashList::AddFirst(TObject *obj)
{
   // Add object at the beginning of the list.

   TList::AddFirst(obj);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddFirst(TObject *obj, Option_t *opt)
{
   // Add object at the beginning of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   TList::AddFirst(obj, opt);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddLast(TObject *obj)
{
   // Add object at the end of the list.

   TList::AddLast(obj);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddLast(TObject *obj, Option_t *opt)
{
   // Add object at the end of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   TList::AddLast(obj, opt);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddBefore(const TObject *before, TObject *obj)
{
   // Insert object before object before in the list.

   TList::AddBefore(before, obj);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddBefore(TObjLink *before, TObject *obj)
{
   // Insert object before object before in the list.

   TList::AddBefore(before, obj);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddAfter(const TObject *after, TObject *obj)
{
   // Insert object after object after in the list.

   TList::AddAfter(after, obj);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddAfter(TObjLink *after, TObject *obj)
{
   // Insert object after object after in the list.

   TList::AddAfter(after, obj);
   fTable->Add(obj);
}

//______________________________________________________________________________
void THashList::AddAt(TObject *obj, Int_t idx)
{
   // Insert object at location idx in the list.

   TList::AddAt(obj, idx);
   fTable->Add(obj);
}

//______________________________________________________________________________
Float_t THashList::AverageCollisions() const
{
   // Return the average collision rate. The higher the number the longer
   // the linked lists in the hashtable, the slower the lookup. If the number
   // is high, or lookup noticeably too slow, perfrom a Rehash().

   return fTable->AverageCollisions();
}

//______________________________________________________________________________
void THashList::Clear(Option_t *option)
{
   // Remove all objects from the list. Does not delete the objects unless
   // the THashList is the owner (set via SetOwner()).

   fTable->Clear("nodelete");  // clear table so not more lookups
   if (IsOwner())
      TList::Delete(option);
   else
      TList::Clear(option);
}

//______________________________________________________________________________
void THashList::Delete(Option_t *option)
{
   // Remove all objects from the list AND delete all heap based objects.
   // If option="slow" then keep list consistent during delete. This allows
   // recursive list operations during the delete (e.g. during the dtor
   // of an object in this list one can still access the list to search for
   // other not yet deleted objects).

   Bool_t slow = option ? (!strcmp(option, "slow") ? kTRUE : kFALSE) : kFALSE;

   if (!slow) {
      fTable->Clear("nodelete");     // clear table so no more lookups
      TList::Delete(option);         // this deletes the objects
   } else {
      while (fFirst) {
         TObjLink *tlk = fFirst;
         fFirst = fFirst->Next();
         fSize--;
         // remove object from table
         fTable->Remove(tlk->GetObject());
         // delete only heap objects
         if (tlk->GetObject() && tlk->GetObject()->IsOnHeap())
            TCollection::GarbageCollect(tlk->GetObject());

         delete tlk;
      }
      fFirst = fLast = fCache = 0;
      fSize  = 0;
   }
}

//______________________________________________________________________________
TObject *THashList::FindObject(const char *name) const
{
   // Find object using its name. Uses the hash value returned by the
   // TString::Hash() after converting name to a TString.

   return fTable->FindObject(name);
}

//______________________________________________________________________________
TObject *THashList::FindObject(const TObject *obj) const
{
   // Find object using its hash value (returned by its Hash() member).

   return fTable->FindObject(obj);
}

//______________________________________________________________________________
TList *THashList::GetListForObject(const char *name) const
{
   // Return the THashTable's list (bucket) in which obj can be found based on
   // its hash; see THashTable::GetListForObject().

   return fTable->GetListForObject(name);
}

//______________________________________________________________________________
TList *THashList::GetListForObject(const TObject *obj) const
{
   // Return the THashTable's list (bucket) in which obj can be found based on
   // its hash; see THashTable::GetListForObject().

   return fTable->GetListForObject(obj);
}

//______________________________________________________________________________
void THashList::RecursiveRemove(TObject *obj)
{
   // Remove object from this collection and recursively remove the object
   // from all other objects (and collections).
   // This function overrides TCollection::RecursiveRemove that calls
   // the Remove function. THashList::Remove cannot be called because
   // it uses the hash value of the hash table. This hash value
   // is not available anymore when RecursiveRemove is called from
   // the TObject destructor.

   if (!obj) return;

   // Remove obj in the list itself
   TObject *object = TList::Remove(obj);
   if (object) fTable->RemoveSlow(object);

   // Scan again the list and invoke RecursiveRemove for all objects
   TIter next(this);

   while ((object = next())) {
      if (object->TestBit(kNotDeleted)) object->RecursiveRemove(obj);
   }
}

//______________________________________________________________________________
void THashList::Rehash(Int_t newCapacity)
{
   // Rehash the hashlist. If the collision rate becomes too high (i.e.
   // the average size of the linked lists become too long) then lookup
   // efficiency decreases since relatively long lists have to be searched
   // every time. To improve performance rehash the hashtable. This resizes
   // the table to newCapacity slots and refills the table. Use
   // AverageCollisions() to check if you need to rehash.

   fTable->Rehash(newCapacity);
}

//______________________________________________________________________________
TObject *THashList::Remove(TObject *obj)
{
   // Remove object from the list.

   if (!obj || !fTable->FindObject(obj)) return 0;

   TList::Remove(obj);
   return fTable->Remove(obj);
}

//______________________________________________________________________________
TObject *THashList::Remove(TObjLink *lnk)
{
   // Remove object via its objlink from the list.

   if (!lnk) return 0;

   TObject *obj = lnk->GetObject();

   TList::Remove(lnk);
   return fTable->Remove(obj);
}
