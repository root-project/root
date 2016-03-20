// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class THashList
\ingroup Containers
THashList implements a hybrid collection class consisting of a
hash table and a list to store TObject's. The hash table is used for
quick access and lookup of objects while the list allows the objects
to be ordered. The hash value is calculated using the value returned
by the TObject's Hash() function. Each class inheriting from TObject
can override Hash() as it sees fit.
*/

#include "THashList.h"
#include "THashTable.h"


ClassImp(THashList)

////////////////////////////////////////////////////////////////////////////////
/// Create a THashList object. Capacity is the initial hashtable capacity
/// (i.e. number of slots), by default kInitHashTableCapacity = 17, and
/// rehash is the value at which a rehash will be triggered. I.e. when the
/// average size of the linked lists at a slot becomes longer than rehash
/// then the hashtable will be resized and refilled to reduce the collision
/// rate to about 1. The higher the collision rate, i.e. the longer the
/// linked lists, the longer lookup will take. If rehash=0 the table will
/// NOT automatically be rehashed. Use Rehash() for manual rehashing.
///
/// WARNING !!!
/// If the name of an object in the HashList is modified, The hashlist
/// must be Rehashed

THashList::THashList(Int_t capacity, Int_t rehash)
{
   fTable = new THashTable(capacity, rehash);
}

////////////////////////////////////////////////////////////////////////////////
/// For backward compatibility only. Use other ctor.

THashList::THashList(TObject *, Int_t capacity, Int_t rehash)
{
   fTable = new THashTable(capacity, rehash);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete a hashlist. Objects are not deleted unless the THashList is the
/// owner (set via SetOwner()).

THashList::~THashList()
{
   Clear();
   SafeDelete(fTable);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list.

void THashList::AddFirst(TObject *obj)
{
   TList::AddFirst(obj);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list and also store option.
/// Storing an option is useful when one wants to change the behaviour
/// of an object a little without having to create a complete new
/// copy of the object. This feature is used, for example, by the Draw()
/// method. It allows the same object to be drawn in different ways.

void THashList::AddFirst(TObject *obj, Option_t *opt)
{
   TList::AddFirst(obj, opt);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list.

void THashList::AddLast(TObject *obj)
{
   TList::AddLast(obj);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list and also store option.
/// Storing an option is useful when one wants to change the behaviour
/// of an object a little without having to create a complete new
/// copy of the object. This feature is used, for example, by the Draw()
/// method. It allows the same object to be drawn in different ways.

void THashList::AddLast(TObject *obj, Option_t *opt)
{
   TList::AddLast(obj, opt);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void THashList::AddBefore(const TObject *before, TObject *obj)
{
   TList::AddBefore(before, obj);
   fTable->AddBefore(before, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void THashList::AddBefore(TObjLink *before, TObject *obj)
{
   TList::AddBefore(before, obj);
   fTable->AddBefore(before->GetObject(), obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void THashList::AddAfter(const TObject *after, TObject *obj)
{
   TList::AddAfter(after, obj);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void THashList::AddAfter(TObjLink *after, TObject *obj)
{
   TList::AddAfter(after, obj);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object at location idx in the list.

void THashList::AddAt(TObject *obj, Int_t idx)
{
   TList::AddAt(obj, idx);
   fTable->Add(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the average collision rate. The higher the number the longer
/// the linked lists in the hashtable, the slower the lookup. If the number
/// is high, or lookup noticeably too slow, perform a Rehash().

Float_t THashList::AverageCollisions() const
{
   return fTable->AverageCollisions();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from the list. Does not delete the objects unless
/// the THashList is the owner (set via SetOwner()).

void THashList::Clear(Option_t *option)
{
   fTable->Clear("nodelete");  // clear table so not more lookups
   if (IsOwner())
      TList::Delete(option);
   else
      TList::Clear(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from the list AND delete all heap based objects.
/// If option="slow" then keep list consistent during delete. This allows
/// recursive list operations during the delete (e.g. during the dtor
/// of an object in this list one can still access the list to search for
/// other not yet deleted objects).

void THashList::Delete(Option_t *option)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Find object using its name. Uses the hash value returned by the
/// TString::Hash() after converting name to a TString.

TObject *THashList::FindObject(const char *name) const
{
   return fTable->FindObject(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Find object using its hash value (returned by its Hash() member).

TObject *THashList::FindObject(const TObject *obj) const
{
   return fTable->FindObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the THashTable's list (bucket) in which obj can be found based on
/// its hash; see THashTable::GetListForObject().

const TList *THashList::GetListForObject(const char *name) const
{
   return fTable->GetListForObject(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the THashTable's list (bucket) in which obj can be found based on
/// its hash; see THashTable::GetListForObject().

const TList *THashList::GetListForObject(const TObject *obj) const
{
   return fTable->GetListForObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from this collection and recursively remove the object
/// from all other objects (and collections).
/// This function overrides TCollection::RecursiveRemove that calls
/// the Remove function. THashList::Remove cannot be called because
/// it uses the hash value of the hash table. This hash value
/// is not available anymore when RecursiveRemove is called from
/// the TObject destructor.

void THashList::RecursiveRemove(TObject *obj)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Rehash the hashlist. If the collision rate becomes too high (i.e.
/// the average size of the linked lists become too long) then lookup
/// efficiency decreases since relatively long lists have to be searched
/// every time. To improve performance rehash the hashtable. This resizes
/// the table to newCapacity slots and refills the table. Use
/// AverageCollisions() to check if you need to rehash.

void THashList::Rehash(Int_t newCapacity)
{
   fTable->Rehash(newCapacity);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from the list.

TObject *THashList::Remove(TObject *obj)
{
   if (!obj || !fTable->FindObject(obj)) return 0;

   TList::Remove(obj);
   return fTable->Remove(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object via its objlink from the list.

TObject *THashList::Remove(TObjLink *lnk)
{
   if (!lnk) return 0;

   TObject *obj = lnk->GetObject();

   TList::Remove(lnk);
   return fTable->Remove(obj);
}
