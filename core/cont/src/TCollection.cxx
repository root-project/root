// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Collection abstract base class. This class describes the base        //
// protocol all collection classes have to implement. The ROOT          //
// collection classes always store pointers to objects that inherit     //
// from TObject. They never adopt the objects. Therefore, it is the     //
// user's responsability to take care of deleting the actual objects    //
// once they are not needed anymore. In exceptional cases, when the     //
// user is 100% sure nothing else is referencing the objects in the     //
// collection, one can delete all objects and the collection at the     //
// same time using the Delete() function.                               //
//                                                                      //
// Collections can be iterated using an iterator object (see            //
// TIterator). Depending on the concrete collection class there may be  //
// some additional methods of iterating. See the repective classes.     //
//                                                                      //
// TCollection inherits from TObject since we want to be able to have   //
// collections of collections.                                          //
//                                                                      //
// In a later release the collections may become templatized.           //
//                                                                      //
//Begin_Html
/*
<img src="gif/tcollection_classtree.gif">
*/
//End_Html
//////////////////////////////////////////////////////////////////////////

#include "TCollection.h"
#include "Riostream.h"
#include "Varargs.h"
#include "TClass.h"
#include "TROOT.h"
#include "TBrowser.h"
#include "TObjectTable.h"
#include "TRegexp.h"
#include "TPRegexp.h"
#include "TVirtualMutex.h"

TVirtualMutex *gCollectionMutex = 0;

TCollection   *TCollection::fgCurrentCollection = 0;
TObjectTable  *TCollection::fgGarbageCollection = 0;
Bool_t         TCollection::fgEmptyingGarbage   = kFALSE;
Int_t          TCollection::fgGarbageStack      = 0;

ClassImp(TCollection)
ClassImp(TIter)

//______________________________________________________________________________
void TCollection::AddAll(const TCollection *col)
{
   // Add all objects from collection col to this collection.

   TIter next(col);
   TObject *obj;

   while ((obj = next()))
      Add(obj);
}

//______________________________________________________________________________
void TCollection::AddVector(TObject *va_(obj1), ...)
{
   // Add all arguments to the collection. The list of objects must be
   // temrinated by 0, e.g.: l.AddVector(o1, o2, o3, o4, 0);

   va_list ap;
   va_start(ap, va_(obj1));
   TObject *obj;

   Add(va_(obj1));
   while ((obj = va_arg(ap, TObject *)))
      Add(obj);
   va_end(ap);
}

//______________________________________________________________________________
Bool_t TCollection::AssertClass(TClass *cl) const
{
   // Make sure all objects in this collection inherit from class cl.

   TObject *obj;
   TIter    next(this);
   Bool_t   error = kFALSE;

   if (!cl) {
      Error("AssertClass", "class == 0");
      return kTRUE;
   }

   for (int i = 0; (obj = next()); i++)
      if (!obj->InheritsFrom(cl)) {
         Error("AssertClass", "element %d is not an instance of class %s (%s)",
               i, cl->GetName(), obj->ClassName());
         error = kTRUE;
      }
   return error;
}

//______________________________________________________________________________
void TCollection::Browse(TBrowser *b)
{
   // Browse this collection (called by TBrowser).
   // If b=0, there is no Browse call TObject::Browse(0) instead.
   //         This means TObject::Inspect() will be invoked indirectly

   TIter next(this);
   TObject *obj;

   if (b)
      while ((obj = next())) b->Add(obj);
   else
      TObject::Browse(b);
}

//______________________________________________________________________________
TObject *TCollection::Clone(const char *newname) const
{
   // Make a clone of an collection using the Streamer facility.
   // If newname is specified, this will be the name of the new collection.

   TCollection *new_collection = (TCollection*)TObject::Clone(newname);
   if (newname && strlen(newname)) new_collection->SetName(newname);
   return new_collection;
}


//______________________________________________________________________________
Int_t TCollection::Compare(const TObject *obj) const
{
   // Compare two TCollection objects. Returns 0 when equal, -1 when this is
   // smaller and +1 when bigger (like strcmp()).

   if (this == obj) return 0;
   return fName.CompareTo(obj->GetName());
}

//______________________________________________________________________________
void TCollection::Draw(Option_t *option)
{
   // Draw all objects in this collection.

   TIter next(this);
   TObject *object;

   while ((object = next())) {
      object->Draw(option);
   }
}

//______________________________________________________________________________
void TCollection::Dump() const
{
   // Dump all objects in this collection.

   TIter next(this);
   TObject *object;

   while ((object = next())) {
      object->Dump();
   }
}

//______________________________________________________________________________
TObject *TCollection::FindObject(const char *name) const
{
   // Find an object in this collection using its name. Requires a sequential
   // scan till the object has been found. Returns 0 if object with specified
   // name is not found.

   TIter next(this);
   TObject *obj;

   while ((obj = next()))
      if (!strcmp(name, obj->GetName())) return obj;
   return 0;
}

//______________________________________________________________________________
TObject *TCollection::operator()(const char *name) const
{
  // Find an object in this collection by name.

   return FindObject(name);
}

//______________________________________________________________________________
TObject *TCollection::FindObject(const TObject *obj) const
{
   // Find an object in this collection using the object's IsEqual()
   // member function. Requires a sequential scan till the object has
   // been found. Returns 0 if object is not found.
   // Typically this function is overridden by a more efficient version
   // in concrete collection classes (e.g. THashTable).

   TIter next(this);
   TObject *ob;

   while ((ob = next()))
      if (ob->IsEqual(obj)) return ob;
   return 0;
}

//______________________________________________________________________________
const char *TCollection::GetName() const
{
  // Return name of this collection.
  // if no name, return the collection class name.

   if (fName.Length() > 0) return fName.Data();
   return ClassName();
}

//______________________________________________________________________________
Int_t TCollection::GrowBy(Int_t delta) const
{
  // Increase the collection's capacity by delta slots.

   if (delta < 0) {
      Error("GrowBy", "delta < 0");
      delta = Capacity();
   }
   return Capacity() + TMath::Range(2, kMaxInt - Capacity(), delta);
}

//______________________________________________________________________________
Bool_t  TCollection::IsArgNull(const char *where, const TObject *obj) const
{
   // Returns true if object is a null pointer.

   return obj ? kFALSE : (Error(where, "argument is a null pointer"), kTRUE);
}

//______________________________________________________________________________
void TCollection::ls(Option_t *option) const
{
   // List (ls) all objects in this collection.
   // Wildcarding supported, eg option="xxx*" lists only objects
   // with names xxx*.

   TROOT::IndentLevel();
   std::cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << " : "
        << Int_t(TestBit(kCanDelete)) << std::endl;

   TRegexp re(option,kTRUE);
   TIter next(this);
   TObject *object;
   char *star = 0;
   if (option) star = (char*)strchr(option,'*');

   TROOT::IncreaseDirLevel();
   while ((object = next())) {
      if (star) {
         TString s = object->GetName();
         if (s != option && s.Index(re) == kNPOS) continue;
      }
      object->ls(option);
   }
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
void TCollection::Paint(Option_t *option)
{
   // Paint all objects in this collection.

   this->R__FOR_EACH(TObject,Paint)(option);
}

//______________________________________________________________________________
void TCollection::PrintCollectionHeader(Option_t*) const
{
   // Print the collection header.

   TROOT::IndentLevel();
   printf("Collection name='%s', class='%s', size=%d\n",
          GetName(), ClassName(), GetSize());
}

//______________________________________________________________________________
const char* TCollection::GetCollectionEntryName(TObject* entry) const
{
   // For given collection entry return the string that is used to
   // identify the object and, potentially, perform wildcard/regexp
   // filtering on.

   return entry->GetName();
}

//______________________________________________________________________________
void TCollection::PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const
{
   // Print the collection entry.

   TCollection* coll = dynamic_cast<TCollection*>(entry);
   if (coll) {
      coll->Print(option, recurse);
   } else {
      TROOT::IndentLevel();
      entry->Print(option);
   }
}

//______________________________________________________________________________
void TCollection::Print(Option_t *option) const
{
   // Default print for collections, calls Print(option, 1).
   // This will print the collection header and Print() methods of
   // all the collection entries.
   //
   // If you want to override Print() for a collection class, first
   // see if you can accomplish it by overriding the following protected
   // methods:
   //   void        PrintCollectionHeader(Option_t* option) const;
   //   const char* GetCollectionEntryName(TObject* entry) const;
   //   void        PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const;
   // Otherwise override the Print(Option_t *option, Int_t)
   // variant. Remember to declare:
   //   using TCollection::Print;
   // somewhere close to the method declaration.

   Print(option, 1);
}

//______________________________________________________________________________
void TCollection::Print(Option_t *option, Int_t recurse) const
{
   // Print the collection header and its elements.
   //
   // If recurse is non-zero, descend into printing of
   // collection-entries with recurse - 1.
   // This means, if recurse is negative, the recursion is infinite.
   //
   // Option is passed recursively.

   PrintCollectionHeader(option);

   if (recurse != 0)
   {
      TIter next(this);
      TObject *object;

      TROOT::IncreaseDirLevel();
      while ((object = next())) {
         PrintCollectionEntry(object, option, recurse - 1);
      }
      TROOT::DecreaseDirLevel();
   }
}

//______________________________________________________________________________
void TCollection::Print(Option_t *option, const char* wildcard, Int_t recurse) const
{
   // Print the collection header and its elements that match the wildcard.
   //
   // If recurse is non-zero, descend into printing of
   // collection-entries with recurse - 1.
   // This means, if recurse is negative, the recursion is infinite.
   //
   // Option is passed recursively, but wildcard is only used on the
   // first level.

   PrintCollectionHeader(option);

   if (recurse != 0)
   {
      if (!wildcard) wildcard = "";
      TRegexp re(wildcard, kTRUE);
      Int_t nch = strlen(wildcard);
      TIter next(this);
      TObject *object;

      TROOT::IncreaseDirLevel();
      while ((object = next())) {
         TString s = GetCollectionEntryName(object);
         if (nch == 0 || s == wildcard || s.Index(re) != kNPOS) {
            PrintCollectionEntry(object, option, recurse - 1);
         }
      }
      TROOT::DecreaseDirLevel();
   }
}

//______________________________________________________________________________
void TCollection::Print(Option_t *option, TPRegexp& regexp, Int_t recurse) const
{
   // Print the collection header and its elements that match the regexp.
   //
   // If recurse is non-zero, descend into printing of
   // collection-entries with recurse - 1.
   // This means, if recurse is negative, the recursion is infinite.
   //
   // Option is passed recursively, but regexp is only used on the
   // first level.

   PrintCollectionHeader(option);

   if (recurse != 0)
   {
      TIter next(this);
      TObject *object;

      TROOT::IncreaseDirLevel();
      while ((object = next())) {
         TString s = GetCollectionEntryName(object);
         if (regexp.MatchB(s)) {
            PrintCollectionEntry(object, option, recurse - 1);
         }
      }
      TROOT::DecreaseDirLevel();
   }
}

//______________________________________________________________________________
void TCollection::RecursiveRemove(TObject *obj)
{
   // Remove object from this collection and recursively remove the object
   // from all other objects (and collections).

   if (!obj) return;

   // Scan list and remove obj in the list itself
   while (Remove(obj))
      ;

   // Scan again the list and invoke RecursiveRemove for all objects
   TIter next(this);
   TObject *object;

   while ((object = next())) {
      if (object->TestBit(kNotDeleted)) object->RecursiveRemove(obj);
   }
}

//______________________________________________________________________________
void TCollection::RemoveAll(TCollection *col)
{
   // Remove all objects in collection col from this collection.

   TIter next(col);
   TObject *obj;

   while ((obj = next()))
      Remove(obj);
}

//_______________________________________________________________________
void TCollection::Streamer(TBuffer &b)
{
   // Stream all objects in the collection to or from the I/O buffer.

   Int_t nobjects;
   TObject *obj;
   UInt_t R__s, R__c;

   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 2)
         TObject::Streamer(b);
      if (v > 1)
         fName.Streamer(b);
      b >> nobjects;
      for (Int_t i = 0; i < nobjects; i++) {
         b >> obj;
         Add(obj);
      }
      b.CheckByteCount(R__s, R__c,TCollection::IsA());
   } else {
      R__c = b.WriteVersion(TCollection::IsA(), kTRUE);
      TObject::Streamer(b);
      fName.Streamer(b);
      nobjects = GetSize();
      b << nobjects;

      TIter next(this);

      while ((obj = next())) {
         b << obj;
      }
      b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
Int_t TCollection::Write(const char *name, Int_t option, Int_t bsize) const
{
   // Write all objects in this collection. By default all objects in
   // the collection are written individually (each object gets its
   // own key). Note, this is recursive, i.e. objects in collections
   // in the collection are also written individually. To write all
   // objects using a single key specify a name and set option to
   // TObject::kSingleKey (i.e. 1).

   if ((option & kSingleKey)) {
      return TObject::Write(name, option, bsize);
   } else {
      option &= ~kSingleKey;
      Int_t nbytes = 0;
      TIter next(this);
      TObject *obj;
      while ((obj = next())) {
         nbytes += obj->Write(name, option, bsize);
      }
      return nbytes;
   }
}

//______________________________________________________________________________
Int_t TCollection::Write(const char *name, Int_t option, Int_t bsize)
{
   // Write all objects in this collection. By default all objects in
   // the collection are written individually (each object gets its
   // own key). Note, this is recursive, i.e. objects in collections
   // in the collection are also written individually. To write all
   // objects using a single key specify a name and set option to
   // TObject::kSingleKey (i.e. 1).

   return ((const TCollection*)this)->Write(name,option,bsize);
}

// -------------------- Static data members access -----------------------------
//______________________________________________________________________________
TCollection *TCollection::GetCurrentCollection()
{
   // Return the globally accessible collection.

   return fgCurrentCollection;
}

//______________________________________________________________________________
void TCollection::SetCurrentCollection()
{
   // Set this collection to be the globally accesible collection.

   fgCurrentCollection = this;
}

//______________________________________________________________________________
void TCollection::StartGarbageCollection()
{
   // Set up for garbage collection.

   R__LOCKGUARD2(gCollectionMutex);
   if (!fgGarbageCollection) {
      fgGarbageCollection = new TObjectTable;
      fgEmptyingGarbage   = kFALSE;
      fgGarbageStack      = 0;
   }
   fgGarbageStack++;
}

//______________________________________________________________________________
void TCollection::EmptyGarbageCollection()
{
   // Do the garbage collection.

   R__LOCKGUARD2(gCollectionMutex);
   if (fgGarbageStack > 0) fgGarbageStack--;
   if (fgGarbageCollection && fgGarbageStack == 0 && fgEmptyingGarbage == kFALSE) {
      fgEmptyingGarbage = kTRUE;
      fgGarbageCollection->Delete();
      fgEmptyingGarbage = kFALSE;
      SafeDelete(fgGarbageCollection);
   }
}

//______________________________________________________________________________
void TCollection::GarbageCollect(TObject *obj)
{
   // Add to the list of things to be cleaned up.

   {
      R__LOCKGUARD2(gCollectionMutex);
      if (fgGarbageCollection) {
         if (!fgEmptyingGarbage) {
            fgGarbageCollection->Add(obj);
            return;
         }
      }
   }
   delete obj;
}

//______________________________________________________________________________
void TCollection::SetOwner(Bool_t enable)
{
   // Set whether this collection is the owner (enable==true)
   // of its content.  If it is the owner of its contents,
   // these objects will be deleted whenever the collection itself
   // is delete.   The objects might also be deleted or destructed when Clear
   // is called (depending on the collection).

   if (enable)
      SetBit(kIsOwner);
   else
      ResetBit(kIsOwner);
}

//______________________________________________________________________________
TIter::TIter(const TIter &iter)
{
   // Copy a TIter. This involves allocating a new TIterator of the right
   // sub class and assigning it with the original.

   if (iter.fIterator) {
      fIterator = iter.GetCollection()->MakeIterator();
      fIterator->operator=(*iter.fIterator);
   } else
      fIterator = 0;
}

//______________________________________________________________________________
TIter &TIter::operator=(const TIter &rhs)
{
   // Assigning an TIter to another. This involves allocatiing a new TIterator
   // of the right sub class and assigning it with the original.

   if (this != &rhs) {
      if (rhs.fIterator) {
         delete fIterator;
         fIterator = rhs.GetCollection()->MakeIterator();
         fIterator->operator=(*rhs.fIterator);
      }
   }
   return *this;
}

//______________________________________________________________________________
TIter &TIter::Begin()
{
   // Pointing to the first element of the container.

   fIterator->Reset();
   fIterator->Next();
   return *this;
}

//______________________________________________________________________________
TIter TIter::End()
{
   // Pointing to the element after the last - to a nullptr value in our case.

   return TIter(static_cast<TIterator*>(nullptr));
}
