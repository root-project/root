// @(#)root/cont:$Id$
// Author: Fons Rademakers   12/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMap
#define ROOT_TMap


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMap                                                                 //
//                                                                      //
// TMap implements an associative array of (key,value) pairs using a    //
// hash table for efficient retrieval (therefore TMap does not conserve //
// the order of the entries). The hash value is calculated              //
// using the value returned by the keys Hash() function. Both key and   //
// value need to inherit from TObject.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCollection.h"
#include "THashTable.h"

#include <iterator>


class THashTableIter;
class TMapIter;
class TPair;
class TBrowser;


class TMap : public TCollection {

friend class  TMapIter;

private:
   THashTable   *fTable;     //Hash table used to store TPair's

   TMap(const TMap& map) = delete;
   TMap& operator=(const TMap& map) = delete;

protected:
   enum EStatusBits { kIsOwnerValue = BIT(15) };

   void        PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const override;

public:
   typedef TMapIter Iterator_t;

   TMap(Int_t capacity = TCollection::kInitHashTableCapacity, Int_t rehash = 0);
   virtual           ~TMap();
   void              Add(TObject *obj) override;
   void              Add(TObject *key, TObject *value);
   Float_t           AverageCollisions() const;
   Int_t             Capacity() const;
   void              Clear(Option_t *option="") override;
   Int_t             Collisions(const char *keyname) const;
   Int_t             Collisions(TObject *key) const;
   void              Delete(Option_t *option="") override;
   void              DeleteKeys() { Delete(); }
   void              DeleteValues();
   void              DeleteAll();
   Bool_t            DeleteEntry(TObject *key);
   TObject          *FindObject(const char *keyname) const override;
   TObject          *FindObject(const TObject *key) const override;
   TObject         **GetObjectRef(const TObject *obj) const override { return fTable->GetObjectRef(obj); }
   const THashTable *GetTable() const { return fTable; }
   TObject          *GetValue(const char *keyname) const;
   TObject          *GetValue(const TObject *key) const;
   Bool_t            IsOwnerValue() const { return TestBit(kIsOwnerValue); }
   TObject          *operator()(const char *keyname) const { return GetValue(keyname); }
   TObject          *operator()(const TObject *key) const { return GetValue(key); }
   TIterator        *MakeIterator(Bool_t dir = kIterForward) const override;
   void              Rehash(Int_t newCapacity, Bool_t checkObjValidity = kTRUE);
   TObject          *Remove(TObject *key) override;
   TPair            *RemoveEntry(TObject *key);
   virtual void      SetOwnerValue(Bool_t enable = kTRUE);
   virtual void      SetOwnerKeyValue(Bool_t ownkeys = kTRUE, Bool_t ownvals = kTRUE);
   Int_t             Write(const char *name=nullptr, Int_t option=0, Int_t bufsize=0) override;
   Int_t             Write(const char *name=nullptr, Int_t option=0, Int_t bufsize=0) const override;

   ClassDefOverride(TMap,3)  //A (key,value) map
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPair                                                                //
//                                                                      //
// Class used by TMap to store (key,value) pairs.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TPair : public TObject {

private:
   TObject  *fKey;
   TObject  *fValue;

   TPair& operator=(const TPair&) = delete;

public:
   TPair(TObject *key, TObject *value) : fKey(key), fValue(value) { }
   TPair(const TPair &a) : TObject(), fKey(a.fKey), fValue(a.fValue) { }
   virtual               ~TPair();
   Bool_t                IsFolder() const override { return kTRUE;}
   void                  Browse(TBrowser *b) override;
   const char           *GetName() const override { return fKey->GetName(); }
   const char           *GetTitle() const override { return fKey->GetTitle(); }
   ULong_t               Hash() const override { return fKey->Hash(); }
   Bool_t                IsEqual(const TObject *obj) const override { return fKey->IsEqual(obj); }
   TObject              *Key() const { return fKey; }
   TObject              *Value() const { return fValue; }
   void                  SetValue(TObject *val) { fValue = val; }

   ClassDefOverride(TPair,0); // Pair TObject*, TObject*
};

typedef TPair   TAssoc;     // for backward compatibility


// Preventing warnings with -Weffc++ in GCC since it is a false positive for the TMapIter destructor.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMapIter                                                             //
//                                                                      //
// Iterator of a map.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMapIter : public TIterator {

private:
   const TMap       *fMap;         //map being iterated
   THashTableIter   *fCursor;      //current position in map
   Bool_t            fDirection;   //iteration direction

   TMapIter() : fMap(nullptr), fCursor(nullptr), fDirection(kIterForward) { }

public:
   using iterator_category = std::bidirectional_iterator_tag;
   using value_type = TObject *;
   using difference_type = std::ptrdiff_t;
   using pointer = TObject **;
   using const_pointer = const TObject **;
   using reference = const TObject *&;

   TMapIter(const TMap *map, Bool_t dir = kIterForward);
   TMapIter(const TMapIter &iter);
   ~TMapIter();
   TIterator &operator=(const TIterator &rhs) override;
   TMapIter  &operator=(const TMapIter &rhs);

   const TCollection *GetCollection() const override { return fMap; }
   TObject           *Next() override;
   void               Reset() override;
   Bool_t             operator!=(const TIterator &aIter) const override;
   Bool_t             operator!=(const TMapIter &aIter) const;
   TObject           *operator*() const override;

   ClassDefOverride(TMapIter,0)  //Map iterator
};

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif

#endif
