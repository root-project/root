// @(#)root/cont:$Name:  $:$Id: TMap.h,v 1.9 2001/12/07 21:58:59 brun Exp $
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

#ifndef ROOT_TCollection
#include "TCollection.h"
#endif
#ifndef ROOT_THashTable
#include "THashTable.h"
#endif

class THashTableIter;
class TMapIter;
class TBrowser;

class TMap : public TCollection {

friend class  TMapIter;

private:
   THashTable   *fTable;         //Hash table used to store TAssociation's

public:
   TMap(Int_t capacity = TCollection::kInitHashTableCapacity, Int_t rehash = 0);
   virtual           ~TMap();
   void              Add(TObject *obj);
   void              Add(TObject *key, TObject *value);
   Float_t           AverageCollisions() const;
   Int_t             Capacity() const;
   void              Clear(Option_t *option="");
   Int_t             Collisions(const char *keyname) const;
   Int_t             Collisions(TObject *key) const;
   void              Delete(Option_t *option="");
   void              DeleteAll();
   TObject          *FindObject(const char *keyname) const;
   TObject          *FindObject(const TObject *key) const;
   TObject         **GetObjectRef(TObject *obj) const {return fTable->GetObjectRef(obj);}
   TObject          *GetValue(TObject *key) const;
   TIterator        *MakeIterator(Bool_t dir = kIterForward) const;
   void              Print(Option_t *option="") const;
   void              Rehash(Int_t newCapacity, Bool_t checkObjValidity = kTRUE);
   TObject          *Remove(TObject *key);

   ClassDef(TMap,3)  //A map
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAssoc                                                               //
//                                                                      //
// Internal class used by TMap to store associations.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TAssoc : public TObject {

private:
   TObject  *fKey;
   TObject  *fValue;

public:
   TAssoc(TObject *key, TObject *value) : fKey(key), fValue(value) { }
   TAssoc(const TAssoc &a) : TObject(), fKey(a.fKey), fValue(a.fValue) { }
   virtual               ~TAssoc() { }
   Bool_t                IsFolder() const { return kTRUE;}
   virtual void          Browse(TBrowser *b);
   const char           *GetName() const { return fKey->GetName(); }
   ULong_t               Hash() const { return fKey->Hash(); }
   Bool_t                IsEqual(const TObject *obj) const { return fKey->IsEqual(obj); }
   TObject              *Key() const { return fKey; }
   TObject              *Value() const { return fValue; }
};


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

   TMapIter() : fMap(0), fCursor(0) { }

public:
   TMapIter(const TMap *map, Bool_t dir = kIterForward);
   TMapIter(const TMapIter &iter);
   ~TMapIter();
   TIterator &operator=(const TIterator &rhs);
   TMapIter  &operator=(const TMapIter &rhs);

   const TCollection *GetCollection() const { return fMap; }
   TObject           *Next();
   void               Reset();

   ClassDef(TMapIter,0)  //Map iterator
};

#endif
