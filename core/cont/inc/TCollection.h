// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCollection
#define ROOT_TCollection


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCollection                                                          //
//                                                                      //
// Collection abstract base class. This class inherits from TObject     //
// because we want to be able to have collections of collections.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TIterator
#include "TIterator.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TClass;
class TObjectTable;
class TVirtualMutex;
class TIter;

const Bool_t kIterForward  = kTRUE;
const Bool_t kIterBackward = !kIterForward;

R__EXTERN TVirtualMutex *gCollectionMutex;

class TCollection : public TObject {

private:
   static TCollection  *fgCurrentCollection;  //used by macro R__FOR_EACH
   static TObjectTable *fgGarbageCollection;  //used by garbage collector
   static Bool_t        fgEmptyingGarbage;    //used by garbage collector
   static Int_t         fgGarbageStack;       //used by garbage collector

   TCollection(const TCollection &);    //private and not-implemented, collections
   void operator=(const TCollection &); //are too complex to be automatically copied

protected:
   enum { kIsOwner = BIT(14) };

   TString   fName;               //name of the collection
   Int_t     fSize;               //number of elements in collection

   TCollection() : fName(), fSize(0) { }

   virtual void        PrintCollectionHeader(Option_t* option) const;
   virtual const char* GetCollectionEntryName(TObject* entry) const;
   virtual void        PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const;

public:
   enum { kInitCapacity = 16, kInitHashTableCapacity = 17 };

   virtual            ~TCollection() { }
   virtual void       Add(TObject *obj) = 0;
   void               AddVector(TObject *obj1, ...);
   virtual void       AddAll(const TCollection *col);
   Bool_t             AssertClass(TClass *cl) const;
   void               Browse(TBrowser *b);
   Int_t              Capacity() const { return fSize; }
   virtual void       Clear(Option_t *option="") = 0;
   virtual TObject   *Clone(const char *newname="") const;
   Int_t              Compare(const TObject *obj) const;
   Bool_t             Contains(const char *name) const { return FindObject(name) != 0; }
   Bool_t             Contains(const TObject *obj) const { return FindObject(obj) != 0; }
   virtual void       Delete(Option_t *option="") = 0;
   virtual void       Draw(Option_t *option="");
   virtual void       Dump() const ;
   virtual TObject   *FindObject(const char *name) const;
   TObject           *operator()(const char *name) const;
   virtual TObject   *FindObject(const TObject *obj) const;
   virtual Int_t      GetEntries() const { return GetSize(); }
   virtual const char *GetName() const;
   virtual TObject  **GetObjectRef(const TObject *obj) const = 0;
   virtual Int_t      GetSize() const { return fSize; }
   virtual Int_t      GrowBy(Int_t delta) const;
   ULong_t            Hash() const { return fName.Hash(); }
   Bool_t             IsArgNull(const char *where, const TObject *obj) const;
   virtual Bool_t     IsEmpty() const { return GetSize() <= 0; }
   virtual Bool_t     IsFolder() const { return kTRUE; }
   Bool_t             IsOwner() const { return TestBit(kIsOwner); }
   Bool_t             IsSortable() const { return kTRUE; }
   virtual void       ls(Option_t *option="") const ;
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const = 0;
   virtual TIterator *MakeReverseIterator() const { return MakeIterator(kIterBackward); }
   virtual void       Paint(Option_t *option="");
   virtual void       Print(Option_t *option="") const;
   virtual void       Print(Option_t *option, Int_t recurse) const;
   virtual void       Print(Option_t *option, const char* wildcard, Int_t recurse=1) const;
   virtual void       Print(Option_t *option, TPRegexp& regexp, Int_t recurse=1) const;
   virtual void       RecursiveRemove(TObject *obj);
   virtual TObject   *Remove(TObject *obj) = 0;
   virtual void       RemoveAll(TCollection *col);
   void               RemoveAll() { Clear(); }
   void               SetCurrentCollection();
   void               SetName(const char *name) { fName = name; }
   virtual void       SetOwner(Bool_t enable = kTRUE);
   virtual Int_t      Write(const char *name=0, Int_t option=0, Int_t bufsize=0);
   virtual Int_t      Write(const char *name=0, Int_t option=0, Int_t bufsize=0) const;

   static TCollection  *GetCurrentCollection();
   static void          StartGarbageCollection();
   static void          GarbageCollect(TObject *obj);
   static void          EmptyGarbageCollection();

   TIter begin() const;
   TIter end() const;

   ClassDef(TCollection,3)  //Collection abstract base class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIter                                                                //
//                                                                      //
// Iterator wrapper. Type of iterator used depends on type of           //
// collection.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TIter {

private:
   TIterator    *fIterator;         //collection iterator

protected:
   TIter() : fIterator(nullptr) { }

public:
   TIter(const TCollection *col, Bool_t dir = kIterForward)
         : fIterator(col ? col->MakeIterator(dir) : 0) { }
   TIter(TIterator *it) : fIterator(it) { }
   TIter(const TIter &iter);
   TIter &operator=(const TIter &rhs);
   virtual ~TIter() { SafeDelete(fIterator); }
   TObject           *operator()() { return Next(); }
   TObject           *Next() { return fIterator ? fIterator->Next() : nullptr; }
   const TCollection *GetCollection() const { return fIterator ? fIterator->GetCollection() : nullptr; }
   Option_t          *GetOption() const { return fIterator ? fIterator->GetOption() : ""; }
   void               Reset() { if (fIterator) fIterator->Reset(); }
   TIter             &operator++() { Next(); return *this; }
   Bool_t             operator==(const TIter &aIter) const {
      if (fIterator == nullptr)
         return aIter.fIterator == nullptr || **aIter.fIterator == nullptr;
      if (aIter.fIterator == nullptr)
         return fIterator == nullptr || **fIterator == nullptr;
      return *fIterator == *aIter.fIterator;
   }
   Bool_t             operator!=(const TIter &aIter) const {
      return !(*this == aIter);
   }
   TObject           *operator*() const { return fIterator ? *(*fIterator): nullptr; }
   TIter             &Begin();
   static TIter       End();

   ClassDef(TIter,0)  //Iterator wrapper
};

template <class T>
class TIterCategory: public TIter, public std::iterator_traits<typename T::Iterator_t> {

public:
   TIterCategory(const TCollection *col, Bool_t dir = kIterForward) : TIter(col, dir) { }
   TIterCategory(TIterator *it) : TIter(it) { }
   virtual ~TIterCategory() { }
   TIterCategory &Begin() { TIter::Begin(); return *this; }
   static TIterCategory End() { return TIterCategory(static_cast<TIterator*>(nullptr)); }
};


inline TIter TCollection::begin() const { return ++(TIter(this)); }
inline TIter TCollection::end() const { return TIter::End(); }


//---- R__FOR_EACH macro -------------------------------------------------------

// Macro to loop over all elements of a list of type "type" while executing
// procedure "proc" on each element

#define R__FOR_EACH(type,proc) \
    SetCurrentCollection(); \
    TIter _NAME3_(nxt_,type,proc)(TCollection::GetCurrentCollection()); \
    type *_NAME3_(obj_,type,proc); \
    while ((_NAME3_(obj_,type,proc) = (type*) _NAME3_(nxt_,type,proc)())) \
       _NAME3_(obj_,type,proc)->proc

#endif
