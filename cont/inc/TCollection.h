// @(#)root/cont:$Name:  $:$Id: TCollection.h,v 1.2 2000/05/24 10:31:48 brun Exp $
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


const Bool_t kIterForward  = kTRUE;
const Bool_t kIterBackward = !kIterForward;


class TCollection : public TObject {

private:
   static TCollection  *fgCurrentCollection;  //used by macro ForEach
   static TObjectTable *fgGarbageCollection;  //used by garbage collector
   static Bool_t        fgEmptyingGarbage;    //used by garbage collector
   static Int_t         fgGarbageStack;       //used by garbage collector

   TCollection(const TCollection &);    // private and not-implemented, collections
   void operator=(const TCollection &); // are too sensitive to be automatically copied

protected:
   TString   fName;               //name of the collection
   TObject  *fParent;             //parent object of collection
   Int_t     fSize;               //number of elements in collection

   TCollection() : fParent(0), fSize(0) { }
   TCollection(TObject *parent) : fParent(parent), fSize(0) { }

public:
   enum { kInitCapacity = 16, kInitHashTableCapacity = 17 };

   virtual            ~TCollection() { }
   virtual void       Add(TObject *obj) = 0;
   void               AddVector(TObject *obj1, ...);
   virtual void       AddAll(TCollection *col);
   Bool_t             AssertClass(TClass *cl) const;
   void               Browse(TBrowser *b);
   Int_t              Capacity() const { return fSize; }
   virtual void       Clear(Option_t *option="") = 0;
   Bool_t             Contains(const char *name) const { return FindObject(name) != 0; }
   Bool_t             Contains(TObject *obj) const { return FindObject(obj) != 0; }
   virtual void       Delete(Option_t *option="") = 0;
   virtual void       Draw(Option_t *option="");
   virtual void       Dump();
   virtual TObject   *FindObject(const char *name) const;
   TObject           *operator()(const char *name) const;
   virtual TObject   *FindObject(TObject *obj) const;
   virtual const char *GetName() const { return fName.Data(); }
   TObject           *GetParent() const { return fParent; }
   virtual Int_t      GetSize() const { return fSize; }
   virtual Int_t      GrowBy(Int_t delta) const;
   Bool_t             IsArgNull(const char *where, TObject *obj) const;
   virtual Bool_t     IsEmpty() const { return GetSize() <= 0; }
   Bool_t             IsFolder() const { return kTRUE; }
   virtual void       ls(Option_t *option="");
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const = 0;
   virtual TIterator *MakeReverseIterator() const { return MakeIterator(kIterBackward); }
   virtual void       Paint(Option_t *option="");
   virtual void       Print(Option_t *option="");
   virtual void       RecursiveRemove(TObject *obj);
   virtual TObject   *Remove(TObject *obj) = 0;
   virtual void       RemoveAll(TCollection *col);
   void               RemoveAll() { Clear(); }
   void               SetCurrentCollection();
   virtual void       SetName(const char *name) { fName = name; }
   void               SetParent(TObject *parent) { fParent = parent; }
   virtual Int_t      Write(const char *name=0, Int_t option=0, Int_t bufsize=0);

   static TCollection  *GetCurrentCollection();
   static void          StartGarbageCollection();
   static void          GarbageCollect(TObject *obj);
   static void          EmptyGarbageCollection();

   ClassDef(TCollection,2)  //Collection abstract base class
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
   TIter() : fIterator(0) { }

public:
   TIter(const TCollection *col, Bool_t dir = kIterForward)
        : fIterator(col ? col->MakeIterator(dir) : 0) { }
   TIter(TIterator *it) : fIterator(it) { }
   TIter(const TIter &iter);
   TIter &operator=(const TIter &rhs);
   virtual            ~TIter() { SafeDelete(fIterator) }
   TObject           *operator()() { return fIterator ? fIterator->Next() : 0; }
   TObject           *Next() { return fIterator ? fIterator->Next() : 0; }
   const TCollection *GetCollection() const { return fIterator ? fIterator->GetCollection() : 0; }
   Option_t          *GetOption() const { return fIterator ? fIterator->GetOption() : ""; }
   void               Reset() { if (fIterator) fIterator->Reset(); }

   ClassDef(TIter,0)  //Iterator wrapper
};


//---- ForEach macro -----------------------------------------------------------

// Macro to loop over all elements of a list of type "type" while executing
// procedure "proc" on each element

#define ForEach(type,proc) \
    SetCurrentCollection(); \
    TIter _NAME3_(type,proc,_nxt)(TCollection::GetCurrentCollection()); \
    type *_NAME3_(type,proc,_obj); \
    while ((_NAME3_(type,proc,_obj) = (type*) _NAME3_(type,proc,_nxt)())) \
       _NAME3_(type,proc,_obj)->proc

#endif
