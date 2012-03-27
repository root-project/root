// @(#)root/cont:$Id$
// Author: Fons Rademakers   10/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TList
#define ROOT_TList


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TList                                                                //
//                                                                      //
// A doubly linked list. All classes inheriting from TObject can be     //
// inserted in a TList.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

#include <iterator>

#if (__GNUC__ >= 3) && !defined(__INTEL_COMPILER)
// Prevent -Weffc++ from complaining about the inheritance
// TListIter from std::iterator.
#pragma GCC system_header
#endif

const Bool_t kSortAscending  = kTRUE;
const Bool_t kSortDescending = !kSortAscending;

class TObjLink;
class TListIter;


class TList : public TSeqCollection {

friend  class TListIter;

protected:
   TObjLink  *fFirst;     //! pointer to first entry in linked list
   TObjLink  *fLast;      //! pointer to last entry in linked list
   TObjLink  *fCache;     //! cache to speedup sequential calling of Before() and After() functions
   Bool_t     fAscending; //! sorting order (when calling Sort() or for TSortedList)

   TObjLink          *LinkAt(Int_t idx) const;
   TObjLink          *FindLink(const TObject *obj, Int_t &idx) const;
   TObjLink         **DoSort(TObjLink **head, Int_t n);
   Bool_t             LnkCompare(TObjLink *l1, TObjLink *l2);
   virtual TObjLink  *NewLink(TObject *obj, TObjLink *prev = NULL);
   virtual TObjLink  *NewOptLink(TObject *obj, Option_t *opt, TObjLink *prev = NULL);
   virtual void       DeleteLink(TObjLink *lnk);

private:
   TList(const TList&);             // not implemented
   TList& operator=(const TList&);  // not implemented

public:
   typedef TListIter Iterator_t;

   TList() : fFirst(0), fLast(0), fCache(0), fAscending(kTRUE) { }
   TList(TObject *) : fFirst(0), fLast(0), fCache(0), fAscending(kTRUE) { } // for backward compatibility, don't use
   virtual           ~TList();
   virtual void      Clear(Option_t *option="");
   virtual void      Delete(Option_t *option="");
   virtual TObject  *FindObject(const char *name) const;
   virtual TObject  *FindObject(const TObject *obj) const;
   virtual TIterator *MakeIterator(Bool_t dir = kIterForward) const;

   virtual void      Add(TObject *obj) { AddLast(obj); }
   virtual void      Add(TObject *obj, Option_t *opt) { AddLast(obj, opt); }
   virtual void      AddFirst(TObject *obj);
   virtual void      AddFirst(TObject *obj, Option_t *opt);
   virtual void      AddLast(TObject *obj);
   virtual void      AddLast(TObject *obj, Option_t *opt);
   virtual void      AddAt(TObject *obj, Int_t idx);
   virtual void      AddAfter(const TObject *after, TObject *obj);
   virtual void      AddAfter(TObjLink *after, TObject *obj);
   virtual void      AddBefore(const TObject *before, TObject *obj);
   virtual void      AddBefore(TObjLink *before, TObject *obj);
   virtual TObject  *Remove(TObject *obj);
   virtual TObject  *Remove(TObjLink *lnk);
   virtual void      RemoveLast();
   virtual void      RecursiveRemove(TObject *obj);

   virtual TObject  *At(Int_t idx) const;
   virtual TObject  *After(const TObject *obj) const;
   virtual TObject  *Before(const TObject *obj) const;
   virtual TObject  *First() const;
   virtual TObjLink *FirstLink() const { return fFirst; }
   virtual TObject **GetObjectRef(const TObject *obj) const;
   virtual TObject  *Last() const;
   virtual TObjLink *LastLink() const { return fLast; }

   virtual void      Sort(Bool_t order = kSortAscending);
   Bool_t            IsAscending() { return fAscending; }

   ClassDef(TList,5)  //Doubly linked list
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjLink                                                             //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TObjLink {

friend  class TList;

private:
   TObjLink   *fNext;
   TObjLink   *fPrev;
   TObject    *fObject;

   TObjLink(const TObjLink&);            // not implemented
   TObjLink& operator=(const TObjLink&); // not implemented

protected:
   TObjLink() : fNext(NULL), fPrev(NULL), fObject(NULL) { fNext = fPrev = this; }

public:
   TObjLink(TObject *obj) : fNext(NULL), fPrev(NULL), fObject(obj) { }
   TObjLink(TObject *obj, TObjLink *lnk);
   virtual ~TObjLink() { }

   TObject                *GetObject() const { return fObject; }
   TObject               **GetObjectRef() { return &fObject; }
   void                    SetObject(TObject *obj) { fObject = obj; }
   virtual Option_t       *GetAddOption() const { return ""; }
   virtual Option_t       *GetOption() const { return fObject->GetOption(); }
   virtual void            SetOption(Option_t *) { }
   TObjLink               *Next() { return fNext; }
   TObjLink               *Prev() { return fPrev; }
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjOptLink                                                          //
//                                                                      //
// Wrapper around a TObject so it can be stored in a TList including    //
// an option string.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TObjOptLink : public TObjLink {

private:
   TString   fOption;

public:
   TObjOptLink(TObject *obj, Option_t *opt) : TObjLink(obj), fOption(opt) { }
   TObjOptLink(TObject *obj, TObjLink *lnk, Option_t *opt) : TObjLink(obj, lnk), fOption(opt) { }
   ~TObjOptLink() { }
   Option_t        *GetAddOption() const { return fOption.Data(); }
   Option_t        *GetOption() const { return fOption.Data(); }
   void             SetOption(Option_t *option) { fOption = option; }
};


// Preventing warnings with -Weffc++ in GCC since it is a false positive for the TListIter destructor.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListIter                                                            //
//                                                                      //
// Iterator of linked list.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TListIter : public TIterator,
                  public std::iterator<std::bidirectional_iterator_tag,
                                       TObject*, std::ptrdiff_t,
                                       const TObject**, const TObject*&> {

protected:
   const TList       *fList;         //list being iterated
   TObjLink          *fCurCursor;    //current position in list
   TObjLink          *fCursor;       //next position in list
   Bool_t             fDirection;    //iteration direction
   Bool_t             fStarted;      //iteration started

   TListIter() : fList(0), fCurCursor(0), fCursor(0), fDirection(kIterForward),
                 fStarted(kFALSE) { }

public:
   TListIter(const TList *l, Bool_t dir = kIterForward);
   TListIter(const TListIter &iter);
   ~TListIter() { }
   TIterator &operator=(const TIterator &rhs);
   TListIter &operator=(const TListIter &rhs);

   const TCollection *GetCollection() const { return fList; }
   Option_t          *GetOption() const;
   void               SetOption(Option_t *option);
   TObject           *Next();
   void               Reset();
   Bool_t             operator!=(const TIterator &aIter) const;
   Bool_t             operator!=(const TListIter &aIter) const;
   TObject           *operator*() const { return (fCurCursor ? fCurCursor->GetObject() : nullptr); }

   ClassDef(TListIter,0)  //Linked list iterator
};

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif

#endif
