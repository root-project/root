// @(#)root/cont:$Name:  $:$Id: TList.h,v 1.5 2000/12/13 15:13:46 brun Exp $
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
   virtual TObjLink  *NewLink(TObject *obj, TObjLink *prev = 0);
   virtual TObjLink  *NewOptLink(TObject *obj, Option_t *opt, TObjLink *prev = 0);
   virtual void       DeleteLink(TObjLink *lnk);

public:
   TList() { fFirst = fLast = fCache = 0; }
   TList(TObject *) { fFirst = fLast = fCache = 0; } // for backward compatibility, don't use
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
   virtual void      AddAfter(TObject *after, TObject *obj);
   virtual void      AddAfter(TObjLink *after, TObject *obj);
   virtual void      AddBefore(TObject *before, TObject *obj);
   virtual void      AddBefore(TObjLink *before, TObject *obj);
   virtual TObject  *Remove(TObject *obj);
   virtual TObject  *Remove(TObjLink *lnk);

   virtual TObject  *At(Int_t idx) const;
   virtual TObject  *After(TObject *obj) const;
   virtual TObject  *Before(TObject *obj) const;
   virtual TObject  *First() const;
   virtual TObjLink *FirstLink() const { return fFirst; }
   virtual TObject  *Last() const;
   virtual TObjLink *LastLink() const { return fLast; }

   virtual void      Sort(Bool_t order = kSortAscending);
   Bool_t            IsAscending() { return fAscending; }

   ClassDef(TList,4)  //Doubly linked list
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

protected:
   TObjLink() { fNext = fPrev = this; fObject = 0; }

public:
   TObjLink(TObject *obj) : fNext(0), fPrev(0), fObject(obj) { }
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListIter                                                            //
//                                                                      //
// Iterator of linked list.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TListIter : public TIterator {

protected:
   const TList       *fList;         //list being iterated
   TObjLink          *fCurCursor;    //current position in list
   TObjLink          *fCursor;       //next position in list
   Bool_t             fDirection;    //iteration direction
   Bool_t             fStarted;      //iteration started

   TListIter() : fList(0), fCursor(0), fStarted(kFALSE) { }

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
   void               Reset() { fStarted = kFALSE; }

   ClassDef(TListIter,0)  //Linked list iterator
};

#endif
