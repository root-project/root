// @(#)root/meta:$Id$
// Author: Philippe Canal October 2013

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TViewPubFunctions
#define ROOT_TViewPubFunctions

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubFunctions                                                    //
//                                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"

class TClass;


class TViewPubFunctions : public TList {

protected:
   TList   fClasses; // list of the all the (base) classes for which we list methods.

private:
   TViewPubFunctions(const TViewPubFunctions&) = delete;
   TViewPubFunctions& operator=(const TViewPubFunctions&) = delete;

public:
   TViewPubFunctions(TClass *cl = nullptr);
   virtual    ~TViewPubFunctions();

   TObject   *FindObject(const char *name) const override;
   TObject   *FindObject(const TObject *obj) const override;

   TObject  *At(Int_t idx) const override;
   virtual const TList *GetListOfClasses() const { return &fClasses; }
   Int_t        GetSize() const override;
   TIterator   *MakeIterator(Bool_t dir = kIterForward) const override;

   void       Load();

   // All the following routines are explicitly disallow/unsupported for
   // a view
protected:
   void       Clear(Option_t *option="") override;
   void       Delete(Option_t *option="") override;

   void       AddFirst(TObject *obj) override;
   void       AddFirst(TObject *obj, Option_t *opt) override;
   void       AddLast(TObject *obj) override;
   void       AddLast(TObject *obj, Option_t *opt) override;
   void       AddAt(TObject *obj, Int_t idx) override;
   void       AddAfter(const TObject *after, TObject *obj) override;
   void       AddAfter(TObjLink *after, TObject *obj) override;
   void       AddBefore(const TObject *before, TObject *obj) override;
   void       AddBefore(TObjLink *before, TObject *obj) override;

   TObject   *After(const TObject *obj) const override;
   TObject   *Before(const TObject *obj) const override;
   TObject   *First() const override;
   TObjLink  *FirstLink() const override;
   TObject  **GetObjectRef(const TObject *obj) const override;
   TObject   *Last() const override;
   TObjLink  *LastLink() const override;

   void       RecursiveRemove(TObject *obj) override;
   TObject   *Remove(TObject *obj) override;
   TObject   *Remove(TObjLink *lnk) override;

public:
   ClassDefInlineOverride(TViewPubFunctions, 0) // Doubly linked list with hashtable for lookup
};

// Preventing warnings with -Weffc++ in GCC since it is a false positive for the TListIter destructor.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubFunctionsIter                                                //
//                                                                      //
// Iterator of view of linked list.      `1234                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TViewPubFunctionsIter : public TIterator,
                              public std::iterator<std::bidirectional_iterator_tag,
                                                   TObject*, std::ptrdiff_t,
                                                    const TObject**, const TObject*&>
{
protected:
   const TList *fView;   //View we are iterating over.
   TIter        fClassIter;    //iterator over the classes
   TIter        fFuncIter;     //iterator over the method of the current class
   Bool_t       fStarted;      //iteration started
   Bool_t       fDirection;    //iteration direction

   TViewPubFunctionsIter() : fView(0), fClassIter((TCollection *)0), fFuncIter((TCollection *)0), fStarted(kFALSE), fDirection(kIterForward) { }

public:
   TViewPubFunctionsIter(const TViewPubFunctions *l, Bool_t dir = kIterForward);
   TViewPubFunctionsIter(const TViewPubFunctionsIter &iter);
   ~TViewPubFunctionsIter() { }
   TIterator &operator=(const TIterator &rhs) override;
   TViewPubFunctionsIter &operator=(const TViewPubFunctionsIter &rhs);

   const TCollection *GetCollection() const override { return fView; }
   TObject           *Next() override;
   void               Reset() override;
   Bool_t             operator!=(const TIterator &aIter) const override;
   Bool_t             operator!=(const TViewPubFunctionsIter &aIter) const;
   TObject           *operator*() const override { return *fFuncIter; }

   // ClassDefInline does not yet support non default constructible classes
   // ClassDefInline(TViewPubFunctionsIter,0)  //Linked list iterator
};

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif

#endif // ROOT_TViewPubFunctions
