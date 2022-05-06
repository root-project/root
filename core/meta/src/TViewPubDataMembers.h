// @(#)root/meta:$Id$
// Author: Philippe Canal October 2013

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TViewPubDataMembers
#define ROOT_TViewPubDataMembers

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubDataMembers                                                  //
//                                                                      //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"

class TClass;


class TViewPubDataMembers : public TList {

protected:
   TList   fClasses; // list of the all the (base) classes for which we list methods.

private:
   TViewPubDataMembers(const TViewPubDataMembers&) = delete;
   TViewPubDataMembers& operator=(const TViewPubDataMembers&) = delete;

public:
   TViewPubDataMembers(TClass *cl = nullptr);
   virtual    ~TViewPubDataMembers();

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
   ClassDefInlineOverride(TViewPubDataMembers, 0)
};

// Preventing warnings with -Weffc++ in GCC since it is a false positive for the TListIter destructor.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubDataMembersIter                                              //
//                                                                      //
// Iterator of view of linked list.      `                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TViewPubDataMembersIter : public TIterator {
public:
   using iterator_category = std::bidirectional_iterator_tag;
   using value_type = TObject *;
   using difference_type = std::ptrdiff_t;
   using pointer = const TObject **;
   using reference =  const TObject *&;

protected:
   const TList *fView;   //View we are iterating over.
   TIter        fClassIter;    //iterator over the classes
   TIter        fIter;         //iterator over the members of the current class
   Bool_t       fStarted;      //iteration started
   Bool_t       fDirection;    //iteration direction

   TViewPubDataMembersIter() : fView(nullptr), fClassIter((TCollection *)nullptr), fIter((TCollection *)nullptr), fStarted(kFALSE), fDirection(kIterForward) { }

public:
   TViewPubDataMembersIter(const TViewPubDataMembers *l, Bool_t dir = kIterForward);
   TViewPubDataMembersIter(const TViewPubDataMembersIter &iter);
   ~TViewPubDataMembersIter() { }
   TIterator &operator=(const TIterator &rhs) override;
   TViewPubDataMembersIter &operator=(const TViewPubDataMembersIter &rhs);

   const TCollection *GetCollection() const override { return fView; }
   TObject           *Next() override;
   void               Reset() override;
   Bool_t             operator!=(const TIterator &aIter) const override;
   Bool_t             operator!=(const TViewPubDataMembersIter &aIter) const;
   TObject           *operator*() const override { return *fIter; }

   // ClassDefInline does not yet support non default constructible classes
   //    ClassDefInline(TViewPubDataMembersIter,0)
};

#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif

#endif // ROOT_TViewPubDataMembers
