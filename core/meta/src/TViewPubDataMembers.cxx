// @(#)root/cont:$Id$
// Author: Philippe Canal October 2013

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TViewPubDataMembers
View implementing the TList interface and giving access all the
TDictionary describing public data members in a class and all its
base classes without caching any of the TDictionary pointers.

Adding to this collection directly is prohibited.
Iteration can only be done via the TIterator interfaces.
*/

#include "TViewPubDataMembers.h"

#include "TClass.h"
#include "TBaseClass.h"
#include "TError.h"
#include "TDictionary.h"
#include "THashList.h"

// ClassImp(TViewPubDataMembers);

////////////////////////////////////////////////////////////////////////////////
/// loop over all base classes and add them to the container.

static void AddBasesClasses(TList &bases, TClass *cl)
{
   TIter nextBaseClass(cl->GetListOfBases());
   TBaseClass *base;
   while ((base = (TBaseClass*) nextBaseClass())) {
      if (!base->GetClassPointer()) continue;
      if (!(base->Property() & kIsPublic)) continue;

      bases.Add(base->GetClassPointer());
      AddBasesClasses(bases,base->GetClassPointer());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Usual constructor

TViewPubDataMembers::TViewPubDataMembers(TClass *cl /* = 0 */)
{
   if (cl) {
      fClasses.Add(cl);
      AddBasesClasses(fClasses,cl);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor.

TViewPubDataMembers::~TViewPubDataMembers()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Clear is not allowed in this class.
/// See TList::Clear for the intended behavior.

void TViewPubDataMembers::Clear(Option_t * /* option="" */)
{
   ::Error("TViewPubDataMembers::Clear","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Delete is not allowed in this class.
/// See TList::Delete for the intended behavior.

void TViewPubDataMembers::Delete(Option_t * /*option="" */)
{
   ::Error("TViewPubDataMembers::Delete","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Find an object in this list using its name. Requires a sequential
/// scan till the object has been found. Returns 0 if object with specified
/// name is not found.

TObject *TViewPubDataMembers::FindObject(const char * name) const
{
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      THashList *hl = dynamic_cast<THashList*>(cl->GetListOfDataMembers(kFALSE));
      TIter content_next(hl->GetListForObject(name));
      while (TDictionary *p = (TDictionary*) content_next()) {
         // The 'ListForObject' is actually a hash table bucket that can also
         // contain other element/name.
         if (strcmp(name,p->GetName())==0 && (p->Property() & kIsPublic))
            return p;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find an object in this list using the object's IsEqual()
/// member function. Requires a sequential scan till the object has
/// been found. Returns 0 if object is not found.

TObject *TViewPubDataMembers::FindObject(const TObject * obj) const
{
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TObject *result = cl->GetListOfDataMembers(kFALSE)->FindObject(obj);
      if (result) return result;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a list iterator.

TIterator *TViewPubDataMembers::MakeIterator(Bool_t dir /* = kIterForward*/) const
{
   return new TViewPubDataMembersIter(this, dir);
}

////////////////////////////////////////////////////////////////////////////////
/// AddFirst is not allowed in this class.
/// See TList::AddFirst for the intended behavior.

void TViewPubDataMembers::AddFirst(TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::AddFirst","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddFirst is not allowed in this class.
/// See TList::AddFirst for the intended behavior.

void TViewPubDataMembers::AddFirst(TObject * /* obj */, Option_t * /* opt */)
{
   ::Error("TViewPubDataMembers::AddFirst","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddLast is not allowed in this class.
/// See TList::AddLast for the intended behavior.

void TViewPubDataMembers::AddLast(TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::AddLast","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddLast is not allowed in this class.
/// See TList::AddLast for the intended behavior.

void TViewPubDataMembers::AddLast(TObject * /* obj */, Option_t * /* opt */)
{
   ::Error("TViewPubDataMembers::AddLast","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddAt is not allowed in this class.
/// See TList::AddAt for the intended behavior.

void TViewPubDataMembers::AddAt(TObject * /* obj */, Int_t /* idx */)
{
   ::Error("TViewPubDataMembers::AddAt","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddAfter is not allowed in this class.
/// See TList::AddAfter for the intended behavior.

void TViewPubDataMembers::AddAfter(const TObject * /* after */, TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::RemAddLastove","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddAfter is not allowed in this class.
/// See TList::AddAfter for the intended behavior.

void TViewPubDataMembers::AddAfter(TObjLink * /* after */, TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::AddAfter","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddBefore is not allowed in this class.
/// See TList::AddBefore for the intended behavior.

void TViewPubDataMembers::AddBefore(const TObject * /* before */, TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::AddBefore","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddBefore is not allowed in this class.
/// See TList::AddBefore for the intended behavior.

void TViewPubDataMembers::AddBefore(TObjLink * /* before */, TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::AddBefore","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the object at position idx. Returns 0 if idx is out of range.

TObject  *TViewPubDataMembers::At(Int_t idx) const
{
   Int_t i = 0;
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TIter content_next(cl->GetListOfDataMembers(kFALSE));
      while (TDictionary *p = (TDictionary*) content_next()) {
         if (p->Property() & kIsPublic) {
            if (i == idx) return p;
            ++i;
         }
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// After is not allowed in this class.
/// See TList::After for the intended behavior.

TObject  *TViewPubDataMembers::After(const TObject * /* obj */) const
{
   ::Error("TViewPubDataMembers::After","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Before is not allowed in this class.
/// See TList::Before for the intended behavior.

TObject  *TViewPubDataMembers::Before(const TObject * /* obj */) const
{
   ::Error("TViewPubDataMembers::Before","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// First is not allowed in this class.
/// See TList::First for the intended behavior.

TObject  *TViewPubDataMembers::First() const
{
   ::Error("TViewPubDataMembers::First","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// FirstLink is not allowed in this class.
/// See TList::FirstLink for the intended behavior.

TObjLink *TViewPubDataMembers::FirstLink() const
{
   ::Error("TViewPubDataMembers::FirstLink","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// GetObjectRef is not allowed in this class.
/// See TList::GetObjectRef for the intended behavior.

TObject **TViewPubDataMembers::GetObjectRef(const TObject * /* obj */) const
{
   ::Error("TViewPubDataMembers::GetObjectRef","Operation not yet allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the total number of public data members(currently loaded in the list
/// of DataMembers) in this class and all its base classes.

Int_t TViewPubDataMembers::GetSize() const
{
   Int_t size = 0;
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TIter content_next(cl->GetListOfDataMembers(kFALSE));
      while (TDictionary *p = (TDictionary*) content_next())
         if (p->Property() & kIsPublic) ++size;
   }
   return size;

}

////////////////////////////////////////////////////////////////////////////////
/// Load all the DataMembers known to the interpreter for the scope 'fClass'
/// and all its bases classes.

void TViewPubDataMembers::Load()
{
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      cl->GetListOfDataMembers(kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Last is not allowed in this class.
/// See TList::Last for the intended behavior.

TObject  *TViewPubDataMembers::Last() const
{
   ::Error("TViewPubDataMembers::Last","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// LastLink is not allowed in this class.
/// See TList::LastLink for the intended behavior.

TObjLink *TViewPubDataMembers::LastLink() const
{
   ::Error("TViewPubDataMembers::LastLink","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// RecursiveRemove is not allowed in this class.
/// See TList::RecursiveRemove for the intended behavior.

void TViewPubDataMembers::RecursiveRemove(TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::RecursiveRemove","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Remove is not allowed in this class.
/// See TList::Remove for the intended behavior.

TObject   *TViewPubDataMembers::Remove(TObject * /* obj */)
{
   ::Error("TViewPubDataMembers::Remove","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove is not allowed in this class.
/// See TList::Remove for the intended behavior.

TObject   *TViewPubDataMembers::Remove(TObjLink * /* lnk */)
{
   ::Error("TViewPubDataMembers::Remove","Operation not allowed on a view.");
   return 0;
}

/** \class TViewPubDataMembersIter
Iterator of over the view's content.
*/

// ClassImp(TViewPubDataMembersIter);

////////////////////////////////////////////////////////////////////////////////
/// Create a new list iterator. By default the iteration direction
/// is kIterForward. To go backward use kIterBackward.

TViewPubDataMembersIter::TViewPubDataMembersIter(const TViewPubDataMembers *l, Bool_t dir)
: fView(l),fClassIter(l->GetListOfClasses(),dir), fIter((TCollection *)0),
fStarted(kFALSE), fDirection(dir)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TViewPubDataMembersIter::TViewPubDataMembersIter(const TViewPubDataMembersIter &iter) :
TIterator(iter), fView(iter.fView),
fClassIter(iter.fClassIter), fIter(iter.fIter),
fStarted(iter.fStarted), fDirection(iter.fDirection)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Overridden assignment operator.

TIterator &TViewPubDataMembersIter::operator=(const TIterator &rhs)
{
   const TViewPubDataMembersIter *iter = dynamic_cast<const TViewPubDataMembersIter*>(&rhs);
   if (this != &rhs && iter) {
      fView      = iter->fView;
      fClassIter = iter->fClassIter;
      fIter  = iter->fIter;
      fStarted   = iter->fStarted;
      fDirection = iter->fDirection;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Overloaded assignment operator.

TViewPubDataMembersIter &TViewPubDataMembersIter::operator=(const TViewPubDataMembersIter &rhs)
{
   if (this != &rhs) {
      fView      = rhs.fView;
      fClassIter = rhs.fClassIter;
      fIter  = rhs.fIter;
      fStarted   = rhs.fStarted;
      fDirection = rhs.fDirection;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Return next object in the list. Returns 0 when no more objects in list.

TObject *TViewPubDataMembersIter::Next()
{
   if (!fView) return 0;

   if (!fStarted) {
      TClass *current = (TClass*)fClassIter();
      fStarted = kTRUE;
      if (current) {
         fIter.~TIter();
         new (&(fIter)) TIter(current->GetListOfDataMembers(kFALSE),fDirection);
      } else {
         return 0;
      }
   }

   while (1) {

      TDictionary *obj = (TDictionary *)fIter();
      if (!obj) {
         // End of list of DataMembers, move to the next;
         TClass *current = (TClass*)fClassIter();
         if (current) {
            fIter.~TIter();
            new (&(fIter)) TIter(current->GetListOfDataMembers(kFALSE),fDirection);
            continue;
         } else {
            return 0;
         }
      } else if (obj->Property() & kIsPublic) {
         // If it is public we found the next one.
         return obj;
      }

   }
   // Not reachable.
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset list iterator.

void TViewPubDataMembersIter::Reset()
{
   fStarted = kFALSE;
   fClassIter.Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// This operator compares two TIterator objects.

Bool_t TViewPubDataMembersIter::operator!=(const TIterator &aIter) const
{
   const TViewPubDataMembersIter *iter = dynamic_cast<const TViewPubDataMembersIter*>(&aIter);
   if (iter) {
      return (fClassIter != iter->fClassIter || fIter != iter->fIter);
   }
   return false; // for base class we don't implement a comparison
}

////////////////////////////////////////////////////////////////////////////////
/// This operator compares two TViewPubDataMembersIter objects.

Bool_t TViewPubDataMembersIter::operator!=(const TViewPubDataMembersIter &aIter) const
{
   return (fClassIter != aIter.fClassIter || fIter != aIter.fIter);
}

