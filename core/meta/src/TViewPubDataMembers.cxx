// @(#)root/cont:$Id$
// Author: Philippe Canal October 2013

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubDataMembers                                                  //
//                                                                      //
// View implementing the TList interface and giving access all the      //
// TDictionary describing public data members in a class and all its    //
// base classes without caching any of the TDictionary pointers.        //
//                                                                      //
// Adding to this collection directly is prohibited.                    //
// Iteration can only be done via the TIterator interfaces.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TViewPubDataMembers.h"

#include "TClass.h"
#include "TBaseClass.h"
#include "TError.h"
#include "TDictionary.h"
#include "THashList.h"

// ClassImp(TViewPubDataMembers)

//______________________________________________________________________________
static void AddBasesClasses(TList &bases, TClass *cl)
{
   // loop over all base classes and add them to the container.

   TIter nextBaseClass(cl->GetListOfBases());
   TBaseClass *base;
   while ((base = (TBaseClass*) nextBaseClass())) {
      if (!base->GetClassPointer()) continue;
      if (!(base->Property() & kIsPublic)) continue;

      bases.Add(base->GetClassPointer());
      AddBasesClasses(bases,base->GetClassPointer());
   }
}

//______________________________________________________________________________
TViewPubDataMembers::TViewPubDataMembers(TClass *cl /* = 0 */)
{
   // Usual constructor

   if (cl) {
      fClasses.Add(cl);
      AddBasesClasses(fClasses,cl);
   }
}

//______________________________________________________________________________
TViewPubDataMembers::~TViewPubDataMembers()
{
   // Default destructor.

}

//______________________________________________________________________________
void TViewPubDataMembers::Clear(Option_t * /* option="" */)
{
   // Clear is not allowed in this class.
   // See TList::Clear for the intended behavior.

   ::Error("TViewPubDataMembers::Clear","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::Delete(Option_t * /*option="" */)
{
   // Delete is not allowed in this class.
   // See TList::Delete for the intended behavior.

   ::Error("TViewPubDataMembers::Delete","Operation not allowed on a view.");
}

//______________________________________________________________________________
TObject *TViewPubDataMembers::FindObject(const char * name) const
{
   // Find an object in this list using its name. Requires a sequential
   // scan till the object has been found. Returns 0 if object with specified
   // name is not found.

   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      THashList *hl = dynamic_cast<THashList*>(cl->GetListOfDataMembers(kFALSE));
      TIter content_next(hl->GetListForObject(name));
      while (TDictionary *p = (TDictionary*) content_next())
         if (p->Property() & kIsPublic) return p;
   }
   return 0;
}

//______________________________________________________________________________
TObject *TViewPubDataMembers::FindObject(const TObject * obj) const
{
   // Find an object in this list using the object's IsEqual()
   // member function. Requires a sequential scan till the object has
   // been found. Returns 0 if object is not found.

   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TObject *result = cl->GetListOfDataMembers(kFALSE)->FindObject(obj);
      if (result) return result;
   }
   return 0;
}

//______________________________________________________________________________
TIterator *TViewPubDataMembers::MakeIterator(Bool_t dir /* = kIterForward*/) const
{
   // Return a list iterator.

   return new TViewPubDataMembersIter(this, dir);
}

//______________________________________________________________________________
void TViewPubDataMembers::AddFirst(TObject * /* obj */)
{
   // AddFirst is not allowed in this class.
   // See TList::AddFirst for the intended behavior.

   ::Error("TViewPubDataMembers::AddFirst","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddFirst(TObject * /* obj */, Option_t * /* opt */)
{
   // AddFirst is not allowed in this class.
   // See TList::AddFirst for the intended behavior.

   ::Error("TViewPubDataMembers::AddFirst","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddLast(TObject * /* obj */)
{
   // AddLast is not allowed in this class.
   // See TList::AddLast for the intended behavior.

   ::Error("TViewPubDataMembers::AddLast","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddLast(TObject * /* obj */, Option_t * /* opt */)
{
   // AddLast is not allowed in this class.
   // See TList::AddLast for the intended behavior.

   ::Error("TViewPubDataMembers::AddLast","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddAt(TObject * /* obj */, Int_t /* idx */)
{
   // AddAt is not allowed in this class.
   // See TList::AddAt for the intended behavior.

   ::Error("TViewPubDataMembers::AddAt","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddAfter(const TObject * /* after */, TObject * /* obj */)
{
   // AddAfter is not allowed in this class.
   // See TList::AddAfter for the intended behavior.

   ::Error("TViewPubDataMembers::RemAddLastove","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddAfter(TObjLink * /* after */, TObject * /* obj */)
{
   // AddAfter is not allowed in this class.
   // See TList::AddAfter for the intended behavior.

   ::Error("TViewPubDataMembers::AddAfter","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddBefore(const TObject * /* before */, TObject * /* obj */)
{
   // AddBefore is not allowed in this class.
   // See TList::AddBefore for the intended behavior.

   ::Error("TViewPubDataMembers::AddBefore","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubDataMembers::AddBefore(TObjLink * /* before */, TObject * /* obj */)
{
   // AddBefore is not allowed in this class.
   // See TList::AddBefore for the intended behavior.

   ::Error("TViewPubDataMembers::AddBefore","Operation not allowed on a view.");
}

//______________________________________________________________________________
TObject  *TViewPubDataMembers::At(Int_t idx) const
{
   // Returns the object at position idx. Returns 0 if idx is out of range.

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

//______________________________________________________________________________
TObject  *TViewPubDataMembers::After(const TObject * /* obj */) const
{
   // After is not allowed in this class.
   // See TList::After for the intended behavior.

   ::Error("TViewPubDataMembers::After","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject  *TViewPubDataMembers::Before(const TObject * /* obj */) const
{
   // Before is not allowed in this class.
   // See TList::Before for the intended behavior.

   ::Error("TViewPubDataMembers::Before","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject  *TViewPubDataMembers::First() const
{
   // First is not allowed in this class.
   // See TList::First for the intended behavior.

   ::Error("TViewPubDataMembers::First","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObjLink *TViewPubDataMembers::FirstLink() const
{
   // FirstLink is not allowed in this class.
   // See TList::FirstLink for the intended behavior.

   ::Error("TViewPubDataMembers::FirstLink","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject **TViewPubDataMembers::GetObjectRef(const TObject * /* obj */) const
{
   // GetObjectRef is not allowed in this class.
   // See TList::GetObjectRef for the intended behavior.

   ::Error("TViewPubDataMembers::GetObjectRef","Operation not yet allowed on a view.");
   return 0;
}

//______________________________________________________________________________
Int_t TViewPubDataMembers::GetSize() const
{
   // Return the total number of public data members(currently loaded in the list
   // of DataMembers) in this class and all its base classes.

   Int_t size = 0;
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TIter content_next(cl->GetListOfDataMembers(kFALSE));
      while (TDictionary *p = (TDictionary*) content_next())
         if (p->Property() & kIsPublic) ++size;
   }
   return size;

}

//______________________________________________________________________________
void TViewPubDataMembers::Load()
{
   // Load all the DataMembers known to the intepreter for the scope 'fClass'
   // and all its bases classes.

   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      cl->GetListOfDataMembers(kTRUE);
   }
}

//______________________________________________________________________________
TObject  *TViewPubDataMembers::Last() const
{
   // Last is not allowed in this class.
   // See TList::Last for the intended behavior.

   ::Error("TViewPubDataMembers::Last","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObjLink *TViewPubDataMembers::LastLink() const
{
   // LastLink is not allowed in this class.
   // See TList::LastLink for the intended behavior.

   ::Error("TViewPubDataMembers::LastLink","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
void TViewPubDataMembers::RecursiveRemove(TObject * /* obj */)
{
   // RecursiveRemove is not allowed in this class.
   // See TList::RecursiveRemove for the intended behavior.

   ::Error("TViewPubDataMembers::RecursiveRemove","Operation not allowed on a view.");
}

//______________________________________________________________________________
TObject   *TViewPubDataMembers::Remove(TObject * /* obj */)
{
   // Remove is not allowed in this class.
   // See TList::Remove for the intended behavior.

   ::Error("TViewPubDataMembers::Remove","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject   *TViewPubDataMembers::Remove(TObjLink * /* lnk */)
{
   // Remove is not allowed in this class.
   // See TList::Remove for the intended behavior.

   ::Error("TViewPubDataMembers::Remove","Operation not allowed on a view.");
   return 0;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubDataMembersIter                                                //
//                                                                      //
// Iterator of over the view's content                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// ClassImp(TViewPubDataMembersIter)

//______________________________________________________________________________
TViewPubDataMembersIter::TViewPubDataMembersIter(const TViewPubDataMembers *l, Bool_t dir)
: fView(l),fClassIter(l->GetListOfClasses(),dir), fIter((TCollection *)0),
fStarted(kFALSE), fDirection(dir)
{
   // Create a new list iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.
}

//______________________________________________________________________________
TViewPubDataMembersIter::TViewPubDataMembersIter(const TViewPubDataMembersIter &iter) :
TIterator(iter), fView(iter.fView),
fClassIter(iter.fClassIter), fIter(iter.fIter),
fStarted(iter.fStarted), fDirection(iter.fDirection)
{
   // Copy ctor.

}

//______________________________________________________________________________
TIterator &TViewPubDataMembersIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

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

//______________________________________________________________________________
TViewPubDataMembersIter &TViewPubDataMembersIter::operator=(const TViewPubDataMembersIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fView      = rhs.fView;
      fClassIter = rhs.fClassIter;
      fIter  = rhs.fIter;
      fStarted   = rhs.fStarted;
      fDirection = rhs.fDirection;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TViewPubDataMembersIter::Next()
{
   // Return next object in the list. Returns 0 when no more objects in list.

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
   // Not reacheable.
   return 0;
}

//______________________________________________________________________________
void TViewPubDataMembersIter::Reset()
{
   // Reset list iterator.

   fStarted = kFALSE;
   fClassIter.Reset();
}

//______________________________________________________________________________
Bool_t TViewPubDataMembersIter::operator!=(const TIterator &aIter) const
{
   // This operator compares two TIterator objects.

   const TViewPubDataMembersIter *iter = dynamic_cast<const TViewPubDataMembersIter*>(&aIter);
   if (iter) {
      return (fClassIter != iter->fClassIter || fIter != iter->fIter);
   }
   return false; // for base class we don't implement a comparison
}

//______________________________________________________________________________
Bool_t TViewPubDataMembersIter::operator!=(const TViewPubDataMembersIter &aIter) const
{
   // This operator compares two TViewPubDataMembersIter objects.

   return (fClassIter != aIter.fClassIter || fIter != aIter.fIter);
}

