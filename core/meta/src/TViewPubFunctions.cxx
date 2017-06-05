// @(#)root/cont:$Id$
// Author: Philippe Canal October 2013

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TViewPubFunctions
View implementing the TList interface and giving access all the
TFunction describing public methods in a class and all its base
classes without caching any of the TFunction pointers.

Adding to this collection directly is prohibited.
Iteration can only be done via the TIterator interfaces.
*/

#include "TViewPubFunctions.h"

#include "TClass.h"
#include "TBaseClass.h"
#include "TError.h"
#include "TFunction.h"
#include "THashList.h"

// ClassImp(TViewPubFunctions);

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

TViewPubFunctions::TViewPubFunctions(TClass *cl /* = 0 */)
{
   if (cl) {
      fClasses.Add(cl);
      AddBasesClasses(fClasses,cl);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor.

TViewPubFunctions::~TViewPubFunctions()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Clear is not allowed in this class.
/// See TList::Clear for the intended behavior.

void TViewPubFunctions::Clear(Option_t * /* option="" */)
{
   ::Error("TViewPubFunctions::Clear","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Delete is not allowed in this class.
/// See TList::Delete for the intended behavior.

void TViewPubFunctions::Delete(Option_t * /*option="" */)
{
   ::Error("TViewPubFunctions::Delete","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Find an object in this list using its name. Requires a sequential
/// scan till the object has been found. Returns 0 if object with specified
/// name is not found.

TObject *TViewPubFunctions::FindObject(const char * name) const
{
   if (name==0 || name[0]==0) return 0;

   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      THashList *hl = dynamic_cast<THashList*>(cl->GetListOfMethods(kFALSE));
      TIter funcnext(hl->GetListForObject(name));
      while (TFunction *p = (TFunction*) funcnext())
         if (p->Property() & kIsPublic
             && strncmp(p->GetName(),name,strlen(p->GetName())) == 0)
            return p;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find an object in this list using the object's IsEqual()
/// member function. Requires a sequential scan till the object has
/// been found. Returns 0 if object is not found.

TObject *TViewPubFunctions::FindObject(const TObject * obj) const
{
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TObject *result = cl->GetListOfMethods(kFALSE)->FindObject(obj);
      if (result) return result;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a list iterator.

TIterator *TViewPubFunctions::MakeIterator(Bool_t dir /* = kIterForward*/) const
{
   return new TViewPubFunctionsIter(this, dir);
}

////////////////////////////////////////////////////////////////////////////////
/// AddFirst is not allowed in this class.
/// See TList::AddFirst for the intended behavior.

void TViewPubFunctions::AddFirst(TObject * /* obj */)
{
   ::Error("TViewPubFunctions::AddFirst","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddFirst is not allowed in this class.
/// See TList::AddFirst for the intended behavior.

void TViewPubFunctions::AddFirst(TObject * /* obj */, Option_t * /* opt */)
{
   ::Error("TViewPubFunctions::AddFirst","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddLast is not allowed in this class.
/// See TList::AddLast for the intended behavior.

void TViewPubFunctions::AddLast(TObject * /* obj */)
{
   ::Error("TViewPubFunctions::AddLast","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddLast is not allowed in this class.
/// See TList::AddLast for the intended behavior.

void TViewPubFunctions::AddLast(TObject * /* obj */, Option_t * /* opt */)
{
   ::Error("TViewPubFunctions::AddLast","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddAt is not allowed in this class.
/// See TList::AddAt for the intended behavior.

void TViewPubFunctions::AddAt(TObject * /* obj */, Int_t /* idx */)
{
   ::Error("TViewPubFunctions::AddAt","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddAfter is not allowed in this class.
/// See TList::AddAfter for the intended behavior.

void TViewPubFunctions::AddAfter(const TObject * /* after */, TObject * /* obj */)
{
   ::Error("TViewPubFunctions::RemAddLastove","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddAfter is not allowed in this class.
/// See TList::AddAfter for the intended behavior.

void TViewPubFunctions::AddAfter(TObjLink * /* after */, TObject * /* obj */)
{
   ::Error("TViewPubFunctions::AddAfter","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddBefore is not allowed in this class.
/// See TList::AddBefore for the intended behavior.

void TViewPubFunctions::AddBefore(const TObject * /* before */, TObject * /* obj */)
{
   ::Error("TViewPubFunctions::AddBefore","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// AddBefore is not allowed in this class.
/// See TList::AddBefore for the intended behavior.

void TViewPubFunctions::AddBefore(TObjLink * /* before */, TObject * /* obj */)
{
   ::Error("TViewPubFunctions::AddBefore","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the object at position idx. Returns 0 if idx is out of range.

TObject  *TViewPubFunctions::At(Int_t idx) const
{
   Int_t i = 0;
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TIter funcnext(cl->GetListOfMethods(kFALSE));
      while (TFunction *p = (TFunction*) funcnext()) {
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

TObject  *TViewPubFunctions::After(const TObject * /* obj */) const
{
   ::Error("TViewPubFunctions::After","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Before is not allowed in this class.
/// See TList::Before for the intended behavior.

TObject  *TViewPubFunctions::Before(const TObject * /* obj */) const
{
   ::Error("TViewPubFunctions::Before","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// First is not allowed in this class.
/// See TList::First for the intended behavior.

TObject  *TViewPubFunctions::First() const
{
   ::Error("TViewPubFunctions::First","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// FirstLink is not allowed in this class.
/// See TList::FirstLink for the intended behavior.

TObjLink *TViewPubFunctions::FirstLink() const
{
   ::Error("TViewPubFunctions::FirstLink","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// GetObjectRef is not allowed in this class.
/// See TList::GetObjectRef for the intended behavior.

TObject **TViewPubFunctions::GetObjectRef(const TObject * /* obj */) const
{
   ::Error("TViewPubFunctions::GetObjectRef","Operation not yet allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the total number of public methods (currently loaded in the list
/// of functions) in this class and all its base classes.

Int_t TViewPubFunctions::GetSize() const
{
   Int_t size = 0;
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TIter funcnext(cl->GetListOfMethods(kFALSE));
      while (TFunction *p = (TFunction*) funcnext())
         if (p->Property() & kIsPublic) ++size;
   }
   return size;

}

////////////////////////////////////////////////////////////////////////////////
/// Load all the functions known to the interpreter for the scope 'fClass'
/// and all its bases classes.

void TViewPubFunctions::Load()
{
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      cl->GetListOfMethods(kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Last is not allowed in this class.
/// See TList::Last for the intended behavior.

TObject  *TViewPubFunctions::Last() const
{
   ::Error("TViewPubFunctions::Last","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// LastLink is not allowed in this class.
/// See TList::LastLink for the intended behavior.

TObjLink *TViewPubFunctions::LastLink() const
{
   ::Error("TViewPubFunctions::LastLink","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// RecursiveRemove is not allowed in this class.
/// See TList::RecursiveRemove for the intended behavior.

void TViewPubFunctions::RecursiveRemove(TObject * /* obj */)
{
   ::Error("TViewPubFunctions::RecursiveRemove","Operation not allowed on a view.");
}

////////////////////////////////////////////////////////////////////////////////
/// Remove is not allowed in this class.
/// See TList::Remove for the intended behavior.

TObject   *TViewPubFunctions::Remove(TObject * /* obj */)
{
   ::Error("TViewPubFunctions::Remove","Operation not allowed on a view.");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove is not allowed in this class.
/// See TList::Remove for the intended behavior.

TObject   *TViewPubFunctions::Remove(TObjLink * /* lnk */)
{
   ::Error("TViewPubFunctions::Remove","Operation not allowed on a view.");
   return 0;
}

/** \class TViewPubFunctionsIter
Iterator of over the view's content
*/

// ClassImp(TViewPubFunctionsIter);

////////////////////////////////////////////////////////////////////////////////
/// Create a new list iterator. By default the iteration direction
/// is kIterForward. To go backward use kIterBackward.

TViewPubFunctionsIter::TViewPubFunctionsIter(const TViewPubFunctions *l, Bool_t dir)
: fView(l),fClassIter(l->GetListOfClasses(),dir), fFuncIter((TCollection *)0),
  fStarted(kFALSE), fDirection(dir)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TViewPubFunctionsIter::TViewPubFunctionsIter(const TViewPubFunctionsIter &iter) :
   TIterator(iter), fView(iter.fView),
   fClassIter(iter.fClassIter), fFuncIter(iter.fFuncIter),
   fStarted(iter.fStarted), fDirection(iter.fDirection)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Overridden assignment operator.

TIterator &TViewPubFunctionsIter::operator=(const TIterator &rhs)
{
   const TViewPubFunctionsIter *iter = dynamic_cast<const TViewPubFunctionsIter*>(&rhs);
   if (this != &rhs && iter) {
      fView      = iter->fView;
      fClassIter = iter->fClassIter;
      fFuncIter  = iter->fFuncIter;
      fStarted   = iter->fStarted;
      fDirection = iter->fDirection;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Overloaded assignment operator.

TViewPubFunctionsIter &TViewPubFunctionsIter::operator=(const TViewPubFunctionsIter &rhs)
{
   if (this != &rhs) {
      fView      = rhs.fView;
      fClassIter = rhs.fClassIter;
      fFuncIter  = rhs.fFuncIter;
      fStarted   = rhs.fStarted;
      fDirection = rhs.fDirection;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Return next object in the list. Returns 0 when no more objects in list.

TObject *TViewPubFunctionsIter::Next()
{
   if (!fView) return 0;

   if (!fStarted) {
      TClass *current = (TClass*)fClassIter();
      fStarted = kTRUE;
      if (current) {
         fFuncIter.~TIter();
         new (&(fFuncIter)) TIter(current->GetListOfMethods(kFALSE),fDirection);
      } else {
         return 0;
      }
   }

   while (1) {

      TFunction *func = (TFunction *)fFuncIter();
      if (!func) {
         // End of list of functions, move to the next;
         TClass *current = (TClass*)fClassIter();
         if (current) {
            fFuncIter.~TIter();
            new (&(fFuncIter)) TIter(current->GetListOfMethods(kFALSE),fDirection);
            continue;
         } else {
            return 0;
         }
      } else if (func->Property() & kIsPublic) {
         // If it is public we found the next one.
         return func;
      }

   }
   // Not reachable.
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset list iterator.

void TViewPubFunctionsIter::Reset()
{
   fStarted = kFALSE;
   fClassIter.Reset();
}

////////////////////////////////////////////////////////////////////////////////
/// This operator compares two TIterator objects.

Bool_t TViewPubFunctionsIter::operator!=(const TIterator &aIter) const
{
   const TViewPubFunctionsIter *iter = dynamic_cast<const TViewPubFunctionsIter*>(&aIter);
   if (iter) {
      return (fClassIter != iter->fClassIter || fFuncIter != iter->fFuncIter);
   }
   return false; // for base class we don't implement a comparison
}

////////////////////////////////////////////////////////////////////////////////
/// This operator compares two TViewPubFunctionsIter objects.

Bool_t TViewPubFunctionsIter::operator!=(const TViewPubFunctionsIter &aIter) const
{
   return (fClassIter != aIter.fClassIter || fFuncIter != aIter.fFuncIter);
}

