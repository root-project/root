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
// TViewPubFunctions                                                    //
//                                                                      //
// View implementing the TList interface and giving access all the      //
// TFunction describing public methods in a class and all its base      //
// classes without caching any of the TFunction pointers.               //
//                                                                      //
// Adding to this collection directly is prohibited.                    //
// Iteration can only be done via the TIterator interfaces.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TViewPubFunctions.h"

#include "TClass.h"
#include "TBaseClass.h"
#include "TError.h"
#include "TFunction.h"
#include "THashList.h"

// ClassImp(TViewPubFunctions)

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
TViewPubFunctions::TViewPubFunctions(TClass *cl /* = 0 */)
{
   // Usual constructor

   if (cl) {
      fClasses.Add(cl);
      AddBasesClasses(fClasses,cl);
   }
}

//______________________________________________________________________________
TViewPubFunctions::~TViewPubFunctions()
{
   // Default destructor.

}

//______________________________________________________________________________
void TViewPubFunctions::Clear(Option_t * /* option="" */)
{
   // Clear is not allowed in this class.
   // See TList::Clear for the intended behavior.

   ::Error("TViewPubFunctions::Clear","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::Delete(Option_t * /*option="" */)
{
   // Delete is not allowed in this class.
   // See TList::Delete for the intended behavior.

   ::Error("TViewPubFunctions::Delete","Operation not allowed on a view.");
}

//______________________________________________________________________________
TObject *TViewPubFunctions::FindObject(const char * name) const
{
   // Find an object in this list using its name. Requires a sequential
   // scan till the object has been found. Returns 0 if object with specified
   // name is not found.

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

//______________________________________________________________________________
TObject *TViewPubFunctions::FindObject(const TObject * obj) const
{
   // Find an object in this list using the object's IsEqual()
   // member function. Requires a sequential scan till the object has
   // been found. Returns 0 if object is not found.

   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TObject *result = cl->GetListOfMethods(kFALSE)->FindObject(obj);
      if (result) return result;
   }
   return 0;
}

//______________________________________________________________________________
TIterator *TViewPubFunctions::MakeIterator(Bool_t dir /* = kIterForward*/) const
{
   // Return a list iterator.

   return new TViewPubFunctionsIter(this, dir);
}

//______________________________________________________________________________
void TViewPubFunctions::AddFirst(TObject * /* obj */)
{
   // AddFirst is not allowed in this class.
   // See TList::AddFirst for the intended behavior.

   ::Error("TViewPubFunctions::AddFirst","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddFirst(TObject * /* obj */, Option_t * /* opt */)
{
   // AddFirst is not allowed in this class.
   // See TList::AddFirst for the intended behavior.

   ::Error("TViewPubFunctions::AddFirst","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddLast(TObject * /* obj */)
{
   // AddLast is not allowed in this class.
   // See TList::AddLast for the intended behavior.

   ::Error("TViewPubFunctions::AddLast","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddLast(TObject * /* obj */, Option_t * /* opt */)
{
   // AddLast is not allowed in this class.
   // See TList::AddLast for the intended behavior.

   ::Error("TViewPubFunctions::AddLast","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddAt(TObject * /* obj */, Int_t /* idx */)
{
   // AddAt is not allowed in this class.
   // See TList::AddAt for the intended behavior.

   ::Error("TViewPubFunctions::AddAt","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddAfter(const TObject * /* after */, TObject * /* obj */)
{
   // AddAfter is not allowed in this class.
   // See TList::AddAfter for the intended behavior.

   ::Error("TViewPubFunctions::RemAddLastove","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddAfter(TObjLink * /* after */, TObject * /* obj */)
{
   // AddAfter is not allowed in this class.
   // See TList::AddAfter for the intended behavior.

   ::Error("TViewPubFunctions::AddAfter","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddBefore(const TObject * /* before */, TObject * /* obj */)
{
   // AddBefore is not allowed in this class.
   // See TList::AddBefore for the intended behavior.

   ::Error("TViewPubFunctions::AddBefore","Operation not allowed on a view.");
}

//______________________________________________________________________________
void TViewPubFunctions::AddBefore(TObjLink * /* before */, TObject * /* obj */)
{
   // AddBefore is not allowed in this class.
   // See TList::AddBefore for the intended behavior.

   ::Error("TViewPubFunctions::AddBefore","Operation not allowed on a view.");
}

//______________________________________________________________________________
TObject  *TViewPubFunctions::At(Int_t idx) const
{
   // Returns the object at position idx. Returns 0 if idx is out of range.

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

//______________________________________________________________________________
TObject  *TViewPubFunctions::After(const TObject * /* obj */) const
{
   // After is not allowed in this class.
   // See TList::After for the intended behavior.

   ::Error("TViewPubFunctions::After","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject  *TViewPubFunctions::Before(const TObject * /* obj */) const
{
   // Before is not allowed in this class.
   // See TList::Before for the intended behavior.

   ::Error("TViewPubFunctions::Before","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject  *TViewPubFunctions::First() const
{
   // First is not allowed in this class.
   // See TList::First for the intended behavior.

   ::Error("TViewPubFunctions::First","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObjLink *TViewPubFunctions::FirstLink() const
{
   // FirstLink is not allowed in this class.
   // See TList::FirstLink for the intended behavior.

   ::Error("TViewPubFunctions::FirstLink","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject **TViewPubFunctions::GetObjectRef(const TObject * /* obj */) const
{
   // GetObjectRef is not allowed in this class.
   // See TList::GetObjectRef for the intended behavior.

   ::Error("TViewPubFunctions::GetObjectRef","Operation not yet allowed on a view.");
   return 0;
}

//______________________________________________________________________________
Int_t TViewPubFunctions::GetSize() const
{
   // Return the total number of public methods (currently loaded in the list
   // of functions) in this class and all its base classes.

   Int_t size = 0;
   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      TIter funcnext(cl->GetListOfMethods(kFALSE));
      while (TFunction *p = (TFunction*) funcnext())
         if (p->Property() & kIsPublic) ++size;
   }
   return size;

}

//______________________________________________________________________________
void TViewPubFunctions::Load()
{
   // Load all the functions known to the intepreter for the scope 'fClass'
   // and all its bases classes.

   TIter next(&fClasses);
   while (TClass *cl = (TClass*)next()) {
      cl->GetListOfMethods(kTRUE);
   }
}

//______________________________________________________________________________
TObject  *TViewPubFunctions::Last() const
{
   // Last is not allowed in this class.
   // See TList::Last for the intended behavior.

   ::Error("TViewPubFunctions::Last","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObjLink *TViewPubFunctions::LastLink() const
{
   // LastLink is not allowed in this class.
   // See TList::LastLink for the intended behavior.

   ::Error("TViewPubFunctions::LastLink","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
void TViewPubFunctions::RecursiveRemove(TObject * /* obj */)
{
   // RecursiveRemove is not allowed in this class.
   // See TList::RecursiveRemove for the intended behavior.

   ::Error("TViewPubFunctions::RecursiveRemove","Operation not allowed on a view.");
}

//______________________________________________________________________________
TObject   *TViewPubFunctions::Remove(TObject * /* obj */)
{
   // Remove is not allowed in this class.
   // See TList::Remove for the intended behavior.

   ::Error("TViewPubFunctions::Remove","Operation not allowed on a view.");
   return 0;
}

//______________________________________________________________________________
TObject   *TViewPubFunctions::Remove(TObjLink * /* lnk */)
{
   // Remove is not allowed in this class.
   // See TList::Remove for the intended behavior.

   ::Error("TViewPubFunctions::Remove","Operation not allowed on a view.");
   return 0;
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewPubFunctionsIter                                                //
//                                                                      //
// Iterator of over the view's content                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// ClassImp(TViewPubFunctionsIter)

//______________________________________________________________________________
TViewPubFunctionsIter::TViewPubFunctionsIter(const TViewPubFunctions *l, Bool_t dir)
: fView(l),fClassIter(l->GetListOfClasses(),dir), fFuncIter((TCollection *)0),
  fStarted(kFALSE), fDirection(dir)
{
   // Create a new list iterator. By default the iteration direction
   // is kIterForward. To go backward use kIterBackward.
}

//______________________________________________________________________________
TViewPubFunctionsIter::TViewPubFunctionsIter(const TViewPubFunctionsIter &iter) :
   TIterator(iter), fView(iter.fView),
   fClassIter(iter.fClassIter), fFuncIter(iter.fFuncIter),
   fStarted(iter.fStarted), fDirection(iter.fDirection)
{
   // Copy ctor.

}

//______________________________________________________________________________
TIterator &TViewPubFunctionsIter::operator=(const TIterator &rhs)
{
   // Overridden assignment operator.

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

//______________________________________________________________________________
TViewPubFunctionsIter &TViewPubFunctionsIter::operator=(const TViewPubFunctionsIter &rhs)
{
   // Overloaded assignment operator.

   if (this != &rhs) {
      fView      = rhs.fView;
      fClassIter = rhs.fClassIter;
      fFuncIter  = rhs.fFuncIter;
      fStarted   = rhs.fStarted;
      fDirection = rhs.fDirection;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TViewPubFunctionsIter::Next()
{
   // Return next object in the list. Returns 0 when no more objects in list.

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
   // Not reacheable.
   return 0;
}

//______________________________________________________________________________
void TViewPubFunctionsIter::Reset()
{
   // Reset list iterator.

   fStarted = kFALSE;
   fClassIter.Reset();
}

//______________________________________________________________________________
Bool_t TViewPubFunctionsIter::operator!=(const TIterator &aIter) const
{
   // This operator compares two TIterator objects.

   const TViewPubFunctionsIter *iter = dynamic_cast<const TViewPubFunctionsIter*>(&aIter);
   if (iter) {
      return (fClassIter != iter->fClassIter || fFuncIter != iter->fFuncIter);
   }
   return false; // for base class we don't implement a comparison
}

//______________________________________________________________________________
Bool_t TViewPubFunctionsIter::operator!=(const TViewPubFunctionsIter &aIter) const
{
   // This operator compares two TViewPubFunctionsIter objects.

   return (fClassIter != aIter.fClassIter || fFuncIter != aIter.fFuncIter);
}

