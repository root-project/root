// @(#)root/cont
// Author: Bianca-Cristina Cristescu February 2014

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfEnumsWithLock                                                         //
//                                                                      //
// A collection of TEnum objects designed for fast access given a       //
// DeclId_t and for keep track of TEnum that were described             //
// unloaded enum.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <forward_list>

#include "TListOfEnumsWithLock.h"
#include "TClass.h"
#include "TExMap.h"
#include "TEnum.h"
#include "TGlobal.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

ClassImp(TListOfEnumsWithLock)

//______________________________________________________________________________
TListOfEnumsWithLock::TListOfEnumsWithLock(TClass *cl /*=0*/) :
TListOfEnums(cl)
{
}

//______________________________________________________________________________
TListOfEnumsWithLock::~TListOfEnumsWithLock()
{
   // Destructor.

}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddFirst(TObject *obj)
{
   // Add object at the beginning of the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddFirst(obj);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddFirst(TObject *obj, Option_t *opt)
{
   // Add object at the beginning of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddFirst(obj, opt);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddLast(TObject *obj)
{
   // Add object at the end of the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddLast(obj);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddLast(TObject *obj, Option_t *opt)
{
   // Add object at the end of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddLast(obj, opt);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddAt(TObject *obj, Int_t idx)
{
   // Insert object at location idx in the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddAt(obj, idx);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddAfter(const TObject *after, TObject *obj)
{
   // Insert object after object after in the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddAfter(after, obj);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddAfter(TObjLink *after, TObject *obj)
{
   // Insert object after object after in the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddAfter(after, obj);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddBefore(const TObject *before, TObject *obj)
{
   // Insert object before object before in the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddBefore(before, obj);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::AddBefore(TObjLink *before, TObject *obj)
{
   // Insert object before object before in the list.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddBefore(before, obj);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::Clear(Option_t *option)
{
   // Remove all objects from the list. Does not delete the objects unless
   // the THashList is the owner (set via SetOwner()).

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::Clear(option);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::Delete(Option_t *option /* ="" */)
{
   // Delete all TDataMember object files.

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::Delete(option);
}

//______________________________________________________________________________
TObject *TListOfEnumsWithLock::FindObject(const char *name) const
{
   // Specialize FindObject to do search for the
   // a enum just by name or create it if its not already in the list

   R__LOCKGUARD(gInterpreterMutex);
   TObject *result = TListOfEnums::FindObject(name);
   if (!result) {


      TInterpreter::DeclId_t decl;
      if (GetClass()) decl = gInterpreter->GetEnum(GetClass(), name);
      else        decl = gInterpreter->GetEnum(0, name);
      if (decl) result = const_cast<TListOfEnumsWithLock *>(this)->Get(decl, name);
   }
   return result;
}


//______________________________________________________________________________
TObject* TListOfEnumsWithLock::FindObject(const TObject* obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::FindObject(obj);
}

//______________________________________________________________________________
TEnum *TListOfEnumsWithLock::GetObject(const char *name) const
{
   // Return an object from the list of enums *if and only if* is has already
   // been loaded in the list.  This is an internal routine.

   R__LOCKGUARD(gInterpreterMutex);
   return (TEnum*)THashList::FindObject(name);
}

//______________________________________________________________________________
void TListOfEnumsWithLock::RecursiveRemove(TObject *obj)
{
   // Remove object from this collection and recursively remove the object
   // from all other objects (and collections).
   // This function overrides TCollection::RecursiveRemove that calls
   // the Remove function. THashList::Remove cannot be called because
   // it uses the hash value of the hash table. This hash value
   // is not available anymore when RecursiveRemove is called from
   // the TObject destructor.

   if (!obj) return;

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::RecursiveRemove(obj);
}

//______________________________________________________________________________
TObject *TListOfEnumsWithLock::Remove(TObject *obj)
{
   // Remove object from the list.

   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Remove(obj);
}

//______________________________________________________________________________
TObject *TListOfEnumsWithLock::Remove(TObjLink *lnk)
{
   // Remove object via its objlink from the list.

   if (!lnk) return 0;

   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Remove(lnk);
}

//______________________________________________________________________________
TIterator* TListOfEnumsWithLock::MakeIterator(Bool_t dir ) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return new TListOfEnumsWithLockIter(this,dir);
}

//______________________________________________________________________________
TObject* TListOfEnumsWithLock::At(Int_t idx) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::At(idx);
}

//______________________________________________________________________________
TObject* TListOfEnumsWithLock::After(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::After(obj);
}

//______________________________________________________________________________
TObject* TListOfEnumsWithLock::Before(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Before(obj);
}

//______________________________________________________________________________
TObject* TListOfEnumsWithLock::First() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::First();
}

//______________________________________________________________________________
TObjLink* TListOfEnumsWithLock::FirstLink() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::FirstLink();
}

//______________________________________________________________________________
TObject** TListOfEnumsWithLock::GetObjectRef(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::GetObjectRef(obj);
}

//______________________________________________________________________________
TObject* TListOfEnumsWithLock::Last() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Last();
}

//______________________________________________________________________________
TObjLink* TListOfEnumsWithLock::LastLink() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::LastLink();
}


//______________________________________________________________________________
Int_t TListOfEnumsWithLock::GetLast() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::GetLast();
}

//______________________________________________________________________________
Int_t TListOfEnumsWithLock::IndexOf(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::IndexOf(obj);
}


//______________________________________________________________________________
Int_t TListOfEnumsWithLock::GetSize() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::GetSize();
}

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfEnumsWithLockIter                                                 //
//                                                                      //
// Iterator for TListOfEnumsWithLock.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TListOfEnumsWithLockIter)

//______________________________________________________________________________
TListOfEnumsWithLockIter::TListOfEnumsWithLockIter(const TListOfEnumsWithLock *l, Bool_t dir ):
TListIter(l,dir) {}

//______________________________________________________________________________
TObject *TListOfEnumsWithLockIter::Next()
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListIter::Next();
}
