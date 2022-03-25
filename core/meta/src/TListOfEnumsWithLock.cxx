// @(#)root/cont
// Author: Bianca-Cristina Cristescu February 2014

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TListOfEnumsWithLock
A collection of TEnum objects designed for fast access given a
DeclId_t and for keep track of TEnum that were described
unloaded enum.
*/

#include <forward_list>

#include "TListOfEnumsWithLock.h"
#include "TClass.h"
#include "TExMap.h"
#include "TEnum.h"
#include "TGlobal.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

ClassImp(TListOfEnumsWithLock);

////////////////////////////////////////////////////////////////////////////////

TListOfEnumsWithLock::TListOfEnumsWithLock(TClass *cl /*=0*/) :
TListOfEnums(cl)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TListOfEnumsWithLock::~TListOfEnumsWithLock()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list.

void TListOfEnumsWithLock::AddFirst(TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddFirst(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list and also store option.
/// Storing an option is useful when one wants to change the behaviour
/// of an object a little without having to create a complete new
/// copy of the object. This feature is used, for example, by the Draw()
/// method. It allows the same object to be drawn in different ways.

void TListOfEnumsWithLock::AddFirst(TObject *obj, Option_t *opt)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddFirst(obj, opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list.

void TListOfEnumsWithLock::AddLast(TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddLast(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list and also store option.
/// Storing an option is useful when one wants to change the behaviour
/// of an object a little without having to create a complete new
/// copy of the object. This feature is used, for example, by the Draw()
/// method. It allows the same object to be drawn in different ways.

void TListOfEnumsWithLock::AddLast(TObject *obj, Option_t *opt)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddLast(obj, opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object at location idx in the list.

void TListOfEnumsWithLock::AddAt(TObject *obj, Int_t idx)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddAt(obj, idx);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void TListOfEnumsWithLock::AddAfter(const TObject *after, TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddAfter(after, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void TListOfEnumsWithLock::AddAfter(TObjLink *after, TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddAfter(after, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void TListOfEnumsWithLock::AddBefore(const TObject *before, TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddBefore(before, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void TListOfEnumsWithLock::AddBefore(TObjLink *before, TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::AddBefore(before, obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from the list. Does not delete the objects unless
/// the THashList is the owner (set via SetOwner()).

void TListOfEnumsWithLock::Clear(Option_t *option)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::Clear(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all TDataMember object files.

void TListOfEnumsWithLock::Delete(Option_t *option /* ="" */)
{
   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::Delete(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Specialize FindObject to do search for the
/// a enum just by name or create it if its not already in the list

TObject *TListOfEnumsWithLock::FindObject(const char *name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::FindObject(name);
}


////////////////////////////////////////////////////////////////////////////////

TObject* TListOfEnumsWithLock::FindObject(const TObject* obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::FindObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Return an object from the list of enums *if and only if* is has already
/// been loaded in the list.  This is an internal routine.

TEnum *TListOfEnumsWithLock::GetObject(const char *name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return (TEnum*)THashList::FindObject(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from this collection and recursively remove the object
/// from all other objects (and collections).
/// This function overrides TCollection::RecursiveRemove that calls
/// the Remove function. THashList::Remove cannot be called because
/// it uses the hash value of the hash table. This hash value
/// is not available anymore when RecursiveRemove is called from
/// the TObject destructor.

void TListOfEnumsWithLock::RecursiveRemove(TObject *obj)
{
   if (!obj) return;

   R__LOCKGUARD(gInterpreterMutex);
   TListOfEnums::RecursiveRemove(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from the list.

TObject *TListOfEnumsWithLock::Remove(TObject *obj)
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Remove(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object via its objlink from the list.

TObject *TListOfEnumsWithLock::Remove(TObjLink *lnk)
{
   if (!lnk) return 0;

   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Remove(lnk);
}

////////////////////////////////////////////////////////////////////////////////

TIterator* TListOfEnumsWithLock::MakeIterator(Bool_t dir ) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return new TListOfEnumsWithLockIter(this,dir);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfEnumsWithLock::At(Int_t idx) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::At(idx);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfEnumsWithLock::After(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::After(obj);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfEnumsWithLock::Before(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Before(obj);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfEnumsWithLock::First() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::First();
}

////////////////////////////////////////////////////////////////////////////////

TObjLink* TListOfEnumsWithLock::FirstLink() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::FirstLink();
}

////////////////////////////////////////////////////////////////////////////////

TObject** TListOfEnumsWithLock::GetObjectRef(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::GetObjectRef(obj);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfEnumsWithLock::Last() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::Last();
}

////////////////////////////////////////////////////////////////////////////////

TObjLink* TListOfEnumsWithLock::LastLink() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::LastLink();
}


////////////////////////////////////////////////////////////////////////////////

Int_t TListOfEnumsWithLock::GetLast() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::GetLast();
}

////////////////////////////////////////////////////////////////////////////////

Int_t TListOfEnumsWithLock::IndexOf(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::IndexOf(obj);
}


////////////////////////////////////////////////////////////////////////////////

Int_t TListOfEnumsWithLock::GetSize() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListOfEnums::GetSize();
}

/** \class TListOfEnumsWithLockIter
Iterator for TListOfEnumsWithLock.
*/

ClassImp(TListOfEnumsWithLockIter);

////////////////////////////////////////////////////////////////////////////////

TListOfEnumsWithLockIter::TListOfEnumsWithLockIter(const TListOfEnumsWithLock *l, Bool_t dir ):
TListIter(l,dir) {}

////////////////////////////////////////////////////////////////////////////////

TObject *TListOfEnumsWithLockIter::Next()
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListIter::Next();
}
