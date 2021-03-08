// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TListOfFunctions
A collection of TFunction objects designed for fast access given a
DeclId_t and for keep track of TFunction that were described
unloaded function.
*/

#include "TListOfFunctions.h"
#include "TClass.h"
#include "TExMap.h"
#include "TFunction.h"
#include "TMethod.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

ClassImp(TListOfFunctions);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TListOfFunctions::TListOfFunctions(TClass *cl) : fClass(cl),fIds(nullptr),fUnloaded(nullptr),fLastLoadMarker(0)
{
   fIds = new TExMap;
   fUnloaded = new THashList;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TListOfFunctions::~TListOfFunctions()
{
   THashList::Delete();
   delete fIds;
   fUnloaded->Delete();
   delete fUnloaded;
}

////////////////////////////////////////////////////////////////////////////////
/// Add pair<id, object> to the map of functions and their ids.

void TListOfFunctions::MapObject(TObject *obj)
{
   TFunction *f = dynamic_cast<TFunction*>(obj);
   if (f) {
      fIds->Add((Long64_t)f->GetDeclId(),(Long64_t)f);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list.

void TListOfFunctions::AddFirst(TObject *obj)
{
   THashList::AddFirst(obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list and also store option.
/// Storing an option is useful when one wants to change the behaviour
/// of an object a little without having to create a complete new
/// copy of the object. This feature is used, for example, by the Draw()
/// method. It allows the same object to be drawn in different ways.

void TListOfFunctions::AddFirst(TObject *obj, Option_t *opt)
{
   THashList::AddFirst(obj,opt);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list.

void TListOfFunctions::AddLast(TObject *obj)
{
   THashList::AddLast(obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list and also store option.
/// Storing an option is useful when one wants to change the behaviour
/// of an object a little without having to create a complete new
/// copy of the object. This feature is used, for example, by the Draw()
/// method. It allows the same object to be drawn in different ways.

void TListOfFunctions::AddLast(TObject *obj, Option_t *opt)
{
   THashList::AddLast(obj, opt);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object at location idx in the list.

void TListOfFunctions::AddAt(TObject *obj, Int_t idx)
{
   THashList::AddAt(obj, idx);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void TListOfFunctions::AddAfter(const TObject *after, TObject *obj)
{
   THashList::AddAfter(after, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void TListOfFunctions::AddAfter(TObjLink *after, TObject *obj)
{
   THashList::AddAfter(after, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void TListOfFunctions::AddBefore(const TObject *before, TObject *obj)
{
   THashList::AddBefore(before, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void TListOfFunctions::AddBefore(TObjLink *before, TObject *obj)
{
   THashList::AddBefore(before, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from the list. Does not delete the objects unless
/// the THashList is the owner (set via SetOwner()).

void TListOfFunctions::Clear(Option_t *option)
{
   fUnloaded->Clear(option);
   fIds->Clear();
   THashList::Clear(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all TFunction object files.

void TListOfFunctions::Delete(Option_t *option /* ="" */)
{
   fUnloaded->Delete(option);
   fIds->Clear();
   THashList::Delete(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Specialize FindObject to do search for the
/// a function just by name or create it if its not already in the list

TObject *TListOfFunctions::FindObject(const char *name) const
{
   R__LOCKGUARD(gInterpreterMutex);
   TObject *result = THashList::FindObject(name);
   if (!result) {

      TInterpreter::DeclId_t decl;
      if (fClass) decl = gInterpreter->GetFunction(fClass->GetClassInfo(),name);
      else        decl = gInterpreter->GetFunction(nullptr,name);
      if (decl) result = const_cast<TListOfFunctions*>(this)->Get(decl);
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the set of overloads for this name, collecting all available ones.
/// Can construct and insert new TFunction-s.

TList* TListOfFunctions::GetListForObjectNonConst(const char* name)
{
   R__LOCKGUARD(gInterpreterMutex);

   TList* overloads = (TList*)fOverloads.FindObject(name);
   TExMap overloadsSet;
   Bool_t wasEmpty = true;
   if (!overloads) {
      overloads = new TList();
      overloads->SetName(name);
      fOverloads.Add(overloads);
   } else {
      TIter iOverload(overloads);
      while (TFunction* over = (TFunction*)iOverload()) {
         wasEmpty = false;
         overloadsSet.Add((Long64_t)(ULong64_t)over->GetDeclId(),
                          (Long64_t)(ULong64_t)over);
      }
   }

   // Update if needed.
   std::vector<DeclId_t> overloadDecls;
   ClassInfo_t* ci = fClass ? fClass->GetClassInfo() : nullptr;
   gInterpreter->GetFunctionOverloads(ci, name, overloadDecls);
   for (std::vector<DeclId_t>::const_iterator iD = overloadDecls.begin(),
           eD = overloadDecls.end(); iD != eD; ++iD) {
      TFunction* over = Get(*iD);
      if (wasEmpty || !overloadsSet.GetValue((Long64_t)(ULong64_t)over->GetDeclId())) {
          overloads->Add(over);
      }
   }

   return overloads;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the set of overloads for this name, collecting all available ones.
/// Can construct and insert new TFunction-s.

TList* TListOfFunctions::GetListForObject(const char* name) const
{
   return const_cast<TListOfFunctions*>(this)->GetListForObjectNonConst(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the set of overloads for function obj, collecting all available ones.
/// Can construct and insert new TFunction-s.

TList* TListOfFunctions::GetListForObject(const TObject* obj) const
{
   if (!obj) return nullptr;
   return const_cast<TListOfFunctions*>(this)
      ->GetListForObjectNonConst(obj->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Return the TMethod or TFunction describing the function corresponding
/// to the Decl 'id'. Return NULL if not found.

TFunction *TListOfFunctions::Find(DeclId_t id) const
{
   if (!id) return nullptr;

   R__LOCKGUARD(gInterpreterMutex);
   return (TFunction*)fIds->GetValue((Long64_t)id);
}

////////////////////////////////////////////////////////////////////////////////
/// Return (after creating it if necessary) the TMethod or TFunction
/// describing the function corresponding to the Decl 'id'.

TFunction *TListOfFunctions::Get(DeclId_t id)
{
   if (!id) return nullptr;

   R__LOCKGUARD(gInterpreterMutex);
   //need the Find and possible Add to be one atomic operation
   TFunction *f = Find(id);
   if (f) return f;

   if (fClass) {
      if (!gInterpreter->ClassInfo_Contains(fClass->GetClassInfo(),id)) return nullptr;
   } else {
      if (!gInterpreter->ClassInfo_Contains(nullptr,id)) return nullptr;
   }

   MethodInfo_t *m = gInterpreter->MethodInfo_Factory(id);

   // Let's see if this is a reload ...
   const char *name = gInterpreter->MethodInfo_Name(m);
   if (const TList* bucketForMethod = fUnloaded->GetListForObject(name)) {
      TString mangledName( gInterpreter->MethodInfo_GetMangledName(m) );
      TIter    next(bucketForMethod);
      TFunction *uf;
      while ((uf = (TFunction *) next())) {
         if (uf->GetMangledName() == mangledName) {
            // Reuse
            fUnloaded->Remove(uf);

            uf->Update(m);
            f = uf;
            break;
         }
      }
   }
   if (!f) {
      if (fClass) f = new TMethod(m, fClass);
      else f = new TFunction(m);
   }
   // Calling 'just' THahList::Add would turn around and call
   // TListOfFunctions::AddLast which should *also* do the fIds->Add.
   THashList::AddLast(f);
   fIds->Add((Long64_t)id,(Long64_t)f);

   return f;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a pair<id, object> from the map of functions and their ids.

void TListOfFunctions::UnmapObject(TObject *obj)
{
   TFunction *f = dynamic_cast<TFunction*>(obj);
   if (f) {
      fIds->Remove((Long64_t)f->GetDeclId());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from this collection and recursively remove the object
/// from all other objects (and collections).
/// This function overrides TCollection::RecursiveRemove that calls
/// the Remove function. THashList::Remove cannot be called because
/// it uses the hash value of the hash table. This hash value
/// is not available anymore when RecursiveRemove is called from
/// the TObject destructor.

void TListOfFunctions::RecursiveRemove(TObject *obj)
{
   if (!obj) return;

   THashList::RecursiveRemove(obj);
   fUnloaded->RecursiveRemove(obj);
   UnmapObject(obj);

}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from the list.

TObject* TListOfFunctions::Remove(TObject *obj)
{
   Bool_t found;

   found = THashList::Remove(obj);
   if (!found) {
      found = fUnloaded->Remove(obj);
   }
   UnmapObject(obj);
   if (found) return obj;
   else return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object via its objlink from the list.

TObject* TListOfFunctions::Remove(TObjLink *lnk)
{
   if (!lnk) return nullptr;

   TObject *obj = lnk->GetObject();

   THashList::Remove(lnk);
   fUnloaded->Remove(obj);

   UnmapObject(obj);
   return obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Load all the functions known to the interpreter for the scope 'fClass'
/// into this collection.

void TListOfFunctions::Load()
{
   if (fClass && fClass->GetClassInfo() == nullptr) return;

   R__LOCKGUARD(gInterpreterMutex);

   ULong64_t currentTransaction = gInterpreter->GetInterpreterStateMarker();
   if (currentTransaction == fLastLoadMarker) {
      return;
   }
   fLastLoadMarker = currentTransaction;

   ClassInfo_t *info;
   if (fClass) info = fClass->GetClassInfo();
   else info = gInterpreter->ClassInfo_Factory();

   MethodInfo_t *t = gInterpreter->MethodInfo_Factory(info);
   while (gInterpreter->MethodInfo_Next(t)) {
      if (gInterpreter->MethodInfo_IsValid(t)) {
         TDictionary::DeclId_t mid = gInterpreter->GetDeclId(t);
         // Get will check if there is already there or create a new one
         // (or re-use a previously unloaded version).
         Get(mid);
      }
   }
   gInterpreter->MethodInfo_Delete(t);
   if (!fClass) gInterpreter->ClassInfo_Delete(info);
}

////////////////////////////////////////////////////////////////////////////////
/// Mark 'all func' as being unloaded.
/// After the unload, the function can no longer be found directly,
/// until the decl can be found again in the interpreter (in which
/// the func object will be reused.

void TListOfFunctions::Unload()
{
   TObjLink *lnk = FirstLink();
   while (lnk) {
      TFunction *func = (TFunction*)lnk->GetObject();

      fIds->Remove((Long64_t)func->GetDeclId());
      fUnloaded->Add(func);

      lnk = lnk->Next();
   }

   THashList::Clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Mark 'func' as being unloaded.
/// After the unload, the function can no longer be found directly,
/// until the decl can be found again in the interpreter (in which
/// the func object will be reused.

void TListOfFunctions::Unload(TFunction *func)
{
   if (THashList::Remove(func)) {
      // We contains the object, let remove it from the other internal
      // list and move it to the list of unloaded objects.

      fIds->Remove((Long64_t)func->GetDeclId());
      fUnloaded->Add(func);
   }
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfFunctions::FindObject(const TObject* obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::FindObject(obj);
}

////////////////////////////////////////////////////////////////////////////////

TIterator* TListOfFunctions::MakeIterator(Bool_t dir ) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return new TListOfFunctionsIter(this,dir);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfFunctions::At(Int_t idx) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::At(idx);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfFunctions::After(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::After(obj);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfFunctions::Before(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::Before(obj);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfFunctions::First() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::First();
}

////////////////////////////////////////////////////////////////////////////////

TObjLink* TListOfFunctions::FirstLink() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::FirstLink();
}

////////////////////////////////////////////////////////////////////////////////

TObject** TListOfFunctions::GetObjectRef(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::GetObjectRef(obj);
}

////////////////////////////////////////////////////////////////////////////////

TObject* TListOfFunctions::Last() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::Last();
}

////////////////////////////////////////////////////////////////////////////////

TObjLink* TListOfFunctions::LastLink() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::LastLink();
}


////////////////////////////////////////////////////////////////////////////////

Int_t TListOfFunctions::GetLast() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::GetLast();
}

////////////////////////////////////////////////////////////////////////////////

Int_t TListOfFunctions::IndexOf(const TObject *obj) const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::IndexOf(obj);
}


////////////////////////////////////////////////////////////////////////////////

Int_t TListOfFunctions::GetSize() const
{
   R__LOCKGUARD(gInterpreterMutex);
   return THashList::GetSize();
}

/** \class TListOfFunctionsIter
Iterator for TListOfFunctions.
*/

ClassImp(TListOfFunctionsIter);

////////////////////////////////////////////////////////////////////////////////

TListOfFunctionsIter::TListOfFunctionsIter(const TListOfFunctions *l, Bool_t dir ):
  TListIter(l,dir) {}

////////////////////////////////////////////////////////////////////////////////

TObject *TListOfFunctionsIter::Next()
{
   R__LOCKGUARD(gInterpreterMutex);
   return TListIter::Next();
}

