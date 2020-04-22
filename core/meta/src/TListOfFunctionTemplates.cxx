// @(#)root/cont
// Author: Bianca-Cristina Cristescu March 2014

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TListOfFunctionTemplates
A collection of TFunction objects designed for fast access given a
DeclId_t and for keep track of TFunction that were described
unloaded function.
*/

#include "TListOfFunctionTemplates.h"
#include "TClass.h"
#include "TExMap.h"
#include "TFunction.h"
#include "TFunctionTemplate.h"
#include "TMethod.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

ClassImp(TListOfFunctionTemplates);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TListOfFunctionTemplates::TListOfFunctionTemplates(TClass *cl) : fClass(cl),fIds(0),
                          fUnloaded(0),fLastLoadMarker(0)
{
   fIds = new TExMap;
   fUnloaded = new THashList;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TListOfFunctionTemplates::~TListOfFunctionTemplates()
{
   THashList::Delete();
   delete fIds;
   fUnloaded->Delete();
   delete fUnloaded;
}

////////////////////////////////////////////////////////////////////////////////
/// Add pair<id, object> to the map of functions and their ids.

void TListOfFunctionTemplates::MapObject(TObject *obj)
{
   TFunctionTemplate *f = dynamic_cast<TFunctionTemplate*>(obj);
   if (f) {
      fIds->Add((Long64_t)f->GetDeclId(),(Long64_t)f);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the beginning of the list.

void TListOfFunctionTemplates::AddFirst(TObject *obj)
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

void TListOfFunctionTemplates::AddFirst(TObject *obj, Option_t *opt)
{
   THashList::AddFirst(obj,opt);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Add object at the end of the list.

void TListOfFunctionTemplates::AddLast(TObject *obj)
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

void TListOfFunctionTemplates::AddLast(TObject *obj, Option_t *opt)
{
   THashList::AddLast(obj, opt);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object at location idx in the list.

void TListOfFunctionTemplates::AddAt(TObject *obj, Int_t idx)
{
   THashList::AddAt(obj, idx);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void TListOfFunctionTemplates::AddAfter(const TObject *after, TObject *obj)
{
   THashList::AddAfter(after, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object after object after in the list.

void TListOfFunctionTemplates::AddAfter(TObjLink *after, TObject *obj)
{
   THashList::AddAfter(after, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void TListOfFunctionTemplates::AddBefore(const TObject *before, TObject *obj)
{
   THashList::AddBefore(before, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Insert object before object before in the list.

void TListOfFunctionTemplates::AddBefore(TObjLink *before, TObject *obj)
{
   THashList::AddBefore(before, obj);
   MapObject(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all objects from the list. Does not delete the objects unless
/// the THashList is the owner (set via SetOwner()).

void TListOfFunctionTemplates::Clear(Option_t *option)
{
   fUnloaded->Clear(option);
   fIds->Clear();
   THashList::Clear(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete all TFunction object files.

void TListOfFunctionTemplates::Delete(Option_t *option /* ="" */)
{
   fUnloaded->Delete(option);
   fIds->Clear();
   THashList::Delete(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Specialize FindObject to do search for the
/// a function just by name or create it if its not already in the list

TObject *TListOfFunctionTemplates::FindObject(const char *name) const
{
   TObject *result = THashList::FindObject(name);
   if (!result) {

      R__LOCKGUARD(gInterpreterMutex);

      TInterpreter::DeclId_t decl;
      if (fClass) decl = gInterpreter->GetFunctionTemplate(fClass->GetClassInfo(),name);
      else        decl = gInterpreter->GetFunctionTemplate(0,name);
      if (decl) result = const_cast<TListOfFunctionTemplates*>(this)->Get(decl);
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the set of overloads for this name, collecting all available ones.
/// Can construct and insert new TFunction-s.

TList* TListOfFunctionTemplates::GetListForObjectNonConst(const char* name)
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
      while (TFunctionTemplate* over = (TFunctionTemplate*)iOverload()) {
         wasEmpty = false;
         overloadsSet.Add((Long64_t)(ULong64_t)over->GetDeclId(),
                          (Long64_t)(ULong64_t)over);
      }
   }

   // Update if needed.
   std::vector<DeclId_t> overloadDecls;
   ClassInfo_t* ci = fClass ? fClass->GetClassInfo() : 0;
   gInterpreter->GetFunctionOverloads(ci, name, overloadDecls);
   for (std::vector<DeclId_t>::const_iterator iD = overloadDecls.begin(),
           eD = overloadDecls.end(); iD != eD; ++iD) {
      TFunctionTemplate* over = Get(*iD);
      if (wasEmpty || !overloadsSet.GetValue((Long64_t)(ULong64_t)over->GetDeclId())) {
          overloads->Add(over);
      }
   }

   return overloads;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the set of overloads for this name, collecting all available ones.
/// Can construct and insert new TFunction-s.

TList* TListOfFunctionTemplates::GetListForObject(const char* name) const
{
   return const_cast<TListOfFunctionTemplates*>(this)->GetListForObjectNonConst(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the set of overloads for function obj, collecting all available ones.
/// Can construct and insert new TFunction-s.

TList* TListOfFunctionTemplates::GetListForObject(const TObject* obj) const
{
   if (!obj) return 0;
   return const_cast<TListOfFunctionTemplates*>(this)
      ->GetListForObjectNonConst(obj->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Return (after creating it if necessary) the TMethod or TFunction
/// describing the function corresponding to the Decl 'id'.

TFunctionTemplate *TListOfFunctionTemplates::Get(DeclId_t id, bool verify)
{
   if (!id) return 0;

   TFunctionTemplate *f = (TFunctionTemplate*)fIds->GetValue((Long64_t)id);
   if (!f) {
      if (verify) {
         if (fClass) {
            if (!gInterpreter->ClassInfo_Contains(fClass->GetClassInfo(),id)) return 0;
         } else {
            if (!gInterpreter->ClassInfo_Contains(0,id)) return 0;
         }
      }

      R__LOCKGUARD(gInterpreterMutex);

      FuncTempInfo_t *m = gInterpreter->FuncTempInfo_Factory(id);

      // Let's see if this is a reload ...
      TString name;
      gInterpreter->FuncTempInfo_Name(m, name);
      TFunctionTemplate* update = (TFunctionTemplate*)fUnloaded->FindObject(name);
      if (update) {
         fUnloaded->Remove(update);
         update->Update(m);
         f = update;
      }
      if (!f) {
         if (fClass) f = new TFunctionTemplate(m, fClass);
         else f = new TFunctionTemplate(m, 0);
      }
      // Calling 'just' THahList::Add would turn around and call
      // TListOfFunctionTemplates::AddLast which should *also* do the fIds->Add.
      THashList::AddLast(f);
      fIds->Add((Long64_t)id,(Long64_t)f);
   }
   return f;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a pair<id, object> from the map of functions and their ids.

void TListOfFunctionTemplates::UnmapObject(TObject *obj)
{
   TFunctionTemplate *f = dynamic_cast<TFunctionTemplate*>(obj);
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

void TListOfFunctionTemplates::RecursiveRemove(TObject *obj)
{
   if (!obj) return;

   THashList::RecursiveRemove(obj);
   fUnloaded->RecursiveRemove(obj);
   UnmapObject(obj);

}

////////////////////////////////////////////////////////////////////////////////
/// Remove object from the list.

TObject* TListOfFunctionTemplates::Remove(TObject *obj)
{
   Bool_t found;

   found = THashList::Remove(obj);
   if (!found) {
      found = fUnloaded->Remove(obj);
   }
   UnmapObject(obj);
   if (found) return obj;
   else return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove object via its objlink from the list.

TObject* TListOfFunctionTemplates::Remove(TObjLink *lnk)
{
   if (!lnk) return 0;

   TObject *obj = lnk->GetObject();

   THashList::Remove(lnk);
   fUnloaded->Remove(obj);

   UnmapObject(obj);
   return obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Load all the functions known to the interpreter for the scope 'fClass'
/// into this collection.

void TListOfFunctionTemplates::Load()
{
   if (fClass && fClass->GetClassInfo() == 0) return;

   R__LOCKGUARD(gInterpreterMutex);

   ULong64_t currentTransaction = gInterpreter->GetInterpreterStateMarker();
   if (currentTransaction == fLastLoadMarker) {
      return;
   }
   fLastLoadMarker = currentTransaction;

   gInterpreter->LoadFunctionTemplates(fClass);
}

////////////////////////////////////////////////////////////////////////////////
/// Mark 'all func' as being unloaded.
/// After the unload, the function can no longer be found directly,
/// until the decl can be found again in the interpreter (in which
/// the func object will be reused.

void TListOfFunctionTemplates::Unload()
{
   TObjLink *lnk = FirstLink();
   while (lnk) {
      TFunctionTemplate *func = (TFunctionTemplate*)lnk->GetObject();

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

void TListOfFunctionTemplates::Unload(TFunctionTemplate *func)
{
   if (THashList::Remove(func)) {
      // We contains the object, let remove it from the other internal
      // list and move it to the list of unloaded objects.

      fIds->Remove((Long64_t)func->GetDeclId());
      fUnloaded->Add(func);
   }
}
