// @(#)root/cont
// Author: Bianca-Cristina Cristescu March 2014

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfFunctionTemplates                                                     //
//                                                                      //
// A collection of TFunction objects designed for fast access given a   //
// DeclId_t and for keep track of TFunction that were described         //
// unloaded function.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TListOfFunctionTemplates.h"
#include "TClass.h"
#include "TExMap.h"
#include "TFunction.h"
#include "TFunctionTemplate.h"
#include "TMethod.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

ClassImp(TListOfFunctionTemplates)

//______________________________________________________________________________
TListOfFunctionTemplates::TListOfFunctionTemplates(TClass *cl) : fClass(cl),fIds(0),
                          fUnloaded(0),fLastLoadMarker(0)
{
   // Constructor.

   fIds = new TExMap;
   fUnloaded = new THashList;
}

//______________________________________________________________________________
TListOfFunctionTemplates::~TListOfFunctionTemplates()
{
   // Destructor.

   THashList::Delete();
   delete fIds;
   fUnloaded->Delete();
   delete fUnloaded;
}

//______________________________________________________________________________
void TListOfFunctionTemplates::MapObject(TObject *obj)
{
   // Add pair<id, object> to the map of functions and their ids.

   TFunctionTemplate *f = dynamic_cast<TFunctionTemplate*>(obj);
   if (f) {
      fIds->Add((Long64_t)f->GetDeclId(),(Long64_t)f);
   }
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddFirst(TObject *obj)
{
   // Add object at the beginning of the list.

   THashList::AddFirst(obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddFirst(TObject *obj, Option_t *opt)
{
   // Add object at the beginning of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   THashList::AddFirst(obj,opt);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddLast(TObject *obj)
{
   // Add object at the end of the list.

   THashList::AddLast(obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddLast(TObject *obj, Option_t *opt)
{
   // Add object at the end of the list and also store option.
   // Storing an option is useful when one wants to change the behaviour
   // of an object a little without having to create a complete new
   // copy of the object. This feature is used, for example, by the Draw()
   // method. It allows the same object to be drawn in different ways.

   THashList::AddLast(obj, opt);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddAt(TObject *obj, Int_t idx)
{
   // Insert object at location idx in the list.

   THashList::AddAt(obj, idx);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddAfter(const TObject *after, TObject *obj)
{
   // Insert object after object after in the list.

   THashList::AddAfter(after, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddAfter(TObjLink *after, TObject *obj)
{
   // Insert object after object after in the list.

   THashList::AddAfter(after, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddBefore(const TObject *before, TObject *obj)
{
   // Insert object before object before in the list.

   THashList::AddBefore(before, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::AddBefore(TObjLink *before, TObject *obj)
{
   // Insert object before object before in the list.

   THashList::AddBefore(before, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::Clear(Option_t *option)
{
   // Remove all objects from the list. Does not delete the objects unless
   // the THashList is the owner (set via SetOwner()).

   fUnloaded->Clear(option);
   fIds->Clear();
   THashList::Clear(option);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::Delete(Option_t *option /* ="" */)
{
   // Delete all TFunction object files.

   fUnloaded->Delete(option);
   fIds->Clear();
   THashList::Delete(option);
}

//______________________________________________________________________________
TObject *TListOfFunctionTemplates::FindObject(const char *name) const
{
   // Specialize FindObject to do search for the
   // a function just by name or create it if its not already in the list

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

//______________________________________________________________________________
TList* TListOfFunctionTemplates::GetListForObjectNonConst(const char* name)
{
   // Return the set of overloads for this name, collecting all available ones.
   // Can construct and insert new TFunction-s.

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

//______________________________________________________________________________
TList* TListOfFunctionTemplates::GetListForObject(const char* name) const
{
   // Return the set of overloads for this name, collecting all available ones.
   // Can construct and insert new TFunction-s.
   return const_cast<TListOfFunctionTemplates*>(this)->GetListForObjectNonConst(name);
}

//______________________________________________________________________________
TList* TListOfFunctionTemplates::GetListForObject(const TObject* obj) const
{
   // Return the set of overloads for function obj, collecting all available ones.
   // Can construct and insert new TFunction-s.
   if (!obj) return 0;
   return const_cast<TListOfFunctionTemplates*>(this)
      ->GetListForObjectNonConst(obj->GetName());
}

//______________________________________________________________________________
TFunctionTemplate *TListOfFunctionTemplates::Get(DeclId_t id)
{
   // Return (after creating it if necessary) the TMethod or TFunction
   // describing the function corresponding to the Decl 'id'.

   if (!id) return 0;

   TFunctionTemplate *f = (TFunctionTemplate*)fIds->GetValue((Long64_t)id);
   if (!f) {
      if (fClass) {
         if (!gInterpreter->ClassInfo_Contains(fClass->GetClassInfo(),id)) return 0;
      } else {
         if (!gInterpreter->ClassInfo_Contains(0,id)) return 0;
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

//______________________________________________________________________________
void TListOfFunctionTemplates::UnmapObject(TObject *obj)
{
   // Remove a pair<id, object> from the map of functions and their ids.
   TFunctionTemplate *f = dynamic_cast<TFunctionTemplate*>(obj);
   if (f) {
      fIds->Remove((Long64_t)f->GetDeclId());
   }
}

//______________________________________________________________________________
void TListOfFunctionTemplates::RecursiveRemove(TObject *obj)
{
   // Remove object from this collection and recursively remove the object
   // from all other objects (and collections).
   // This function overrides TCollection::RecursiveRemove that calls
   // the Remove function. THashList::Remove cannot be called because
   // it uses the hash value of the hash table. This hash value
   // is not available anymore when RecursiveRemove is called from
   // the TObject destructor.

   if (!obj) return;

   THashList::RecursiveRemove(obj);
   fUnloaded->RecursiveRemove(obj);
   UnmapObject(obj);

}

//______________________________________________________________________________
TObject* TListOfFunctionTemplates::Remove(TObject *obj)
{
   // Remove object from the list.

   Bool_t found;

   found = THashList::Remove(obj);
   if (!found) {
      found = fUnloaded->Remove(obj);
   }
   UnmapObject(obj);
   if (found) return obj;
   else return 0;
}

//______________________________________________________________________________
TObject* TListOfFunctionTemplates::Remove(TObjLink *lnk)
{
   // Remove object via its objlink from the list.

   if (!lnk) return 0;

   TObject *obj = lnk->GetObject();

   THashList::Remove(lnk);
   fUnloaded->Remove(obj);

   UnmapObject(obj);
   return obj;
}

//______________________________________________________________________________
void TListOfFunctionTemplates::Load()
{
   // Load all the functions known to the intepreter for the scope 'fClass'
   // into this collection.

   if (fClass && fClass->GetClassInfo() == 0) return;

   R__LOCKGUARD(gInterpreterMutex);

   ULong64_t currentTransaction = gInterpreter->GetInterpreterStateMarker();
   if (currentTransaction == fLastLoadMarker) {
      return;
   }
   fLastLoadMarker = currentTransaction;

   gInterpreter->LoadFunctionTemplates(fClass);
}

//______________________________________________________________________________
void TListOfFunctionTemplates::Unload()
{
   // Mark 'all func' as being unloaded.
   // After the unload, the function can no longer be found directly,
   // until the decl can be found again in the interpreter (in which
   // the func object will be reused.

   TObjLink *lnk = FirstLink();
   while (lnk) {
      TFunctionTemplate *func = (TFunctionTemplate*)lnk->GetObject();

      fIds->Remove((Long64_t)func->GetDeclId());
      fUnloaded->Add(func);

      lnk = lnk->Next();
   }

   THashList::Clear();
}

//______________________________________________________________________________
void TListOfFunctionTemplates::Unload(TFunctionTemplate *func)
{
   // Mark 'func' as being unloaded.
   // After the unload, the function can no longer be found directly,
   // until the decl can be found again in the interpreter (in which
   // the func object will be reused.

   if (THashList::Remove(func)) {
      // We contains the object, let remove it from the other internal
      // list and move it to the list of unloaded objects.

      fIds->Remove((Long64_t)func->GetDeclId());
      fUnloaded->Add(func);
   }
}
