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
// TListOfEnums                                                         //
//                                                                      //
// A collection of TEnum objects designed for fast access given a       //
// DeclId_t and for keep track of TEnum that were described             //
// unloaded enum.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TListOfEnums.h"
#include "TClass.h"
#include "TExMap.h"
#include "TEnum.h"
#include "TGlobal.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

ClassImp(TListOfEnums)

//______________________________________________________________________________
TListOfEnums::TListOfEnums(TClass *cl) : fClass(cl),fIds(0),fUnloaded(0),fIsLoaded(kFALSE),
              fLastLoadMarker(0)
{
   // Constructor.

   fIds = new TExMap;
   fUnloaded = new THashList;
}

//______________________________________________________________________________
TListOfEnums::~TListOfEnums()
{
   // Destructor.

   THashList::Delete();
   delete fIds;
   fUnloaded->Delete();
   delete fUnloaded;
}

//______________________________________________________________________________
void TListOfEnums::MapObject(TObject *obj)
{
   // Add pair<id, object> to the map of functions and their ids.

   TEnum *e = dynamic_cast<TEnum*>(obj);
   if (e) {
      fIds->Add((Long64_t)e->GetDeclId(),(Long64_t)e);
   }
}

//______________________________________________________________________________
void TListOfEnums::AddFirst(TObject *obj)
{
   // Add object at the beginning of the list.

   THashList::AddFirst(obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::AddFirst(TObject *obj, Option_t *opt)
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
void TListOfEnums::AddLast(TObject *obj)
{
   // Add object at the end of the list.

   THashList::AddLast(obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::AddLast(TObject *obj, Option_t *opt)
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
void TListOfEnums::AddAt(TObject *obj, Int_t idx)
{
   // Insert object at location idx in the list.

   THashList::AddAt(obj, idx);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::AddAfter(const TObject *after, TObject *obj)
{
   // Insert object after object after in the list.

   THashList::AddAfter(after, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::AddAfter(TObjLink *after, TObject *obj)
{
   // Insert object after object after in the list.

   THashList::AddAfter(after, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::AddBefore(const TObject *before, TObject *obj)
{
   // Insert object before object before in the list.

   THashList::AddBefore(before, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::AddBefore(TObjLink *before, TObject *obj)
{
   // Insert object before object before in the list.

   THashList::AddBefore(before, obj);
   MapObject(obj);
}

//______________________________________________________________________________
void TListOfEnums::Clear(Option_t *option)
{
   // Remove all objects from the list. Does not delete the objects unless
   // the THashList is the owner (set via SetOwner()).

   fUnloaded->Clear(option);
   fIds->Clear();
   THashList::Clear(option);
   fIsLoaded = kFALSE;
}

//______________________________________________________________________________
void TListOfEnums::Delete(Option_t *option /* ="" */)
{
   // Delete all TDataMember object files.

   fUnloaded->Delete(option);
   THashList::Delete(option);
   fIsLoaded = kFALSE;
}

//______________________________________________________________________________
TObject *TListOfEnums::FindObject(const char *name) const
{
   // Specialize FindObject to do search for the
   // a enum just by name or create it if its not already in the list

   TObject *result = THashList::FindObject(name);
   if (!result) {
      TInterpreter::DeclId_t decl;
      if (fClass) decl = gInterpreter->GetEnum(fClass, name);
      else        decl = gInterpreter->GetEnum(0, name);
      if (decl) result = const_cast<TListOfEnums*>(this)->Get(decl, name);
   }
   return result;
}

//______________________________________________________________________________
TEnum *TListOfEnums::Get(DeclId_t id, const char *name)
{
   // Return (after creating it if necessary) the TEnum
   // describing the enum corresponding to the Decl 'id'.

   if (!id) return 0;

   TEnum *e = (TEnum*)fIds->GetValue((Long64_t)id);
   if (!e) {
      if (fClass) {
         if (!fClass->HasInterpreterInfoInMemory()) {
            // The interpreter does not know about this class yet (or a problem
            // occurred that prevented the proper updating of fClassInfo).
            // So this decl can not possibly be part of this class.
            // [In addition calling GetClassInfo would trigger a late parsing
            //  of the header which we want to avoid].
            return 0;
         }
         if (!gInterpreter->ClassInfo_Contains(fClass->GetClassInfo(),id)) return 0;
      } else {
         if (!gInterpreter->ClassInfo_Contains(0,id)) return 0;
      }

      // Let's see if this is a reload ...
      // can we check for reloads for enums?
      e = (TEnum *)fUnloaded->FindObject(name);
      if (e) {
         e->Update(id);
         gInterpreter->UpdateEnumConstants(e, fClass);
      } else {
         e = gInterpreter->CreateEnum((void*)id, fClass);
      }
      // Calling 'just' THahList::Add would turn around and call
      // TListOfEnums::AddLast which should *also* do the fIds->Add.
      THashList::AddLast(e);
      fIds->Add((Long64_t)id,(Long64_t)e);
   }
   return e;
}

//______________________________________________________________________________
void TListOfEnums::UnmapObject(TObject *obj)
{
   // Remove a pair<id, object> from the map of functions and their ids.
   TEnum *e = dynamic_cast<TEnum*>(obj);
   if (e) {
      fIds->Remove((Long64_t)e->GetDeclId());
   }
}

//______________________________________________________________________________
void TListOfEnums::RecursiveRemove(TObject *obj)
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
TObject* TListOfEnums::Remove(TObject *obj)
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
TObject* TListOfEnums::Remove(TObjLink *lnk)
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
void TListOfEnums::Load()
{
   // Load all the DataMembers known to the intepreter for the scope 'fClass'
   // into this collection.

   if (fClass && fClass->Property() & (kIsClass|kIsStruct|kIsUnion)) {
      // Class and union are not extendable, if we already
      // loaded all the data member there is no need to recheck
      if (fIsLoaded) return;
   }

   // This will provoke the parsing of the headers if need be.
   if (fClass && fClass->GetClassInfo() == 0) return;

   ULong64_t currentTransaction = gInterpreter->GetInterpreterStateMarker();
   if (currentTransaction == fLastLoadMarker) {
      return;
   }
   fLastLoadMarker = currentTransaction;

   // In the case of namespace, even if we have loaded before we need to
   // load again in case there was new data member added.

   // Mark the list as loaded to avoid an infinite recursion in the case
   // where we have a data member that is a variable size array.  In that
   // case TDataMember::Init needs to get/load the list to find the data
   // member used as the array size.
   fIsLoaded = kTRUE;

   gInterpreter->LoadEnums(fClass);
}

//______________________________________________________________________________
void TListOfEnums::Unload()
{
   // Mark 'all func' as being unloaded.
   // After the unload, the data member can no longer be found directly,
   // until the decl can be found again in the interpreter (in which
   // the func object will be reused.

   TObjLink *lnk = FirstLink();
   while (lnk) {
      TEnum *data = (TEnum *)lnk->GetObject();

      fIds->Remove((Long64_t)data->GetDeclId());
      fUnloaded->Add(data);

      lnk = lnk->Next();
   }

   THashList::Clear();
   fIsLoaded = kFALSE;
}

//______________________________________________________________________________
void TListOfEnums::Unload(TEnum *e)
{
   // Mark 'func' as being unloaded.
   // After the unload, the data member can no longer be found directly,
   // until the decl can be found again in the interpreter (in which
   // the func object will be reused.

   if (THashList::Remove(e)) {
      // We contains the object, let remove it from the other internal
      // list and move it to the list of unloaded objects.
      fIds->Remove((Long64_t)e->GetDeclId());
      fUnloaded->Add(e);
   }
}
