// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfFunctions
#define ROOT_TListOfFunctions

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfFunctions                                                     //
//                                                                      //
// A collection of TFunction objects designed for fast access given a   //
// DeclId_t and for keep track of TFunction that were described         //
// unloaded function.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THashList.h"

#include "THashTable.h"

#include "TDictionary.h"

class TExMap;
class TFunction;

class TListOfFunctions : public THashList
{
private:
   friend class TClass;
   TClass    *fClass; // Context of this list.  Not owned.

   TExMap    *fIds;      // Map from DeclId_t to TFunction*
   THashList *fUnloaded; // Holder of TFunction for unloaded functions.
   THashTable fOverloads; // TLists of overloads.
   ULong64_t  fLastLoadMarker; // Represent interpreter state when we last did a full load.

   TListOfFunctions(const TListOfFunctions&) = delete;
   TListOfFunctions& operator=(const TListOfFunctions&) = delete;
   TList     *GetListForObjectNonConst(const char* name);

   void       MapObject(TObject *obj);
   void       UnmapObject(TObject *obj);

public:
   typedef TDictionary::DeclId_t DeclId_t;

   TListOfFunctions(TClass *cl);
   ~TListOfFunctions();

   void       Clear(Option_t *option="") override;
   void       Delete(Option_t *option="") override;

   TObject   *FindObject(const TObject* obj) const override;
   TObject   *FindObject(const char *name) const override;
   virtual TList     *GetListForObject(const char* name) const;
   virtual TList     *GetListForObject(const TObject* obj) const;
   TIterator *MakeIterator(Bool_t dir = kIterForward) const override;

   TObject   *At(Int_t idx) const override;
   TObject   *After(const TObject *obj) const override;
   TObject   *Before(const TObject *obj) const override;
   TObject   *First() const override;
   TObjLink  *FirstLink() const override;
   TObject  **GetObjectRef(const TObject *obj) const override;
   TObject   *Last() const override;
   TObjLink  *LastLink() const override;

   Int_t      GetLast() const override;
   Int_t      IndexOf(const TObject *obj) const override;

   Int_t      GetSize() const override;

   TFunction *Find(DeclId_t id) const;
   TFunction *Get(DeclId_t id);

   void       AddFirst(TObject *obj) override;
   void       AddFirst(TObject *obj, Option_t *opt) override;
   void       AddLast(TObject *obj) override;
   void       AddLast(TObject *obj, Option_t *opt) override;
   void       AddAt(TObject *obj, Int_t idx) override;
   void       AddAfter(const TObject *after, TObject *obj) override;
   void       AddAfter(TObjLink *after, TObject *obj) override;
   void       AddBefore(const TObject *before, TObject *obj) override;
   void       AddBefore(TObjLink *before, TObject *obj) override;

   void       RecursiveRemove(TObject *obj) override;
   TObject   *Remove(TObject *obj) override;
   TObject   *Remove(TObjLink *lnk) override;

   void Load();
   void Unload();
   void Unload(TFunction *func);

   ClassDefOverride(TListOfFunctions,0);  // List of TFunctions for a class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfFunctionsIter                                                 //
//                                                                      //
// Iterator of TListOfFunctions.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TListOfFunctionsIter : public TListIter
{
public:
   TListOfFunctionsIter(const TListOfFunctions *l, Bool_t dir = kIterForward);

   using TListIter::operator=;

   TObject           *Next() override;

   ClassDefOverride(TListOfFunctionsIter,0)
};


#endif // ROOT_TListOfFunctions
