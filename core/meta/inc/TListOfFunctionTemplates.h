// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfFunctionTemplates
#define ROOT_TListOfFunctionTemplates

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfFunctionTemplates                                             //
//                                                                      //
// A collection of TFunctionTemplate objects designed for fast access   //
// given a DeclId_t and for keep track of TFunctionTempalte that were   //
// described unloaded function.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THashList.h"

#include "THashTable.h"

#include "TDictionary.h"

class TExMap;
class TFunctionTemplate;

class TListOfFunctionTemplates : public THashList
{
private:
   friend class TClass;

   typedef TDictionary::DeclId_t DeclId_t;
   TClass    *fClass; // Context of this list.  Not owned.

   TExMap    *fIds;      // Map from DeclId_t to TFunction*
   THashList *fUnloaded; // Holder of TFunction for unloaded functions.
   THashTable fOverloads; // TLists of overloads.
   ULong64_t  fLastLoadMarker; // Represent interpreter state when we last did a full load.

   TListOfFunctionTemplates(const TListOfFunctionTemplates&) = delete;
   TListOfFunctionTemplates& operator=(const TListOfFunctionTemplates&) = delete;
   TList     *GetListForObjectNonConst(const char* name);

   void       MapObject(TObject *obj);
   void       UnmapObject(TObject *obj);

public:

   TListOfFunctionTemplates(TClass *cl);
   ~TListOfFunctionTemplates();

   void       Clear(Option_t *option="") override;
   void       Delete(Option_t *option="") override;

   using THashList::FindObject;
   TObject   *FindObject(const char *name) const override;
   virtual TList     *GetListForObject(const char* name) const;
   virtual TList     *GetListForObject(const TObject* obj) const;

   TFunctionTemplate *Get(DeclId_t id);

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
   void Unload(TFunctionTemplate *func);

   ClassDefOverride(TListOfFunctionTemplates,0);  // List of TFunctions for a class
};

#endif // ROOT_TListOfFunctionTemplates
