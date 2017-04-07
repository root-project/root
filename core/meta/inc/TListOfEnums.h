// @(#)root/cont
// Author: Bianca-Cristina Cristescu February 2014

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfEnums
#define ROOT_TListOfEnums

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfEnums                                                         //
//                                                                      //
// A collection of TEnum objects designed for fast access given a       //
// DeclId_t and for keep track of TEnum that were described             //
// unloaded enum.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THashList.h"

#include "TDictionary.h"

class TExMap;
class TEnum;

class TListOfEnums : public THashList
{
private:
   friend class TCling;
   friend class TClass;
   friend class TProtoClass;
   friend class TROOT;

   TClass    *fClass; //! Context of this list.  Not owned.

   TExMap    *fIds;      //! Map from DeclId_t to TEnum*
   THashList *fUnloaded; //! Holder of TEnum for unloaded Enums.
   Bool_t     fIsLoaded; //! Mark whether Load was executed.
   ULong64_t  fLastLoadMarker; //! Represent interpreter state when we last did a full load.

   TListOfEnums(const TListOfEnums&) = delete;
   TListOfEnums& operator=(const TListOfEnums&) = delete;

   void MapObject(TObject *obj);
   void UnmapObject(TObject *obj);

   void Load();
   void Unload();
   void Unload(TEnum *e);
   void SetClass(TClass* cl) { fClass = cl; }

public:
   typedef TDictionary::DeclId_t DeclId_t;

protected:
   TClass *GetClass() const {return fClass;}
   TExMap *GetIds() { return fIds;}
   TEnum  *FindUnloaded(const char* name) { return (TEnum*)fUnloaded->FindObject(name);}
   TEnum  *Get(DeclId_t id, const char *name);

public:
   TListOfEnums(TClass *cl = 0);
   ~TListOfEnums() override;

   TEnum     *Find(DeclId_t id) const;
   virtual TEnum *GetObject(const char*) const;

   void Clear(Option_t *option) override;
   void Delete(Option_t *option="") override;

   Bool_t     IsLoaded() const { return fIsLoaded; }
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

   ClassDefOverride(TListOfEnums,2);  // List of TDataMembers for a class
};

#endif // ROOT_TListOfEnums
