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

#ifndef ROOT_THastList
#include "THashList.h"
#endif

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif

class TExMap;
class TEnum;

class TListOfEnums : public THashList
{
private:
   TClass    *fClass; //! Context of this list.  Not owned.

   TExMap    *fIds;      //! Map from DeclId_t to TEnum*
   THashList *fUnloaded; //! Holder of TEnum for unloaded Enums.
   Bool_t     fIsLoaded; //! Mark whether Load was executed.
   ULong64_t  fLastLoadMarker; //! Represent interpreter state when we last did a full load.

   TListOfEnums(const TListOfEnums&);              // not implemented
   TListOfEnums& operator=(const TListOfEnums&);   // not implemented

   void       MapObject(TObject *obj);
   void       UnmapObject(TObject *obj);

public:
   typedef TDictionary::DeclId_t DeclId_t;

   TListOfEnums(TClass *cl = 0);
   ~TListOfEnums();

   virtual void Clear(Option_t *option);
   virtual void Delete(Option_t *option="");

   using THashList::FindObject;
   virtual TObject   *FindObject(const char *name) const;

   TEnum     *Find(DeclId_t id) const;
   TEnum     *Get(DeclId_t id, const char *name);

   Bool_t     IsLoaded() const { return fIsLoaded; }
   void       AddFirst(TObject *obj);
   void       AddFirst(TObject *obj, Option_t *opt);
   void       AddLast(TObject *obj);
   void       AddLast(TObject *obj, Option_t *opt);
   void       AddAt(TObject *obj, Int_t idx);
   void       AddAfter(const TObject *after, TObject *obj);
   void       AddAfter(TObjLink *after, TObject *obj);
   void       AddBefore(const TObject *before, TObject *obj);
   void       AddBefore(TObjLink *before, TObject *obj);

   void       RecursiveRemove(TObject *obj);
   TObject   *Remove(TObject *obj);
   TObject   *Remove(TObjLink *lnk);

   void Load();
   void Unload();
   void Unload(TEnum *e);
   void       SetClass(TClass* cl) { fClass = cl; }

   ClassDef(TListOfEnums,2);  // List of TDataMembers for a class
};

#endif // ROOT_TListOfEnums
