// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfDataMembers
#define ROOT_TListOfDataMembers

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfDataMembers                                                   //
//                                                                      //
// A collection of TDataMember objects designed for fast access given a //
// DeclId_t and for keep track of TDataMember that were described       //
// unloaded member.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THashList.h"

#include "TDictionary.h"

class TExMap;
class TDataMember;

class TListOfDataMembers : public THashList
{
private:
   TClass    *fClass;    //! Context of this list.  Not owned.

   TExMap    *fIds;      //! Map from DeclId_t to TDataMember*
   THashList *fUnloaded; //! Holder of TDataMember for unloaded DataMembers.
   Bool_t     fIsLoaded; //! Mark whether Load was executed.
   ULong64_t  fLastLoadMarker; //! Represent interpreter state when we last did a full load.

   TListOfDataMembers(const TListOfDataMembers&);              // not implemented
   TListOfDataMembers& operator=(const TListOfDataMembers&);   // not implemented

   void       MapObject(TObject *obj);
   void       UnmapObject(TObject *obj);

public:
   typedef TDictionary::DeclId_t DeclId_t;

   TListOfDataMembers(TClass *cl = 0);
   // construct from a generic collection of data members objects
   template<class DataMemberList>
   TListOfDataMembers(DataMemberList & dmlist) :
      fClass(0),fIds(0),fUnloaded(0),
      fIsLoaded(kTRUE), fLastLoadMarker(0)
   {
      for (auto * dataMember : dmlist)
         Add(dataMember);
   }

   ~TListOfDataMembers() override;

   void Clear(Option_t *option) override;
   void Delete(Option_t *option = "") override;

   using THashList::FindObject;
   TObject *FindObject(const char *name) const override;

   TDictionary *Find(DeclId_t id) const;
   TDictionary *Get(DeclId_t id);
   TDictionary *Get(DataMemberInfo_t *info, bool skipChecks=kFALSE);

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

   TClass    *GetClass() const { return fClass; }
   void       SetClass(TClass* cl) { fClass = cl; }
   void       Update(TDictionary *member);

   void     RecursiveRemove(TObject *obj) override;
   TObject *Remove(TObject *obj) override;
   TObject *Remove(TObjLink *lnk) override;

   void Load();
   void Unload();
   void Unload(TDictionary *member);

   ClassDef(TListOfDataMembers,2);  // List of TDataMembers for a class
};

#endif // ROOT_TListOfDataMembers
