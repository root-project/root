// @(#)root/cont
// Author: Bianca-Cristina Cristescu February 2014

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfEnumsWithLock
#define ROOT_TListOfEnumsWithLock

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfEnumsWithLock                                                         //
//                                                                      //
// A collection of TEnum objects designed for fast access given a       //
// DeclId_t and for keep track of TEnum that were described             //
// unloaded enum.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TListOfEnums
#include "TListOfEnums.h"
#endif

class TExMap;
class TEnum;

class TListOfEnumsWithLock : public TListOfEnums
{
private:
   typedef TDictionary::DeclId_t DeclId_t;

   TListOfEnumsWithLock(const TListOfEnumsWithLock&) = delete;
   TListOfEnumsWithLock& operator=(const TListOfEnumsWithLock&) = delete;

public:

   TListOfEnumsWithLock(TClass *cl = 0);
   ~TListOfEnumsWithLock() override;

   TEnum *GetObject(const char*) const override;

   void Clear(Option_t *option) override;
   void Delete(Option_t *option="") override;

   TObject   *FindObject(const TObject* obj) const override;
   TObject   *FindObject(const char *name) const override;
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

   ClassDefOverride(TListOfEnumsWithLock,2);  // List of TDataMembers for a class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfEnumsWithLockIter                                             //
//                                                                      //
// Iterator of TListOfEnumsWithLock.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TListOfEnumsWithLockIter : public TListIter
{
 public:
   TListOfEnumsWithLockIter(const TListOfEnumsWithLock *l, Bool_t dir = kIterForward);

   using TListIter::operator=;

   TObject *Next();

   ClassDef(TListOfEnumsWithLockIter,0)
};

#endif // ROOT_TListOfEnumsWithLock
