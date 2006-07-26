// @(#)root/cont:$Name:  $:$Id: TExMap.h,v 1.8 2006/05/23 04:47:36 brun Exp $
// Author: Fons Rademakers   26/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TExMap
#define ROOT_TExMap


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TExMap                                                               //
//                                                                      //
// This class stores a (key,value) pair using an external hash.         //
// The (key,value) are Long_t's and therefore can contain object        //
// pointers or any longs. The map uses an open addressing hashing       //
// method (linear probing).                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TExMapIter;


class TExMap : public TObject {

friend class TExMapIter;

private:
   struct Assoc_t {
   private:
      ULong_t  fHash;
   public:
      Long_t   fKey;
      Long_t   fValue;
      void     SetHash(ULong_t h) { fHash = (h | 1); } // bit(0) is "1" when in use
      ULong_t  GetHash() const { return fHash; }
      Bool_t   InUse() const { return fHash & 1; }
      void     Clear() { fHash = 0x0; }
   };

   Assoc_t    *fTable;
   Int_t       fSize;
   Int_t       fTally;

   Bool_t      HighWaterMark() { return (Bool_t) (fTally >= ((3*fSize)/4)); }
   void        Expand(Int_t newsize);
   Int_t       FindElement(ULong_t hash, Long_t key);
   void        FixCollisions(Int_t index);


public:
   TExMap(Int_t mapSize = 100);
   TExMap(const TExMap &map);
   TExMap& operator=(const TExMap&);
   ~TExMap();

   void      Add(ULong_t hash, Long_t key, Long_t value);
   void      Add(Long_t key, Long_t value) { Add(key, key, value); }
   void      AddAt(UInt_t slot, ULong_t hash, Long_t key, Long_t value);
   void      Delete(Option_t *opt = "");
   Int_t     Capacity() const { return fSize; }
   Int_t     GetSize() const { return fTally; }
   Long_t    GetValue(ULong_t hash, Long_t key);
   Long_t    GetValue(Long_t key) { return GetValue(key, key); }
   Long_t    GetValue(ULong_t hash, Long_t key, UInt_t &slot);
   void      Remove(ULong_t hash, Long_t key);
   void      Remove(Long_t key) { Remove(key, key); }

   Long_t   &operator()(ULong_t hash, Long_t key);
   Long_t   &operator()(Long_t key) { return operator()(key, key); }

   ClassDef(TExMap,1)  //Map with external hash
};


class TExMapIter {

private:
   const TExMap   *fMap;
   Int_t           fCursor;

public:
   TExMapIter(const TExMap *map);
   TExMapIter(const TExMapIter& tei):
     fMap(tei.fMap), fCursor(tei.fCursor) { }
   TExMapIter& operator=(const TExMapIter&);
   virtual ~TExMapIter() { }

   const TExMap  *GetCollection() const { return fMap; }
   Bool_t         Next(ULong_t &hash, Long_t &key, Long_t &value);
   Bool_t         Next(Long_t &key, Long_t &value);
   void           Reset() { fCursor = 0; }

   ClassDef(TExMapIter,0)  // TExMap iterator
};

#endif
