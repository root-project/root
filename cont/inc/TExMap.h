// @(#)root/cont:$Name$:$Id$
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

class TExMap : public TObject {

private:
   struct Assoc_t {
      ULong_t  fHash;
      Long_t   fKey;
      Long_t   fValue;
      Assoc_t(ULong_t hash, Long_t key, Long_t val)
         { fHash = hash; fKey = key; fValue = val; }
   };

   Assoc_t   **fTable;
   Int_t       fSize;
   Int_t       fTally;

   Bool_t      HighWaterMark()
                  { return (Bool_t) (fTally >= ((3*fSize)/4)); }
   void        Expand(Int_t newsize);
   Int_t       FindElement(ULong_t hash, Long_t key);
   void        FixCollisions(Int_t index);

public:
   TExMap(Int_t mapSize = 100);
   ~TExMap();

   void      Add(ULong_t hash, Long_t key, Long_t value);
   void      Add(Long_t key, Long_t value) { Add(key, key, value); }
   void      Delete(Option_t *opt = "");
   Int_t     Capacity() const { return fSize; }
   Int_t     GetSize() const { return fTally; }
   Long_t    GetValue(ULong_t hash, Long_t key);
   Long_t    GetValue(Long_t key) { return GetValue(key, key); }
   void      Remove(ULong_t hash, Long_t key);
   void      Remove(Long_t key) { Remove(key, key); }

   Long_t   &operator()(ULong_t hash, Long_t key);
   Long_t   &operator()(Long_t key) { return operator()(key, key); }

   ClassDef(TExMap,0)  //Map with external hash
};

#endif
