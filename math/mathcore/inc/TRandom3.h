// @(#)root/mathcore:$Id$
// Author: Peter Malzacher   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRandom3
#define ROOT_TRandom3



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandom3                                                             //
//                                                                      //
// random number generator class: Mersenne Twister                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRandom.h"

#include <array>

class TRandom3;

class TRandom3State {
   friend TRandom3;
   std::array<UInt_t,624> fMt;
   Int_t fCount624;
   TRandom3State(const UInt_t *mt, Int_t count624) : fCount624(count624) {
      std::copy_n(mt, 624, fMt.begin());
   }
};

class TRandom3 : public TRandom {

private:
   UInt_t   fMt[624];
   Int_t    fCount624;

public:
   TRandom3(UInt_t seed=4357);
   virtual ~TRandom3();
   // get the current seed (only first element of the seed table)
   virtual  UInt_t    GetSeed() const { return fMt[0];}
   TRandom3State      GetState() const { return TRandom3State(fMt, fCount624);}
   void               SetState(const TRandom3State &state);
   using TRandom::Rndm;
   virtual  Double_t  Rndm( );
   virtual  void      RndmArray(Int_t n, Float_t *array);
   virtual  void      RndmArray(Int_t n, Double_t *array);
   virtual  void      SetSeed(ULong_t seed=0);

   ClassDef(TRandom3,2)  //Random number generator: Mersenne Twister
};

R__EXTERN TRandom *gRandom;

#endif
