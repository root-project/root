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

class TRandom3 : public TRandom {

private:
   UInt_t   fMt[624];
   Int_t    fCount624;

public:
   TRandom3(UInt_t seed=4357);
   ~TRandom3() override;
   /// return current element of the state used for generate the random number
   /// Note that it is not the seed of the generator that was used in the SetSeed function
    UInt_t    GetSeed() const override { return fMt[fCount624];}
   using TRandom::Rndm;
    Double_t  Rndm( ) override;
    void      RndmArray(Int_t n, Float_t *array) override;
    void      RndmArray(Int_t n, Double_t *array) override;
    void      SetSeed(ULong_t seed=0) override;
   virtual const UInt_t *GetState() const { return fMt; }

   ClassDefOverride(TRandom3,2)  //Random number generator: Mersenne Twister
};

R__EXTERN TRandom *gRandom;

#endif
