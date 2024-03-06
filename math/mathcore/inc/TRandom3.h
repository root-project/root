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

class TRandom3 final: public TRandom {

private:
   UInt_t   fMt[624];
   Int_t    fCount624;

public:
   TRandom3(UInt_t seed=4357);
   ~TRandom3() override;
    /// Return one element of the generator state used to generate the random numbers.
    /// Note that it is not the seed of the generator that was used in the SetSeed function and 
    /// the full state (624 numbers) is required to define the generator and have a reproducible output.
    UInt_t    GetSeed() const override { return fMt[fCount624 % 624];}
   using TRandom::Rndm;
   inline Double_t  Rndm( ) override;
   void      RndmArray(Int_t n, Float_t *array) override;
   void      RndmArray(Int_t n, Double_t *array) override;
   void      SetSeed(ULong_t seed=0) override;
   const UInt_t *GetState() const { return fMt; }

   ClassDefOverride(TRandom3,2)  //Random number generator: Mersenne Twister
};

////////////////////////////////////////////////////////////////////////////////
///  Machine independent random number generator.
///  Produces uniformly-distributed floating points in (0,1)
///  Method: Mersenne Twister

Double_t TRandom3::Rndm()
{
   UInt_t y;

   do {
      constexpr Int_t  kM = 397;
      constexpr Int_t  kN = 624;
      constexpr UInt_t kTemperingMaskB =  0x9d2c5680;
      constexpr UInt_t kTemperingMaskC =  0xefc60000;
      constexpr UInt_t kUpperMask =       0x80000000;
      constexpr UInt_t kLowerMask =       0x7fffffff;
      constexpr UInt_t kMatrixA =         0x9908b0df;

      if (fCount624 >= kN) {
         Int_t i;

         for (i=0; i < kN-kM; i++) {
            y = (fMt[i] & kUpperMask) | (fMt[i+1] & kLowerMask);
            fMt[i] = fMt[i+kM] ^ (y >> 1) ^ ((y & 0x1) ? kMatrixA : 0x0);
         }

         for (   ; i < kN-1    ; i++) {
            y = (fMt[i] & kUpperMask) | (fMt[i+1] & kLowerMask);
            fMt[i] = fMt[i+kM-kN] ^ (y >> 1) ^ ((y & 0x1) ? kMatrixA : 0x0);
         }

         y = (fMt[kN-1] & kUpperMask) | (fMt[0] & kLowerMask);
         fMt[kN-1] = fMt[kM-1] ^ (y >> 1) ^ ((y & 0x1) ? kMatrixA : 0x0);
         fCount624 = 0;
      }

      y = fMt[fCount624++];
      y ^=  (y >> 11);
      y ^= ((y << 7 ) & kTemperingMaskB );
      y ^= ((y << 15) & kTemperingMaskC );
      y ^=  (y >> 18);
   } while (y == 0);

   // 2.3283064365386963e-10 == 1./(max<UINt_t>+1)  -> then returned value cannot be = 1.0
   return static_cast<Double_t>(y) * 2.3283064365386963e-10; // * Power(2,-32)
}


R__EXTERN TRandom *gRandom;

#endif
