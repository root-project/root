// @(#)root/mathcore:$Id$
// Authors: L. Moneta    8/2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015 , ROOT MathLib Team                             *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file of Mersenne-Twister engine
//
//
// Created by: Lorenzo Moneta  : Tue 4 Aug 2015
//
//
#include "Math/MersenneTwisterEngine.h"


namespace ROOT {
namespace Math {

   /// set the seed x
   void MersenneTwisterEngine::SetSeed(unsigned int seed) {
      fCount624 = 624;
      if (seed > 0) {
         fMt[0] = seed;
         
         // use multipliers from  Knuth's "Art of Computer Programming" Vol. 2, 3rd Ed. p.106
         for(int i=1; i<624; i++) {
            fMt[i] = (1812433253 * ( fMt[i-1]  ^ ( fMt[i-1] >> 30)) + i );
         }
      }
   }

   /// generate a random double number 
   double MersenneTwisterEngine::Rndm_impl() {

      
      uint32_t y;
      
      const int  kM = 397;
      const int  kN = 624;
      const uint32_t kTemperingMaskB =  0x9d2c5680;
      const uint32_t kTemperingMaskC =  0xefc60000;
      const uint32_t kUpperMask =       0x80000000;
      const uint32_t kLowerMask =       0x7fffffff;
      const uint32_t kMatrixA =         0x9908b0df;
      
      if (fCount624 >= kN) {
         int i;
         
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
      
      // 2.3283064365386963e-10 == 1./(max<UINt_t>+1)  -> then returned value cannot be = 1.0
      if (y) return ( (double) y * 2.3283064365386963e-10); // * Power(2,-32)
      return Rndm_impl();
      
   }
  
   } // namespace Math
} // namespace ROOT
