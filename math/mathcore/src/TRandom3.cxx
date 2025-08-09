// @(#)root/mathcore:$Id$
// Author: Peter Malzacher   31/08/99

/**

\class TRandom3

Random number generator class based on
  M. Matsumoto and T. Nishimura,
  Mersenne Twister: A 623-dimensionally equidistributed
  uniform pseudorandom number generator
  ACM Transactions on Modeling and Computer Simulation,
  Vol. 8, No. 1, January 1998, pp 3--30.

For more information see the Mersenne Twister homepage
  [http://www.math.keio.ac.jp/~matumoto/emt.html]

Advantages:
-  large period (slightly less than 2**19937 -1)
-  relatively fast (slightly slower than TRandom2 but much faster than TRandom1)

Drawbacks:
-  a relative large internal state of 624 integers 
- generate only 32 random bits 
- not passing all the random generator tests. It fails some tests in TestU01
 (see [http://simul.iro.umontreal.ca/testu01/tu01.html])

An alternatively excellent generator passing all tests of TestU01, having 61 random bits and
being as fast as Mersenne and Twister is MIXMAX (TRandomMixMax).
Also, TRandomRanluxpp is a recommended alternative over TRandom3.

@warning TRandom3 is not a fully correct Mersenne and Twister random number generator, since
zeroes of the sequence are skipped, and thus the actual period is slightly less than
2**19937 -1. Consider using instead std::mt19937. Other differences are that, unlike in the paper, 0 (skip)
and 1 (here we divide by UINT_MAX + 1 instead of UINT_MAX - 1) are not included in the sequence.

@ingroup Random

*/

//////////////////////////////////////////////////////////////////////
// Aug.99 ROOT implementation based on CLHEP by P.Malzacher
//
// the original code contains the following copyright notice:
/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */
/* 02111-1307  USA                                                 */
/* Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.       */
/* When you use this, send an email to: matumoto@math.keio.ac.jp   */
/* with an appropriate reference to your work.                     */
/////////////////////////////////////////////////////////////////////

#include "TRandom3.h"
#include "TBuffer.h"
#include "TRandom2.h"
#include "TUUID.h"

TRandom *gRandom = new TRandom3();
#ifdef R__COMPLETE_MEM_TERMINATION
namespace {
   struct TRandomCleanup {
      ~TRandomCleanup() { delete gRandom; gRandom = nullptr; }
   };
   static TRandomCleanup gCleanupRandom;
}
#endif

ClassImp(TRandom3);

////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor.
///
/// If seed is 0, the seed array is automatically computed via a TRandom2
/// object, which internally uses TUUID.
/// In this case the seed is guaranteed to be unique in space and time.

TRandom3::TRandom3(UInt_t seed)
{
   SetName("Random3");
   SetTitle("Random number generator: Mersenne Twister");
   SetSeed(seed);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Default destructor.

TRandom3::~TRandom3()
{
}

////////////////////////////////////////////////////////////////////////////////
///  \brief Machine independent random number generator.
///
///  Produces uniformly-distributed floating points in ]0, 1[.
///  Method: Mersenne Twister
///  Generate number in interval (0,1): 0 and 1 are not included in the interval

Double_t TRandom3::Rndm()
{
   UInt_t y;

   const Int_t  kM = 397;
   const Int_t  kN = 624;
   const UInt_t kTemperingMaskB =  0x9d2c5680;
   const UInt_t kTemperingMaskC =  0xefc60000;
   const UInt_t kUpperMask =       0x80000000;
   const UInt_t kLowerMask =       0x7fffffff;
   const UInt_t kMatrixA =         0x9908b0df;

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

   // 2.3283064365386963e-10 == 1./(UINT_MAX+1UL)  -> then returned value cannot be = 1.0
   if (y) return ( (Double_t) y * 2.3283064365386963e-10); // * Power(2,-32)
   return Rndm();
}

////////////////////////////////////////////////////////////////////////////////
///  \brief Return an array of n random numbers uniformly distributed in ]0, 1[.

void TRandom3::RndmArray(Int_t n, Float_t *array)
{
  for(Int_t i=0; i<n; i++) array[i]=(Float_t)Rndm();
}

////////////////////////////////////////////////////////////////////////////////
///  \brief Return an array of n random numbers uniformly distributed in ]0, 1[.

void TRandom3::RndmArray(Int_t n, Double_t *array)
{
   Int_t k = 0;

   UInt_t y;

   const Int_t  kM = 397;
   const Int_t  kN = 624;
   const UInt_t kTemperingMaskB =  0x9d2c5680;
   const UInt_t kTemperingMaskC =  0xefc60000;
   const UInt_t kUpperMask =       0x80000000;
   const UInt_t kLowerMask =       0x7fffffff;
   const UInt_t kMatrixA =         0x9908b0df;

   while (k < n) {
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

      if (y) {
         array[k] = Double_t( y * 2.3283064365386963e-10); // * Power(2,-32)
         k++;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Set the random generator sequence.
///
/// If seed is 0 (default value) a TRandom2 (internally uses TUUID) is used to
/// generate all 624 unsigned integers of the seed array.
/// In this case the seed is guaranteed to be unique in space and time.
///
/// Upgraded seeding procedure is used to fix a known problem when seeding with
/// values with many zero in the bit pattern (like 2**28), see
/// http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html

void TRandom3::SetSeed(ULong_t seed)
{
   TRandom::SetSeed(seed);
   fCount624 = 624;
   if (seed > 0) {
      fMt[0] = fSeed;

      // use multipliers from  Knuth's "Art of Computer Programming" Vol. 2, 3rd Ed. p.106
      for(Int_t i=1; i<624; i++) {
         fMt[i] = (1812433253 * ( fMt[i-1]  ^ ( fMt[i-1] >> 30)) + i );
      }

   } else {

      // use TRandom2 (which is based on TUUID to generate the seed.
      // TRandom2 works fairly well  and has been tested against example
      // layout in https://savannah.cern.ch/bugs/?99516
      TRandom2 r(0);
      for (Int_t i = 0; i< 624; i++) {
         fMt[i]   = static_cast<UInt_t> (4294967296.*r.Rndm());
      }
      // warm up the generator calling it 10 times
      for (Int_t i = 0; i < 10; ++i) Rndm();
   }


}

////////////////////////////////////////////////////////////////////////////////
/// \brief Streamer for an object of class TRandom3.

void TRandom3::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TRandom3::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TRandom::Streamer(R__b);
      R__b.ReadStaticArray(fMt);
      R__b >> fCount624;
      R__b.CheckByteCount(R__s, R__c, TRandom3::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TRandom3::Class(),this);
   }
}
