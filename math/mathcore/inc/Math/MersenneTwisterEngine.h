// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Aug 4 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// random engines based on ROOT

#ifndef ROOT_Math_MersenneTwisterEngine
#define ROOT_Math_MersenneTwisterEngine

#include "Math/TRandomEngine.h"

#include <cstdint>
#include <vector>
#include <string>

namespace ROOT {

   namespace Math {

      /**
         Random number generator class based on
         M. Matsumoto and T. Nishimura,
         Mersenne Twister: A 623-dimensionally equidistributed
         uniform pseudorandom number generator
         ACM Transactions on Modeling and Computer Simulation,
         Vol. 8, No. 1, January 1998, pp 3--30.

         For more information see the Mersenne Twister homepage
         [http://www.math.keio.ac.jp/~matumoto/emt.html]

         Advantage:

         -  large period 2**19937 -1
         -  relatively fast (slightly slower than TRandom1 and TRandom2 but much faster than TRandom1)

         Note that this is a 32 bit implementation. Only 32 bits of the returned double numbers are random.
         in case more precision is needed, one should use an engine providing at least 48 random bits.

         Drawback:  a relative large internal state of 624 integers

         @ingroup Random
      */

      class MersenneTwisterEngine : public TRandomEngine {


      public:

         typedef  TRandomEngine BaseType;
         typedef  uint32_t Result_t;
         typedef  uint32_t StateInt_t;


         MersenneTwisterEngine(uint32_t seed=4357)  {
            SetSeed(seed);
         }

         ~MersenneTwisterEngine() override {}

         void SetSeed(Result_t seed);

         double Rndm() override {
            return Rndm_impl();
         }
         inline double operator() () { return Rndm_impl(); }

         uint32_t IntRndm() {
            return IntRndm_impl();
         }

         /// minimum integer taht can be generated
         static unsigned int MinInt() { return 0; }
         /// maximum integer that can be generated
         static unsigned int MaxInt() { return 0xffffffff; }  //  2^32 -1

         static int Size() { return kSize; }

         static std::string Name() {
            return "MersenneTwisterEngine";
         }

      protected:
         // functions used for testing

         void SetState(const std::vector<uint32_t> & state) {
            for (unsigned int i = 0; i < kSize; ++i)
               fMt[i] = state[i];
            fCount624 = kSize; // to make sure we re-iterate on the new state
         }

         void GetState(std::vector<uint32_t> & state) {
            state.resize(kSize);
            for (unsigned int i = 0; i < kSize; ++i)
               state[i] = fMt[i];
         }

         int Counter() const { return fCount624; }


      private:

         double Rndm_impl();
         uint32_t IntRndm_impl();

         enum {
            kSize=624
         };
         uint32_t  fMt[kSize];
         int fCount624;
      };


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_TRandomEngines */
