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

#ifndef ROOT_Math_TRandomEngine
#include "Math/TRandomEngine.h"
#endif

#include <stdint.h>

namespace ROOT {

   namespace Math {

      /**
         Random number generator class based on
         M. Matsumoto and T. Nishimura,
         Mersenne Twister: A 623-diminsionally equidistributed
         uniform pseudorandom number generator
         ACM Transactions on Modeling and Computer Simulation,
         Vol. 8, No. 1, January 1998, pp 3--30.
         
         For more information see the Mersenne Twister homepage
         [http://www.math.keio.ac.jp/~matumoto/emt.html]
         
         Advantage: 
         
         -  large period 2**19937 -1
         -  relativly fast (slightly slower than TRandom1 and TRandom2 but much faster than TRandom1)
         
         Drawback:  a relative large internal state of 624 integers
         
         @ingroup Random
      */
      
      class MersenneTwisterEngine : public TRandomEngine {


      public:

         typedef  TRandomEngine BaseType; 
         
         MersenneTwisterEngine(unsigned int seed=4357)  {
            SetSeed(seed);
         }

         virtual ~MersenneTwisterEngine() {}

         void SetSeed(unsigned int seed);


         virtual double Rndm() {
            return Rndm_impl();
         }
         inline double operator() () { return Rndm_impl(); }

         unsigned int IntRndm() {
            // fSeed = (1103515245 * fSeed + 12345) & 0x7fffffffUL;
            // return fSeed;
            return (int) Rndm_impl(); // t.b. impl
         }

      private:

         double Rndm_impl();

         
         uint32_t  fMt[624];
         int fCount624;
      };
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_TRandomEngines */
