// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Aug 4 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// random engines based on ROOT 

#ifndef ROOT_Math_MixMaxEngine
#define ROOT_Math_MixMaxEngine

#include <cstdint>
#include <vector>

#ifndef ROOT_Math_TRandomEngine
#include "Math/TRandomEngine.h"
#endif


struct rng_state_st;    /// forward declare generator state

typedef struct rng_state_st rng_state_t;


namespace ROOT {

   namespace Math {
      

      /**
         MIXMAX Random number generator.
         It is a matrix-recursive random number generator introduced by 
         G. Savvidy in N.Z.Akopov, G.K.Savvidy and N.G.Ter-Arutyunian, _Matrix Generator of Pseudorandom Numbers_, 
	J.Comput.Phys. 97, 573 (1991) [DOI Link](http://dx.doi.org/10.1016/0021-9991(91)90016-E). 
        This is a new very fast impelmentation by K. Savvidy
        by K. Savvidy and described in this paper, 
        K. Savvidy, _The MIXMAX Random Number Generator_, Comp. Phys. Communic. (2015)
        [DOI link](http://dx.doi.org/10.1016/j.cpc.2015.06.003)
        
        The period of the generator is 10^4682 for N=256, and
        10^1597 for N=88

        This implementation is only a wrapper around the real implemention, see mixmax.cxx and mixmax.h
        The generator, in C code, is available also at hepforge: http://mixmax.hepforge.org

         
         @ingroup Random
      */

      
      class MixMaxEngine : public TRandomEngine {


      public:

         typedef  TRandomEngine BaseType;

         // this should be changed for WINDOWS
         typedef unsigned long long int StateInt_t;

         
         
         MixMaxEngine(uint64_t seed=1);

         virtual ~MixMaxEngine();

         /// get the state of the generator
         void GetState(std::vector<StateInt_t> & state) const;

         /// Get the counter (between 0 and Size-1)
         int Counter() const;

         /// Get the size of the generator
         static int Size();

         /// maximum integer that can be generated. For MIXMAX is 2^61-1         
         static uint64_t MaxInt() { return  0x1fffffffffffffff; } //  2^61 -1 

         /// set the generator seed 
         void  SetSeed(unsigned int seed);

         /// set the generator seed using a 64 bits integer
         void SetSeed64(uint64_t seed);

         ///set the full initial generator state and warm up generator by doing some iterations
         void SetState(const std::vector<StateInt_t> & state, bool warmup = true);

         /// set the counter
         void SetCounter(int val);

         // /// set the special number 
         // static void SetSpecialNumber(uint64_t val);
         
         /// set the number we want to use to skip generation
         /// higher value means higher luxury but slower
         static void SetSkipNumber(int /*nskip */) {  } // not implemented

         /// set initial number to be used in the vector.
         /// The previous elements are skipped and not returned.
         static void SetFirstReturnElement(int /*index */) {  } // not implemented



         // generate a random number (virtual interface)
         virtual double Rndm() { return Rndm_impl(); }

         /// generate a double random number (faster interface)
         inline double operator() () { return Rndm_impl(); }

         /// generate an array of random numbers 
         void RndmArray (int n, double * array); 

         /// generate a 64  bit integer number
         uint64_t IntRndm();

      private:

         /// implementation function to generrate the random number
         double Rndm_impl();

         rng_state_t * fRngState;  // mix-max generator state
         
      };


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_TRandomEngines */
