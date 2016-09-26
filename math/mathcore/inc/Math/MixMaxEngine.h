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
#include <string>

#ifndef ROOT_Math_TRandomEngine
#include "Math/TRandomEngine.h"
#endif


// struct rng_state_st;    /// forward declare generator state

// typedef struct rng_state_st rng_state_t;

// namespace mixmax { 
//    template<int Ndim>
//    class mixmax_engine; 
// }

namespace ROOT {

   namespace Math {
      
      template<int N>
      class MixMaxEngineImpl;

      /**
         Wrapper class for the MIXMAX Random number generator.
         It is a matrix-recursive random number generator introduced by
         G. Savvidy.

         See the real implementation in the mixmax.h and mixmax.cxx files.
         The generator code is available also at hepforge: http://mixmax.hepforge.org


         Created by Konstantin Savvidy.

         The code is released under GNU Lesser General Public License v3

         References:

         G.K.Savvidy and N.G.Ter-Arutyunian,
         On the Monte Carlo simulation of physical systems,
         J.Comput.Phys. 97, 566 (1991);
         Preprint EPI-865-16-86, Yerevan, Jan. 1986

         K.Savvidy
         The MIXMAX random number generator
         Comp. Phys. Commun. 196 (2015), pp 161–165
         http://dx.doi.org/10.1016/j.cpc.2015.06.003

         K.Savvidy and G.Savvidy
         Spectrum and Entropy of C-systems. MIXMAX random number generator
         Chaos, Solitons & Fractals, Volume 91, (2016) pp. 33–38
         http://dx.doi.org/10.1016/j.chaos.2016.05.003

         The period of the generator is 10^4682 for N=256, and
         10^1597 for N=88

         This implementation is only a wrapper around the real implemention, see mixmax.cxx and mixmax.h
         The generator, in C code, is available also at hepforge: http://mixmax.hepforge.org


         @ingroup Random
      */

      template<int N, int SkipNumber>
      class MixMaxEngine : public TRandomEngine {

      public:

         typedef  TRandomEngine BaseType;

         // this should be changed for WINDOWS
         typedef uint64_t StateInt_t;
         typedef uint64_t result_t;


         MixMaxEngine(uint64_t seed=1);

         virtual ~MixMaxEngine();


         /// Get the size of the generator
         static int Size();

         /// maximum integer that can be generated. For MIXMAX is 2^61-1
         static uint64_t MaxInt();

         /// minimum integer that can be generated. For MIXMAX is 0
         static uint64_t MinInt();

         /// set the generator seed
         void  SetSeed(result_t seed);

         // generate a random number (virtual interface)
         virtual double Rndm() { return Rndm_impl(); }

         /// generate a double random number (faster interface)
         inline double operator() () { return Rndm_impl(); }

         /// generate an array of random numbers
         void RndmArray (int n, double * array);

         /// generate a 64  bit integer number
         result_t IntRndm();

         /// get name of the generator
         static std::string Name(); 

      protected:
         // protected functions used for tesing the generator

         /// get the state of the generator
         void GetState(std::vector<StateInt_t> & state) const;

         ///set the full initial generator state 
         void SetState(const std::vector<StateInt_t> & state);

         /// Get the counter (between 0 and Size-1)
         int Counter() const;


      private:

         /// implementation function to generate the random number
         double Rndm_impl();

         //rng_state_t * fRngState;  // mix-max generator state
         //mixmax::mixmax_engine<N> * fRng;  // mixmax internal engine class
         MixMaxEngineImpl<N> * fRng;  // mixmax internal engine class
         
      };

      typedef MixMaxEngine<240,0> MixMaxEngine240;
      typedef MixMaxEngine<256,2> MixMaxEngine256;
      typedef MixMaxEngine<17,0> MixMaxEngine17;

      extern template class MixMaxEngine<240,0>;
      extern template class MixMaxEngine<256,2>;
      extern template class MixMaxEngine<17,0>;
      extern template class MixMaxEngine<17,1>;

   } // end namespace Math

} // end namespace ROOT


#include "Math/MixMaxEngine.icc"

#endif /* ROOT_Math_MixMaxEngine */ 
