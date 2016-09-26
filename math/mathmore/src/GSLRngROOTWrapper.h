// wrapper class used to wrap ROOT random number engines with GSL interface

// @(#)root/mathmore:$Id$
// Author: L. Moneta Fri Aug 24 17:20:45 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class GSLRngWrapper

#ifndef ROOT_Math_GSLRngROOTWrapper
#define ROOT_Math_GSLRngROOTWrapper

#include "gsl/gsl_rng.h"


namespace ROOT {

   namespace Math {

      template<class Engine>
      struct GSLRngROOTWrapper {

         Engine * fEngine;
         bool fFirst;

         
         GSLRngROOTWrapper() {
            printf("constructing the ROOT GSL wrapper engine -size = %d\n",Size());
            fEngine = nullptr;
            fFirst = false; 
         }

         ~GSLRngROOTWrapper() {
            if (fEngine) delete fEngine; 
         }

         static double Rndm(void * p) {
            return ((GSLRngROOTWrapper *) p)->fEngine->operator()(); 
         }
         static unsigned long IntRndm(void * p) {
            return ((GSLRngROOTWrapper *) p)->fEngine->IntRndm(); 
         }
         static void Seed(void * p, unsigned long seed) {
            //if (fFirst) {
            // this will be a memory leak because I have no way to delete
            // the engine class 
            auto wr = ((GSLRngROOTWrapper *) p);
            if (wr->fFirst) {
               //printf("calling the seed function with %d on %p . Build Engine class \n",seed,p);               
               wr->fEngine = new Engine();
               wr->fFirst = false;
            }
            // the seed cannot be zero (GSL calls at the beginning with seed 0)
            if (seed == 0) return; 
            ((GSLRngROOTWrapper *) p)->fEngine->SetSeed(seed); 
         }
         static void Free(void *p) {
            auto wr = ((GSLRngROOTWrapper *) p);
            if (wr->fEngine) delete wr->fEngine;
            wr->fFirst = true;
            //printf("deleting gsl mixmax\n");
         }
         
         static unsigned long Max() { return Engine::MaxInt(); }
         static unsigned long Min() { return Engine::MinInt(); }
         static size_t Size() { return sizeof( GSLRngROOTWrapper<Engine>); }
         static std::string Name() { return std::string("GSL_")+Engine::Name(); }
      };
   }
}

#include "Math/MixMaxEngine.h"

// now define and implement the specific types    

typedef ROOT::Math::GSLRngROOTWrapper<ROOT::Math::MixMaxEngine<240,0>> GSLMixMaxWrapper;

static const gsl_rng_type mixmax_type =
{
   GSLMixMaxWrapper::Name().c_str(), 
   GSLMixMaxWrapper::Max(), 
   GSLMixMaxWrapper::Min(),
   GSLMixMaxWrapper::Size(),
   &GSLMixMaxWrapper::Seed,
   &GSLMixMaxWrapper::IntRndm,
   &GSLMixMaxWrapper::Rndm
};

const gsl_rng_type *gsl_rng_mixmax = &mixmax_type;

#endif

