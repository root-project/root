// @(#)root/mathcore:$Id$
// Authors: L. Moneta    8/2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015 , ROOT MathLib Team                             *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file of MixMax engine
//
//
// Created by: Lorenzo Moneta  : Tue 4 Aug 2015
//
//
#include "Math/MixMaxEngine.h"

#include "mixmax.h"


namespace ROOT {
namespace Math {
   
   MixMaxEngine::MixMaxEngine(uint64_t seed) { 
      fRngState = rng_alloc();
      SetSeed64(seed); 
   }

   MixMaxEngine::~MixMaxEngine() { 
      rng_free(fRngState);
   }


   // void MixMaxEngine::SeedUniqueStream(unsigned int clusterID, unsigned int machineID, unsigned int runID, unsigned int  streamID) { 
   //    seed_uniquestream(fRngState, clusterID,  machineID,  runID,   streamID);
   // }

   void MixMaxEngine::SetSeed(unsigned int seed) { 
      seed_spbox(fRngState, seed);
      iterate(fRngState);                    
   }

   void MixMaxEngine::SetSeed64(uint64_t seed) { 
      seed_spbox(fRngState, seed);
      iterate(fRngState);                    
   }

   // unsigned int MixMaxEngine::GetSeed() const { 
   //    return get_next(fRngState);
   // }
         

   // generate one random number in interval ]0,1]
   double MixMaxEngine::Rndm_impl()  { 
      return get_next_float(fRngState);
   }

   // generate one integer number 
   uint64_t MixMaxEngine::IntRndm() { 
      return get_next(fRngState);
   }

                  
   void  MixMaxEngine::RndmArray(int n, double *array){
      // Return an array of n random numbers uniformly distributed in ]0,1]
      fill_array(fRngState, n,  array);
   }

   void MixMaxEngine::SetState(const std::vector<StateInt_t> & state, bool warmup) {
      if (fRngState) rng_free(fRngState);
      fRngState = rng_copy(const_cast<StateInt_t*>(state.data()) );
      if (warmup) iterate(fRngState); 
   }

   void MixMaxEngine::GetState(std::vector<StateInt_t> & state) const {
      int n =  rng_get_N(); 
      state.resize(n);
      for (int i = 0; i < n; ++i)
         state[i] = fRngState->V[i];
   }

   int MixMaxEngine::Size()  {
      return rng_get_N(); 
   }

   int MixMaxEngine::Counter() const {
      return fRngState->counter; 
   }

   void MixMaxEngine::SetCounter(int val) {
      fRngState->counter = val; 
   }

   // void MixMaxEngine::SetSpecialNumber(uint64_t /* val */ ) {
   //    //set_special_number(val); 
   // }


   
  
   } // namespace Math
} // namespace ROOT
