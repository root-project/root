#ifndef ROOT_Math_MixMaxEngineImpl
#define ROOT_Math_MixMaxEngineImpl


namespace {


#if (_N==256)
#include "mixmax_oldS.icc"
  
#else

#include "mixmax.icc"
   
#endif
#undef N
}

#include "Math/MixMaxEngine.h"

namespace ROOT {
   namespace Math { 

template<> 
class MixMaxEngineImpl<_N> {
   rng_state_t * fRngState;
public:

   typedef uint64_t StateInt_t; 
   typedef uint64_t result_t; 
   
   MixMaxEngineImpl(uint64_t seed) {
      fRngState = rng_alloc();
      SetSeed(seed); 
   }
   ~MixMaxEngineImpl() {
      rng_free(fRngState);
   }
   void SetSeedFast(result_t seed) {
      seed_spbox(fRngState, seed);
   }
   void SetSeed(result_t seed) { 
      //seed_spbox(fRngState, seed);
      seed_uniquestream(fRngState, 0, 0, (uint32_t)(seed>>32), (uint32_t)seed );
   }
   double Rndm() {
       return get_next_float(fRngState);
   }
   // generate one integer number 
   result_t IntRndm() { 
      return get_next(fRngState);
   }
   void SetState(const std::vector<StateInt_t> & state) {
      if (fRngState) rng_free(fRngState);
      fRngState = rng_copy(const_cast<StateInt_t*>(state.data()) );
   }
   void GetState(std::vector<StateInt_t> & state) const {
      int n =  rng_get_N(); 
      state.resize(n);
      for (int i = 0; i < n; ++i)
         state[i] = fRngState->V[i];
   }
   void Iterate() {
      iterate(fRngState); 
   }
   int Counter() const {
      return fRngState->counter; 
   }
   void SetCounter(int val) {
      fRngState->counter = val; 
   }
   static int Size()  {
      return rng_get_N(); 
   }

   // to silent some warning
   void RndmArray(int n, double * array) {
      fill_array(fRngState, n, array); 
   }
   void ReadState(const char filename[] ) {
      read_state(fRngState, filename);
   }
   // branch generator given a vector of seed (at least 4 32 bit values)
   void Branch(uint32_t * seedvec) {
      branch_inplace(fRngState, seedvec); 
   }
     
   
};

      
   } // end namesapce Math
} // end namespace ROOT

#endif
