// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Aug 4 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// random engines based on ROOT 

#ifndef ROOT_Math_StdEngine
#define ROOT_Math_StdEngine

#include <random>
#include <string>

namespace ROOT {

   namespace Math {


      class StdRandomEngine {};

      template<class Generator>
      struct StdEngineType {
         static std::string Name() { return "std_random_eng";}
      };
      template<>
         struct StdEngineType<std::minstd_rand> {
         static std::string Name() { return "std_minstd_rand";}
      };
      template<>
      struct StdEngineType<std::mt19937> {
         static std::string Name() { return "std_mt19937";}
      };
      template<>
      struct StdEngineType<std::mt19937_64> {
         static std::string Name() { return "std_mt19937_64";}
      };
      template<>
      struct StdEngineType<std::ranlux24> {
         static std::string Name() { return "std_ranlux24";}
      };
      template<>
      struct StdEngineType<std::ranlux48> {
         static std::string Name() { return "std_ranlux48";}
      };
      template<>
      struct StdEngineType<std::knuth_b> {
         static std::string Name() { return "std_knuth_b";}
      };
      template<>
      struct StdEngineType<std::random_device> {
         static std::string Name() { return "std_random_device";}
      };
      
      
      /** 
          Wrapper class for std::random generator to be included in ROOT 
      */
      
      template <class Generator> 
      class StdEngine {


      public:

         typedef  StdRandomEngine BaseType; 
         typedef  typename Generator::result_type result_t; 
         
         StdEngine() : fGen() {
            fCONS = 1./fGen.max(); 
         }

         
         void SetSeed(result_t seed) { fGen.seed(seed);}
         
         double Rndm() {
            result_t rndm = fGen(); // generate integer number according to the type 
            if (rndm != 0) return  fCONS*rndm;
            return Rndm();
         }

         result_t IntRndm() {
            return fGen();
         }

         double operator() () {
            return Rndm(); 
         }

         static std::string Name()  {
            return StdEngineType<Generator>::Name(); 
         }

         static uint64_t MaxInt() { return Generator::max(); }


      private:
         Generator fGen;
         double fCONS;   //! cached value of maximum integer value generated
      };


      extern template class StdEngine<std::mt19937_64>;
      extern template class StdEngine<std::ranlux48>;

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_StdEngine */
