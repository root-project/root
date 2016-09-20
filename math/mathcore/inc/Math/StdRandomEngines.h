// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Aug 4 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// random engines based on ROOT 

#ifndef ROOT_Math_StdRandomEngines
#define ROOT_Math_StdRandomEngines

#include <random> 


namespace ROOT {

   namespace Math {


      class StdRandomEngine {};

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

         static uint64_t MaxInt() { return Generator::max(); }

      private:
         Generator fGen;
         double fCONS;   //! cached value of maximum integer value generated
      };
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_TRandomEngines */
