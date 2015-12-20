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
         
         StdEngine() : fGen() { }

         void SetSeed(unsigned int seed) { fGen.seed(seed);}
         
         double Rndm() {
            const double kCONS = 4.6566128730774E-10; // (1/pow(2,31)
            unsigned int rndm = fGen(); // generate integer number 
            if (rndm != 0) return  kCONS*rndm;
            return Rndm();
         }

         unsigned int operator() () {
            return fGen(); 
         }

      private:
         Generator fGen; 
      };
      

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_TRandomEngines */
