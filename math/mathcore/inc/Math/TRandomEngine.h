// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Aug 4 2015

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2015  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// interface for random engines based on ROOT 

#ifndef ROOT_Math_TRandomEngine
#define ROOT_Math_TRandomEngine

namespace ROOT {
   namespace Math{

      class TRandomEngine  {
      public:
         virtual double Rndm() = 0;
         virtual ~TRandomEngine() {}
      };
      
   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_TRandomEngine */
