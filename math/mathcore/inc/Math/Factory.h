// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Dec 22 14:43:33 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class Factory

#ifndef ROOT_Math_Factory
#define ROOT_Math_Factory

#include <string>


namespace ROOT { 

   namespace Math { 

   class Minimizer; 
   
//___________________________________________________________________________
/** 
   Factory  class holding static functions to create the interfaces like ROOT::Math::Minimizer
   via the Plugin Manager
*/ 
class Factory { 
      public: 

   /**
      static method to create the corrisponding Minimizer given the string
    */
   static ROOT::Math::Minimizer * CreateMinimizer(const std::string & minimizerType = "Minuit2", const std::string & algoType = "Migrad");
   

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_MinimizerFactory */
