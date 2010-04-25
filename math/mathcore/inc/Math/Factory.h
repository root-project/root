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
   class DistSampler;   
   
//___________________________________________________________________________
/** 
   Factory  class holding static functions to create the interfaces like ROOT::Math::Minimizer
   via the Plugin Manager
*/ 
class Factory { 
      public: 

   /**
      static method to create the corrisponding Minimizer given the string
      Supported Minimizers types are: 
      Minuit (TMinuit), Minuit2, GSLMultiMin, GSLMultiFit, GSLSimAn, Linear, Fumili, Genetic
    */
   static ROOT::Math::Minimizer * CreateMinimizer(const std::string & minimizerType = "Minuit2", const std::string & algoType = "Migrad");

   /**
      static method to create the distribution sampler class given a string specifying the type
      Supported sampler types are: 
      Unuran
    */
   static ROOT::Math::DistSampler * CreateDistSampler(const std::string & samplerType ="Unuran");
   

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_MinimizerFactory */
