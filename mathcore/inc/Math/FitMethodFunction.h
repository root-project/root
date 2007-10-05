// @(#)root/mathcore:$Id$
// Author: L. Moneta Thu Aug 16 15:40:28 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class FitMethodFunction

#ifndef ROOT_Math_FitMethodFunction
#define ROOT_Math_FitMethodFunction

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

namespace ROOT { 

   namespace Math { 


/** 
   FitMethodFunction class 
   Interface for objective functions (like chi2 and likelihood used in the fit)
   In addition to normal function interface provide interface for calculating each 
   data contrinution to the function which is required by some algorithm (like Fumili)

   @ingroup  CppFunctions
*/ 
class FitMethodFunction : public ROOT::Math::IMultiGenFunction {

public:

   typedef  ROOT::Math::IMultiGenFunction BaseFunction; 
   


   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~FitMethodFunction ()  {}  

   /**
      method returning the data i-th contribution to the fit objective function
      For example the residual for the chi2 
    */
   virtual double DataElement(const double *x, unsigned int i) const = 0; 

   /**
      return the number of data points used in evaluating the function
    */
   virtual unsigned int NPoints() const = 0; 

public: 


protected: 


private: 


}; 

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_FitMethodFunction */
