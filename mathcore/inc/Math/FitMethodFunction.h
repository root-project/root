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

//______________________________________________________________________________________
/** 
   FitMethodFunction class 
   Interface for objective functions (like chi2 and likelihood used in the fit)
   In addition to normal function interface provide interface for calculating each 
   data contrinution to the function which is required by some algorithm (like Fumili)

   @ingroup  CppFunctions
*/ 
template<class FunctionType>
class BasicFitMethodFunction : public FunctionType {

public:

   typedef  typename FunctionType::BaseFunc BaseFunction; 

   /// enumeration specyfing the possible fit method types
   enum Type { kUndefined , kLeastSquare, kLogLikelihood }; 

  

   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~BasicFitMethodFunction ()  {}  

   /**
      method returning the data i-th contribution to the fit objective function
      For example the residual for the least square functions or the pdf element for the 
      likelihood functions. 
      Estimating eventually also the gradient of the data element if the passed pointer  is not null
    */
   virtual double DataElement(const double *x, unsigned int i, double *g = 0) const = 0; 


   /**
      return the number of data points used in evaluating the function
    */
   virtual unsigned int NPoints() const = 0; 

   /**
      return the type of method, override if needed
    */
   virtual Type GetType() const { return kUndefined; }

   /**
      return the total number of function calls (overrided if needed)
    */
   virtual unsigned int NCalls() const { return 0; }

public: 


protected: 


private: 


}; 

      // define the normal and gradient function
      typedef BasicFitMethodFunction<ROOT::Math::IMultiGenFunction>  FitMethodFunction;      
      typedef BasicFitMethodFunction<ROOT::Math::IMultiGradFunction> FitMethodGradFunction;


   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_FitMethodFunction */
