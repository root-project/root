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

#include "Math/IFunction.h"

// #ifndef ROOT_Math_IParamFunctionfwd
// #include "Math/IParamFunctionfwd.h"
// #endif

namespace ROOT {

   namespace Math {

//______________________________________________________________________________________
/**
   FitMethodFunction class
   Interface for objective functions (like chi2 and likelihood used in the fit)
   In addition to normal function interface provide interface for calculating each
   data contrinution to the function which is required by some algorithm (like Fumili)

   @ingroup  FitMethodFunc
*/
template<class FunctionType>
class BasicFitMethodFunction : public FunctionType {

public:


   typedef  typename FunctionType::BaseFunc BaseFunction;

   /// enumeration specyfing the possible fit method types
   enum Type_t { kUndefined , kLeastSquare, kLogLikelihood };


   BasicFitMethodFunction(int dim, int npoint) :
      fNDim(dim),
      fNPoints(npoint),
      fNCalls(0)
   {}

   /**
      Virtual Destructor (no operations)
   */
   virtual ~BasicFitMethodFunction ()  {}

   /**
      Number of dimension (parameters) . From IGenMultiFunction interface
    */
   virtual unsigned int NDim() const { return fNDim; }

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
   virtual unsigned int NPoints() const { return fNPoints; }

   /**
      return the type of method, override if needed
    */
   virtual Type_t Type() const { return kUndefined; }

   /**
      return the total number of function calls (overrided if needed)
    */
   virtual unsigned int NCalls() const { return fNCalls; }

   /**
      update number of calls
    */
   virtual void UpdateNCalls() const { fNCalls++; }

   /**
      reset number of function calls
    */
   virtual void ResetNCalls() { fNCalls = 0; }



public:


protected:


private:

   unsigned int fNDim;      // function dimension
   unsigned int fNPoints;   // size of the data
   mutable unsigned int fNCalls; // number of function calls


};

      // define the normal and gradient function
      typedef BasicFitMethodFunction<ROOT::Math::IMultiGenFunction>  FitMethodFunction;
      typedef BasicFitMethodFunction<ROOT::Math::IMultiGradFunction> FitMethodGradFunction;


      // useful template definition to use these interface in
      // generic programming
      // (comment them out since they are not used anymore)
/*
      template<class FunType>
      struct ParamFunctionTrait {
         typedef  IParamMultiFunction PFType;
      };

      // specialization for the gradient param functions
      template<>
      struct ParamFunctionTrait<ROOT::Math::IMultiGradFunction>  {
         typedef  IParamMultiGradFunction PFType;
      };
*/


   } // end namespace Math

} // end namespace ROOT






#endif /* ROOT_Math_FitMethodFunction */
