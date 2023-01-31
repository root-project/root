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
#include <vector>
#include <limits>

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
   data contribution to the function which is required by some algorithm (like Fumili)

   @ingroup  FitMethodFunc
*/
template<class FunctionType>
class BasicFitMethodFunction : public FunctionType {

public:


   typedef  typename FunctionType::BaseFunc BaseFunction;

   /// enumeration specifying the possible fit method types
   enum Type_t { kUndefined , kLeastSquare, kLogLikelihood, kPoissonLikelihood };


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
      Estimating also the gradient of the data element if the passed pointer  is not null
      and the Hessian. The flag fullHessian is set when one needs to compute the full Hessian (not the approximated one)
      and should be used when the full second derivatives of the model functions are available
    */
   virtual double DataElement(const double *x, unsigned int i, double *g = nullptr, double *h = nullptr, bool fullHessian = false) const = 0;

   // flag to indicate if full Hessian computation is supported
   virtual bool HasHessian() const { return false;}

   /**
    * Computes the full Hessian. Return false if Hessian is not supported
    */
   virtual bool Hessian(const double * x, double * hess) const {
      //return full Hessian of  the objective function which is Sum(F(i))
      unsigned int np = NPoints();
      unsigned int ndim = NDim();
      unsigned int nh = ndim*(ndim+1)/2;
      for (unsigned int k = 0; k < nh;  ++k) {
         hess[k] = 0;
      }
      std::vector<double> g(np);  // gradient of the F(i)
      std::vector<double> h(nh);  // hessian of F(i)
      for (unsigned int i = 0; i < np; i++) {
         double f = DataElement(x,i,g.data(),h.data(),true);
         if (f == std::numeric_limits<double>::quiet_NaN() ) return false;
         for (unsigned int j = 0; j < nh; j++) {
            hess[j] += h[j];
         }
      }
      return true;
   }

   /**
    * Computes the Second derivatives. Return false if this is not supported
    */
   virtual bool G2(const double * , double * ) const { return false; }

   /**
      return the number of data points used in evaluating the function
    */
   virtual unsigned int NPoints() const { return fNPoints; }

   /**
      return the type of method, override if needed
    */
   virtual Type_t Type() const { return kUndefined; }

   /**
      return the total number of function calls (override if needed)
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


   /**
      Static function to indicate if a function is supporting gradient
   */
   static bool IsAGradFCN() {
     return false;
  }

private:

   unsigned int fNDim;      // function dimension
   unsigned int fNPoints;   // size of the data
   mutable unsigned int fNCalls; // number of function calls


};

template<>
inline bool BasicFitMethodFunction<ROOT::Math::IMultiGradFunction>::IsAGradFCN() {
   return true;
}

// define the normal and gradient function
typedef BasicFitMethodFunction<ROOT::Math::IMultiGenFunction>  FitMethodFunction;
typedef BasicFitMethodFunction<ROOT::Math::IMultiGradFunction> FitMethodGradFunction;



} // end namespace Math

} // end namespace ROOT






#endif /* ROOT_Math_FitMethodFunction */
