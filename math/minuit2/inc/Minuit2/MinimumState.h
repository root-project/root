// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MinimumState
#define ROOT_Minuit2_MinimumState

#include "Minuit2/MinimumParameters.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/FunctionGradient.h"

#include <memory>

namespace ROOT {

namespace Minuit2 {

/** MinimumState keeps the information (position, Gradient, 2nd deriv, etc)
    after one minimization step (usually in MinimumBuilder).
 */

class MinimumState {

public:
   /// Invalid state.
   MinimumState(unsigned int n) : MinimumState(MinimumParameters(n, 0.0), MinimumError(n), FunctionGradient(n), 0.0, 0)
   {
   }

   /// Constructor without parameter values, but with function value, edm and nfcn.
   /// This constructor will result in a state that is flagged as not valid
   MinimumState(double fval, double edm, int nfcn)
      : MinimumState(MinimumParameters(0, fval), MinimumError(0), FunctionGradient(0), edm, nfcn)
   {
   }

   /// Constuctor with only parameter values, edm and nfcn, but without errors (covariance).
   /// The resulting state it will be considered valid, since it contains the parameter values,
   /// although it will has not the error matrix (MinimumError) with
   /// HasCovariance() returning false.
   MinimumState(const MinimumParameters &states, double edm, int nfcn)
      : MinimumState(states, MinimumError(states.Vec().size()), FunctionGradient(states.Vec().size()), edm, nfcn)
   {
   }

   /// Constructor with parameters values, errors and gradient
   MinimumState(const MinimumParameters &states, const MinimumError &err, const FunctionGradient &grad, double edm,
                int nfcn)
      : fPtr{new Data{states, err, grad, edm, nfcn}}
   {
   }

   const MinimumParameters &Parameters() const { return fPtr->fParameters; }
   const MnAlgebraicVector &Vec() const { return Parameters().Vec(); }
   int size() const { return Vec().size(); }

   const MinimumError &Error() const { return fPtr->fError; }
   const FunctionGradient &Gradient() const { return fPtr->fGradient; }
   double Fval() const { return Parameters().Fval(); }
   double Edm() const { return fPtr->fEDM; }
   int NFcn() const { return fPtr->fNFcn; }

   bool IsValid() const
   {
      if (HasParameters() && HasCovariance())
         return Parameters().IsValid() && Error().IsValid();
      else if (HasParameters())
         return Parameters().IsValid();
      else
         return false;
   }

   bool HasParameters() const { return Parameters().IsValid(); }
   bool HasCovariance() const { return Error().IsAvailable(); }

private:
   struct Data {
      MinimumParameters fParameters;
      MinimumError fError;
      FunctionGradient fGradient;
      double fEDM;
      int fNFcn;
   };

   std::shared_ptr<Data> fPtr;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MinimumState
