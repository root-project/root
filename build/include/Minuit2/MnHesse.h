// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnHesse
#define ROOT_Minuit2_MnHesse

#include "Minuit2/MnConfig.h"
#include "Minuit2/MnStrategy.h"

#include <ROOT/RSpan.hxx>

#include <vector>

namespace ROOT {

namespace Minuit2 {

class FCNBase;
class MnUserParameterState;
class MnUserParameters;
class MnUserCovariance;
class MnUserTransformation;
class MinimumState;
class MnMachinePrecision;
class MnFcn;
class FunctionMinimum;

//_______________________________________________________________________
/**
    API class for calculating the numerical covariance matrix
    (== 2x Inverse Hessian == 2x Inverse 2nd derivative); can be used by the
    user or Minuit itself
 */

class MnHesse {

public:
   /// default constructor with default strategy
   MnHesse() : fStrategy(MnStrategy(1)) {}

   /// constructor with user-defined strategy level
   MnHesse(unsigned int stra) : fStrategy(MnStrategy(stra)) {}

   /// conctructor with specific strategy
   MnHesse(const MnStrategy &stra) : fStrategy(stra) {}

   /// FCN + MnUserParameterState
   MnUserParameterState operator()(const FCNBase &, const MnUserParameterState &, unsigned int maxcalls = 0) const;
   ///
   /// API to use MnHesse after minimization when function minimum is avalilable, otherwise information on the last
   /// state will be lost. (It would be needed to re-call the gradient and spend extra useless function calls) The
   /// Function Minimum is updated (modified) by adding the Hesse results as last state of minimization
   ///
   void operator()(const FCNBase &, FunctionMinimum &, unsigned int maxcalls = 0) const;

   /// internal interface
   ///
   MinimumState
   operator()(const MnFcn &, const MinimumState &, const MnUserTransformation &, unsigned int maxcalls = 0) const;

   /// forward interface of MnStrategy
   unsigned int Ncycles() const { return fStrategy.HessianNCycles(); }
   double Tolerstp() const { return fStrategy.HessianStepTolerance(); }
   double TolerG2() const { return fStrategy.HessianG2Tolerance(); }

private:
   MnStrategy fStrategy;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnHesse
