// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_Numerical2PGradientCalculator
#define ROOT_Minuit2_Numerical2PGradientCalculator

#include "Minuit2/MnConfig.h"

#include "Minuit2/GradientCalculator.h"

namespace ROOT {

namespace Minuit2 {

class MnFcn;
class MnUserTransformation;
class MnStrategy;

/**
   class performing the numerical gradient calculation
 */

class Numerical2PGradientCalculator : public GradientCalculator {

public:
   Numerical2PGradientCalculator(const MnFcn &fcn, const MnUserTransformation &par, const MnStrategy &stra)
      : fFcn(fcn), fTransformation(par), fStrategy(stra)
   {
   }

   FunctionGradient operator()(const MinimumParameters &) const override;

   FunctionGradient operator()(const MinimumParameters &, const FunctionGradient &) const override;

private:
   const MnFcn &fFcn;
   const MnUserTransformation &fTransformation;
   const MnStrategy &fStrategy;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_Numerical2PGradientCalculator
