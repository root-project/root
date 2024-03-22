// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_InitialGradientCalculator
#define ROOT_Minuit2_InitialGradientCalculator

#include "Minuit2/GradientCalculator.h"

namespace ROOT {

namespace Minuit2 {

class MnFcn;
class MnUserTransformation;
class MnMachinePrecision;

/**
   Class to calculate an initial estimate of the gradient
 */
class InitialGradientCalculator : public GradientCalculator {

public:
   InitialGradientCalculator(const MnFcn &fcn, const MnUserTransformation &par) : fFcn(fcn), fTransformation(par) {}

   FunctionGradient operator()(const MinimumParameters &) const override;

   FunctionGradient operator()(const MinimumParameters &, const FunctionGradient &) const override;

   const MnFcn &Fcn() const { return fFcn; }
   const MnUserTransformation &Trafo() const { return fTransformation; }
   const MnMachinePrecision &Precision() const;

private:
   const MnFcn &fFcn;
   const MnUserTransformation &fTransformation;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_InitialGradientCalculator
