// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_AnalyticalGradientCalculator
#define ROOT_Minuit2_AnalyticalGradientCalculator

#include "Minuit2/GradientCalculator.h"
#include "Minuit2/MnMatrixfwd.h"

namespace ROOT {

namespace Minuit2 {

class FCNGradientBase;
class MnUserTransformation;


class AnalyticalGradientCalculator : public GradientCalculator {

public:
   AnalyticalGradientCalculator(const FCNGradientBase &fcn, const MnUserTransformation &state)
      : fGradFunc(fcn), fTransformation(state)
   {
   }

   ~AnalyticalGradientCalculator() override {}

   FunctionGradient operator()(const MinimumParameters &) const override;

   FunctionGradient operator()(const MinimumParameters &, const FunctionGradient &) const override;

   /// compute Hessian matrix
   bool Hessian(const MinimumParameters &, MnAlgebraicSymMatrix &) const override;

   /// compute second derivatives (diagonal of Hessian)
   virtual bool G2(const MinimumParameters &, MnAlgebraicVector &) const;

   virtual bool CheckGradient() const;

   virtual bool CanComputeG2() const;

   virtual bool CanComputeHessian() const;

protected:
   const FCNGradientBase &fGradFunc;
   const MnUserTransformation &fTransformation;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_AnalyticalGradientCalculator
