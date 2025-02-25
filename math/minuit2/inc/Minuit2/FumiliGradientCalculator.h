// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FumiliGradientCalculator
#define ROOT_Minuit2_FumiliGradientCalculator

#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/MnMatrix.h"

namespace ROOT {

namespace Minuit2 {

class FumiliFCNBase;
class MnUserTransformation;

/// Fumili gradient calculator using external gradient provided by FCN
/// Note that the computed Hessian and G2 are an approximation valid for small residuals
class FumiliGradientCalculator : public AnalyticalGradientCalculator {

public:
   FumiliGradientCalculator(const FumiliFCNBase &fcn, const MnUserTransformation &trafo, int n);

   FunctionGradient operator()(const MinimumParameters &) const override;

   FunctionGradient operator()(const MinimumParameters &, const FunctionGradient &) const override;

   const MnUserTransformation &Trafo() const { return fTransformation; }

   const MnAlgebraicSymMatrix &GetHessian() const { return fHessian; }

   bool Hessian(const MinimumParameters &, MnAlgebraicSymMatrix &) const override;

   bool G2(const MinimumParameters &, MnAlgebraicVector &) const override;

   bool CanComputeG2() const override { return true;}

   bool CanComputeHessian() const override { return true;}


private:
   const FumiliFCNBase &fFcn;
   mutable MnAlgebraicSymMatrix fHessian;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_FumiliGradientCalculator
