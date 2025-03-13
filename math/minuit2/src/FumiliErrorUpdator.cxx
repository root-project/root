// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/FumiliErrorUpdator.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/FumiliGradientCalculator.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MnMatrix.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MnPrint.h"

#include <limits>

namespace ROOT {

namespace Minuit2 {

MinimumError
FumiliErrorUpdator::Update(const MinimumState &s0, const MinimumParameters &p1, const FunctionGradient &g1) const
{
   // dummy methods to suppress unused variable warnings
   // this member function should never be called within
   // the Fumili method...
   s0.Fval();
   p1.Fval();
   g1.IsValid();
   return MinimumError(2);
}

MinimumError FumiliErrorUpdator::Update(const MinimumState &s0, const MinimumParameters &p1,
                                        const GradientCalculator &gc, double lambda) const
{
   // calculate the error matrix using approximation of Fumili
   // use only first derivatives (see tutorial par. 5.1,5.2)
   // The Fumili Hessian is provided by the FumiliGradientCalculator class
   // we apply also the Marquard lambda factor to increase weight of diagonal term
   // as suggester in Numerical Receipt for Marquard method

   MnPrint print("FumiliErrorUpdator");
   print.Debug("Compute covariance matrix using Fumili method");

   // need to downcast to FumiliGradientCalculator
   FumiliGradientCalculator *fgc = dynamic_cast<FumiliGradientCalculator *>(const_cast<GradientCalculator *>(&gc));
   assert(fgc != nullptr);


   // get Hessian from Gradient calculator

   MnAlgebraicSymMatrix h = fgc->GetHessian();

   int nvar = p1.Vec().size();

   // apply Marquard lambda factor
   double eps = 8 * std::numeric_limits<double>::min();
   for (int j = 0; j < nvar; j++) {
      h(j, j) *= (1. + lambda);
      // if h(j,j) is zero what to do ?
      if (std::fabs(h(j, j)) < eps) { // should use DBL_MIN
                                      // put a cut off to avoid zero on diagonals
         if (lambda > 1)
            h(j, j) = lambda * eps;
         else
            h(j, j) = eps;
      }
   }

   MnAlgebraicSymMatrix cov(h);
   int ifail = Invert(cov);
   if (ifail != 0) {
      print.Warn("inversion fails; return diagonal matrix");

      for (unsigned int i = 0; i < cov.Nrow(); i++) {
         cov(i, i) = 1. / cov(i, i);
      }

      // shiould we return the Hessian in addition to cov in this case?
      return MinimumError(cov, MinimumError::MnInvertFailed);
   }

   double dcov = -1;
   if (s0.IsValid()) {

      const MnAlgebraicSymMatrix &v0 = s0.Error().InvHessian();

   // calculate by how much did the covariance matrix changed
   // (if it changed a lot since the last step, probably we
   // are not yet near the Minimum)
      dcov = 0.5 * (s0.Error().Dcovar() + sum_of_elements(cov - v0) / sum_of_elements(cov));
   }

   return MinimumError(cov, h, dcov);
}

} // namespace Minuit2

} // namespace ROOT
