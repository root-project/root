// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/NegativeG2LineSearch.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/VariableMetricEDMEstimator.h"
#include "Minuit2/MnPrint.h"

#include "Math/Util.h"

#include <cmath>

namespace ROOT {

namespace Minuit2 {

MinimumState NegativeG2LineSearch::operator()(const MnFcn &fcn, const MinimumState &st, const GradientCalculator &gc,
                                              const MnMachinePrecision &prec) const
{

   //   when the second derivatives are negative perform a  line search  along Parameter which gives
   //   negative second derivative and magnitude  equal to the Gradient step size.
   //   Recalculate the gradients for all the Parameter after the correction and
   //   continue iteration in case the second derivatives are still negative
   //
   MnPrint print("NegativeG2LineSearch");

   // Print the runtime on returning from the function
   ROOT::Math::Util::TimingScope timingScope([&print](std::string const& s){ print.Info(s); }, "Done after");

   bool negG2 = HasNegativeG2(st.Gradient(), prec);
   if (!negG2)
      return st;

   print.Info("Doing a NegativeG2LineSearch since one of the G2 component is negative");

   unsigned int n = st.Parameters().Vec().size();
   FunctionGradient dgrad = st.Gradient();
   MinimumParameters pa = st.Parameters();
   bool iterate = false;
   unsigned int iter = 0;
   // in case of analytical gradients we don't have step sizes
   bool hasGStep = !dgrad.IsAnalytical();
   //print.Trace("Gradient ", dgrad.Vec(), "G2",dgrad.G2());
   // gradient present in the state must have G2 otherwise something is wrong
   if (!dgrad.HasG2()) {
      print.Error("Input gradient to NG2LS must have G2 already computed");
      return st;
   }

   do {
      iterate = false;
      for (unsigned int i = 0; i < n; i++) {

         if (dgrad.G2()(i) <= 0) {

            // check also the gradient (if it is zero ) I can skip the param)

            if (std::fabs(dgrad.Vec()(i)) < prec.Eps() && std::fabs(dgrad.G2()(i)) < prec.Eps())
               continue;
            //       if(dgrad.G2()(i) < prec.Eps()) {
            // do line search if second derivative negative
            MnAlgebraicVector step(n);
            MnLineSearch lsearch;

            // when using analytical gradient use as step size a dummy value of 1
            // maybe could do better using user given parameter step sizes
            // tested using inverse of G2() gives worse behaviour
            if (dgrad.Vec()(i) < 0)
               step(i) = (hasGStep) ? dgrad.Gstep()(i) : 1;
            else
               step(i) = (hasGStep) ? -dgrad.Gstep()(i) : -1;

            double gdel = step(i) * dgrad.Vec()(i);

            // if using sec derivative information
            // double g2del = step(i)*step(i) * dgrad.G2()(i);

            print.Debug("Iter", iter, "param", i, pa.Vec()(i), "grad2", dgrad.G2()(i), "grad",
                        dgrad.Vec()(i), "grad step", step(i), " gdel ", gdel);

            auto pp = lsearch(fcn, pa, step, gdel, prec);

            print.Debug("Line search result", pp.X(), "f(0)", pa.Fval(), "f(1)", pp.Y());

            step *= pp.X();
            pa = MinimumParameters(pa.Vec() + step, pp.Y());

            dgrad = gc(pa, dgrad);
            // re-compute also G2 if needed
            if (!dgrad.HasG2()) {
               //no need to compute Hessian here but only G2
               print.Debug("Compute  G2 at the new point", pa.Vec());
               MnAlgebraicVector g2(n);
               bool ret = gc.G2(pa,g2);
               if (!ret) {
                  print.Error("Cannot compute G2");
                  assert(false);
                  return st;
               }

               dgrad = FunctionGradient(dgrad.Grad(), g2);
            }

            print.Debug("New result after Line search - iter", iter, "param", i, pa.Vec()(i), "step", step(i), "new grad2",
                        dgrad.G2()(i), "new grad", dgrad.Vec()(i));

            iterate = true;
            break;
         }
      }
   } while (iter++ < 2 * n && iterate);

   // even if we computed the Hessian it is still better to use the diagonal part, the G2
   print.Debug("Approximate new covariance after NegativeG2LS using only G2");
   MnAlgebraicSymMatrix mat(n);
   for (unsigned int i = 0; i < n; i++) {
      mat(i, i) = std::fabs(dgrad.G2()(i)) > prec.Eps() ? 1. / dgrad.G2()(i) :
                  1; // use an arbitrary value (e.g. 1)
   }

   MinimumError err(mat, 1.);
   double edm = VariableMetricEDMEstimator().Estimate(dgrad, err);

   if (edm < 0) {
      err = MinimumError(mat, MinimumError::MnNotPosDef);
   }

   return MinimumState(pa, err, dgrad, edm, fcn.NumOfCalls());
}

bool NegativeG2LineSearch::HasNegativeG2(const FunctionGradient &grad, const MnMachinePrecision & /*prec */) const
{
   // check if function gradient has any component which is negative

   for (unsigned int i = 0; i < grad.Vec().size(); i++)

      if (grad.G2()(i) <= 0) {

         return true;
      }

   return false;
}

} // namespace Minuit2

} // namespace ROOT
