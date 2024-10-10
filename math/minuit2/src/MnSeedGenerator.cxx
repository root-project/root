// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnSeedGenerator.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/MnMatrix.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MinuitParameter.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/VariableMetricEDMEstimator.h"
#include "Minuit2/NegativeG2LineSearch.h"
#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/HessianGradientCalculator.h"
#include "Minuit2/MnPrint.h"

#include <cmath>

namespace ROOT {

namespace Minuit2 {

MinimumSeed MnSeedGenerator::
operator()(const MnFcn &fcn, const GradientCalculator &gc, const MnUserParameterState &st, const MnStrategy &stra) const
{

   MnPrint print("MnSeedGenerator");

   // find seed (initial minimization point) using the calculated gradient
   const unsigned int n = st.VariableParameters();
   const MnMachinePrecision &prec = st.Precision();

   print.Info("Computing seed using NumericalGradient calculator");

   print.Debug(n, "free parameters, FCN pointer", &fcn);

   // initial starting values
   MnAlgebraicVector x(n);
   for (unsigned int i = 0; i < n; i++)
      x(i) = st.IntParameters()[i];
   double fcnmin = fcn(x);

   MinimumParameters pa(x, fcnmin);
   FunctionGradient dgrad = gc(pa);
   MnAlgebraicSymMatrix mat(n);
   double dcovar = 1.;
   if (st.HasCovariance()) {
      for (unsigned int i = 0; i < n; i++) {
         mat(i, i) = st.IntCovariance()(i, i) > prec.Eps() ? st.IntCovariance()(i, i)
                     : dgrad.G2()(i) > prec.Eps()          ? 1. / dgrad.G2()(i)
                                                           : 1.0;
         for (unsigned int j = i + 1; j < n; j++)
            mat(i, j) = st.IntCovariance()(i, j);
      }
      dcovar = 0.;
   } else {
      for (unsigned int i = 0; i < n; i++)
        // if G2 is small better using an arbitrary value (e.g. 1)
        mat(i, i) = dgrad.G2()(i) > prec.Eps() ? 1. / dgrad.G2()(i) : 1.0;
   }
   MinimumError err(mat, dcovar);

   double edm = VariableMetricEDMEstimator().Estimate(dgrad, err);
   MinimumState state(pa, err, dgrad, edm, fcn.NumOfCalls());

   print.Info("Initial state:", MnPrint::Oneline(state));

   if (!st.HasCovariance()) {
      NegativeG2LineSearch ng2ls;
      if (ng2ls.HasNegativeG2(dgrad, prec)) {
         print.Debug("Negative G2 Found", "\n  point:", x, "\n  grad :", dgrad.Grad(), "\n  g2   :", dgrad.G2());

         state = ng2ls(fcn, state, gc, prec);

         print.Info("Negative G2 found - new state:", state);
      }
   }

   if (stra.Strategy() == 2 && !st.HasCovariance()) {
      // calculate full 2nd derivative

      print.Debug("calling MnHesse");

      MinimumState tmp = MnHesse(stra)(fcn, state, st.Trafo());

      print.Info("run Hesse - Initial seeding state:", tmp);

      return MinimumSeed(tmp, st.Trafo());
   }

   print.Info("Initial state ",state);

   return MinimumSeed(state, st.Trafo());
}

MinimumSeed MnSeedGenerator::operator()(const MnFcn &fcn, const AnalyticalGradientCalculator &gc,
                                        const MnUserParameterState &st, const MnStrategy &stra) const
{
   MnPrint print("MnSeedGenerator");

   // check gradient (slow: will require more function evaluations)
   //if (gc.CheckGradient()) {
   //      //CheckGradient(st,trado,stra,grd)
   //}

   if (!gc.CanComputeG2()) {
      print.Info("Using analytical (external) gradient calculator but cannot compute G2 - use then numerical gradient for G2");
      Numerical2PGradientCalculator ngc(fcn, st.Trafo(), stra);
      return this->operator()(fcn, ngc, st, stra);
   }



   if (gc.CanComputeHessian())
      print.Info("Computing seed using analytical (external) gradients and Hessian calculator");
   else
      print.Info("Computing seed using analytical (external) gradients and G2 calculator");



   // find seed (initial point for minimization) using analytical gradient
   unsigned int n = st.VariableParameters();
   const MnMachinePrecision &prec = st.Precision();

   // initial starting values
   MnAlgebraicVector x(n);
   for (unsigned int i = 0; i < n; i++)
      x(i) = st.IntParameters()[i];
   double fcnmin = fcn(x);
   MinimumParameters pa(x, fcnmin);

   // compute function gradient
   FunctionGradient grad = gc(pa);
   double dcovar = 0;
   MnAlgebraicSymMatrix mat(n);
   // if we can compute Hessian compute it and use it
   bool computedHessian = false;
   if (!grad.HasG2()) {
      assert(gc.CanComputeHessian());
      MnAlgebraicSymMatrix  hmat(n);
      bool ret = gc.Hessian(pa, hmat);
      if (!ret) {
         print.Error("Cannot compute G2 and Hessian");
         assert(true);
      }
      // update gradient using G2 from Hessian calculation
      MnAlgebraicVector g2(n);
      for (unsigned int i = 0; i < n; i++)
         g2(i) = hmat(i,i);
      grad = FunctionGradient(grad.Grad(),g2);

      print.Debug("Computed analytical G2",g2);

      // when Hessian has been computed invert to get covariance
      // we prefer not using full Hessian in strategy 1 since we need to be sure that
      // is pos-defined. Uncomment following line if want to have seed with the full Hessian
      //computedHessian = true;
      if (computedHessian) {
         mat = MinimumError::InvertMatrix(hmat);
         print.Info("Use full Hessian as seed");
         print.Debug("computed Hessian",hmat);
         print.Debug("computed Error matrix (H^-1)",mat);
      }
   }
   // do this only when we have not computed the Hessian or always ?
   if (!computedHessian) {
      // check if minimum state has covariance - if not use computed G2
      // should maybe this an option, sometimes is not good to re-use existing covariance
      if (st.HasCovariance()) {
         print.Info("Using existing covariance matrix");
         for (unsigned int i = 0; i < n; i++) {
            mat(i, i) = st.IntCovariance()(i, i) > prec.Eps() ? st.IntCovariance()(i, i)
                        : grad.G2()(i) > prec.Eps()           ? 1. / grad.G2()(i)
                                                              : 1.0;
            for (unsigned int j = i + 1; j < n; j++)
               mat(i, j) = st.IntCovariance()(i, j);
         }
         dcovar = 0.;
      } else {
         for (unsigned int i = 0; i < n; i++) {
            // if G2 is very small, better using an arbitrary value (e.g. 1.)
            mat(i, i) = grad.G2()(i) > prec.Eps() ? 1. / grad.G2()(i) : 1.0;
         }
         dcovar = 1.;
      }
   } else  {
      print.Info("Computing seed using full Hessian");
   }

   MinimumError err(mat, dcovar);
   double edm = VariableMetricEDMEstimator().Estimate(grad, err);

   if (!grad.HasG2()) {
      print.Error("Cannot compute seed because G2 is not computed");
   }
   MinimumState state(pa, err, grad, edm, fcn.NumOfCalls());

   if (!st.HasCovariance()) {
      NegativeG2LineSearch ng2ls;
      if (ng2ls.HasNegativeG2(grad, prec)) {
         // do a negative line search - can use current gradient calculator
         // Numerical2PGradientCalculator ngc(fcn, st.Trafo(), stra);
         state = ng2ls(fcn, state, gc, prec);
      }
   }

   // compute Hessian above will not have posdef check as it is done if we call MnHesse
   if (stra.Strategy() == 2 && !st.HasCovariance() && !computedHessian) {
      // can calculate full 2nd derivative
      MinimumState tmpState = MnHesse(stra)(fcn, state, st.Trafo());
      print.Info("Compute full Hessian: Initial seeding state is ",tmpState);
      return MinimumSeed(tmpState, st.Trafo());
   }

   print.Info("Initial seeding state ",state);

   return MinimumSeed(state, st.Trafo());
}
#if 0
bool CheckGradient(MinimumState & st, MnUserTransformation & trafo, MnStrategy & stra)
{

   const MinimumParameters & pa = st.Parameters();
   const FunctionGradient & grd = st.FunctionGradient();

   // I think one should use Numerical2PGradientCalculator
   // since step sizes and G2 of initial gradient are wrong
   InitialGradientCalculator igc(fcn, trafo, stra);
   FunctionGradient tmp = igc(pa);
   // should also use G2 from grd (in case Analyticalgradient can compute Hessian ?)
   FunctionGradient dgrad(grd.Grad(), tmp.G2(), tmp.Gstep());

   // do check computing gradient with HessianGradientCalculator which refines the gradient given an initial one
      bool good = true;
      HessianGradientCalculator hgc(fcn, trafo, MnStrategy(2));
      std::pair<FunctionGradient, MnAlgebraicVector> hgrd = hgc.DeltaGradient(pa, dgrad);
      for (unsigned int i = 0; i < n; i++) {
         if (std::fabs(hgrd.first.Grad()(i) - grd.Grad()(i)) > hgrd.second(i)) {
            int externalParameterIndex = trafo.ExtOfInt(i);
            const char *parameter_name = trafo.Name(externalParameterIndex);
            print.Warn("Gradient discrepancy of external Parameter too large:"
                       "parameter_name =",
                       parameter_name, "externalParameterIndex =", externalParameterIndex, "internal =", i);
            good = false;
         }
      }
      if (!good) {
         print.Error("Minuit does not accept user specified Gradient. To force acceptance, override 'virtual bool "
                     "CheckGradient() const' of FCNGradientBase.h in the derived class.");

         assert(good);
      }
      return good
}
#endif

} // namespace Minuit2

} // namespace ROOT
