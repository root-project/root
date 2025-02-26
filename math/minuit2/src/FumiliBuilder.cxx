// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/FumiliBuilder.h"
#include "Minuit2/FumiliStandardMaximumLikelihoodFCN.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnPosDef.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {

namespace Minuit2 {

FunctionMinimum FumiliBuilder::Minimum(const MnFcn &fcn, const GradientCalculator &gc, const MinimumSeed &seed,
                                       const MnStrategy &strategy, unsigned int maxfcn, double edmval) const
{
   // top level function to find minimum from a given initial seed
   // iterate on a minimum search in case of first attempt is not successful

   MnPrint print("FumiliBuilder", PrintLevel());

   edmval *= 0.0001;
   // edmval *= 0.1; // use small factor for Fumili

   print.Debug("Convergence when edm <", edmval);

   if (seed.Parameters().Vec().size() == 0) {
      print.Warn("No variable parameters are defined! - Return current function value ");
      return FunctionMinimum(seed, fcn.Up());
   }

   // estimate initial edm value
   double edm = Estimator().Estimate(seed.Gradient(), seed.Error());
   print.Debug("initial edm is ", edm);
   //double edm = seed.State().Edm();

   FunctionMinimum min(seed, fcn.Up());

   if (edm < 0.) {
      print.Error("Initial matrix not positive defined, edm = ",edm,"\nExit minimization ");
      return min;
   }

   std::vector<MinimumState> result;
   //   result.reserve(1);
   result.reserve(8);

   result.push_back(seed.State());

   print.Info("Start iterating until Edm is <", edmval, '\n', "Initial state", MnPrint::Oneline(seed.State()));

   if (TraceIter())
      TraceIteration(result.size() - 1, result.back());

   // do actual iterations

   // try first with a maxfxn = 50% of maxfcn
   // Fumili in principle needs much less iterations
   int maxfcn_eff = int(0.5 * maxfcn);
   int ipass = 0;
   double edmprev = 1;

   do {

      min = Minimum(fcn, gc, seed, result, maxfcn_eff, edmval);

      // second time check for validity of function Minimum
      if (ipass > 0) {
         if (!min.IsValid()) {

            print.Warn("FunctionMinimum is invalid");

            return min;
         }
      }

      // resulting edm of minimization
      edm = result.back().Edm();

      print.Debug("Approximate Edm", edm, "npass", ipass);

      // call always Hesse (error matrix from Fumili is never accurate since is approximate)

      if (strategy.Strategy() > 0) {
      print.Debug("FumiliBuilder will verify convergence and Error matrix; "
                  "dcov",
                  min.Error().Dcovar());

      //       // recalculate the gradient using the numerical gradient calculator
      //       Numerical2PGradientCalculator ngc(fcn, min.Seed().Trafo(), strategy);
      //       FunctionGradient ng = ngc( min.State().Parameters() );
      //       MinimumState tmp( min.State().Parameters(), min.State().Error(), ng, min.State().Edm(),
      //       min.State().NFcn() );

      MinimumState st = MnHesse(strategy)(fcn, min.State(), min.Seed().Trafo(), maxfcn);
      result.push_back(st);

      print.Info("After Hessian");
      if (TraceIter())
         TraceIteration(result.size() - 1, result.back());

      // check edm
      edm = st.Edm();

      print.Debug("Edm", edm, "State", st);
      }

      // break the loop if edm is NOT getting smaller
      if (ipass > 0 && edm >= edmprev) {
         print.Warn("Stop iterations, no improvements after Hesse; current Edm", edm, "previous value", edmprev);
         break;
      }
      if (edm > edmval) {
         print.Debug("Tolerance not sufficient, continue minimization; Edm", edm, "Requested", edmval);
      } else {
         // Case when edm < edmval after Heasse but min is flagged with a bad edm:
         // make then a new Function minimum since now edm is ok
         if (min.IsAboveMaxEdm()) {
            min = FunctionMinimum(min.Seed(), min.States(), min.Up());
            break;
         }
      }

      // end loop on iterations
      // ? need a maximum here (or max of function calls is enough ? )
      // continnue iteration (re-calculate function Minimum if edm IS NOT sufficient)
      // no need to check that hesse calculation is done (if isnot done edm is OK anyway)
      // count the pass to exit second time when function Minimum is invalid
      // increase by 20% maxfcn for doing some more tests
      if (ipass == 0)
         maxfcn_eff = maxfcn;
      ipass++;
      edmprev = edm;

   } while (edm > edmval);

   // Add latest state (Hessian calculation)
   min.Add(result.back());

   return min;
}

FunctionMinimum FumiliBuilder::Minimum(const MnFcn &fcn, const GradientCalculator &gc, const MinimumSeed &seed,
                                       std::vector<MinimumState> &result, unsigned int maxfcn, double edmval) const
{

   // function performing the minimum searches using the FUMILI algorithm
   // after the modification when I iterate on this functions, so it can be called many times,
   //  the seed is used here only to get precision and construct the returned FunctionMinimum object

   /*
    Three options were possible:

    1) create two parallel and completely separate hierarchies, in which case
    the FumiliMinimizer would NOT inherit from ModularFunctionMinimizer,
    FumiliBuilder would not inherit from MinimumBuilder etc

    2) Use the inheritance (base classes of ModularFunctionMinimizer,
                            MinimumBuilder etc), but recreate the member functions Minimize() and
    Minimum() respectively (naming them for example minimize2() and
                            minimum2()) so that they can take FumiliFCNBase as Parameter instead FCNBase
    (otherwise one wouldn't be able to call the Fumili-specific methods).

    3) Cast in the daughter classes derived from ModularFunctionMinimizer,
    MinimumBuilder.

    The first two would mean to duplicate all the functionality already existent,
    which is a very bad practice and Error-prone. The third one is the most
    elegant and effective solution, where the only constraint is that the user
    must know that they have to pass a subclass of FumiliFCNBase to the FumiliMinimizer
    and not just a subclass of FCNBase.
    BTW, the first two solutions would have meant to recreate also a parallel
    structure for MnFcn...
    **/
   //  const FumiliFCNBase* tmpfcn =  dynamic_cast<const FumiliFCNBase*>(&(fcn.Fcn()));

   const MnMachinePrecision &prec = seed.Precision();

   const MinimumState &initialState = result.back();

   double edm = initialState.Edm();

   MnPrint print("FumiliBuilder");

   print.Info("Iterating FumiliBuilder", maxfcn, edmval);

   print.Debug("Initial State:", "\n  Parameter", initialState.Vec(), "\n  Gradient", initialState.Gradient().Vec(),
               "\n  Inv Hessian", initialState.Error().InvHessian(), "\n  edm", initialState.Edm(), "\n  maxfcn",
               maxfcn, "\n  tolerance", edmval);

   // iterate until edm is small enough or max # of iterations reached
   edm *= (1. + 3. * initialState.Error().Dcovar());
   MnAlgebraicVector step(initialState.Gradient().Vec().size());

   // initial lambda Value


   const bool doLineSearch = (fMethodType == kLineSearch);
   const bool doTrustRegion = (fMethodType == kTrustRegion || fMethodType == kTrustRegionScaled);
   const bool scaleTR = (fMethodType == kTrustRegionScaled);
   // trust region parameter
   // use as initial value 0.3 * || x0 ||
   auto & x0 = initialState.Vec();
   double normX0 = std::sqrt(inner_product(x0,x0));
   double delta = 0.3 * std::max(1.0 , normX0);
   const double eta = 0.1;
   // use same values as used in GSL implementation
   const double tr_factor_up = 3;
   const double tr_factor_down = 0.5;
   bool acceptNewPoint = true;

   if (doLineSearch)
      print.Info("Using Fumili with a line search algorithm");
   else if (doTrustRegion && scaleTR)
      print.Info("Using Fumili with a scaled trust region algorithm with factors up/down",tr_factor_up,tr_factor_down);
   else
      print.Info("Using Fumili with a trust region algorithm with factors up/down",tr_factor_up,tr_factor_down);


   double lambda = (doLineSearch) ? 0.001 : 0;

   MnFcnCaller fcnCaller{fcn};

   do {

      //     const MinimumState& s0 = result.back();
      MinimumState s0 = result.back();

      step = -1. * s0.Error().InvHessian() * s0.Gradient().Vec();

      print.Debug("Iteration -", result.size(), "\n  Fval", s0.Fval(), "numOfCall", fcn.NumOfCalls(),
                  "\n  Internal Parameter values", s0.Vec(), "\n  Newton step", step);

      double gdel = inner_product(step, s0.Gradient().Grad());
      //not sure if this is needed ??
      if (gdel > 0.) {
         print.Warn("Matrix not pos.def, gdel =", gdel, " > 0");

         MnPosDef psdf;
         s0 = psdf(s0, prec);
         step = -1. * s0.Error().InvHessian() * s0.Gradient().Vec();
         gdel = inner_product(step, s0.Gradient().Grad());

         print.Warn("After correction, gdel =", gdel);

         if (gdel > 0.) {
            result.push_back(s0);
            return FunctionMinimum(seed, result, fcn.Up());
         }
      }

      // take a full step

      //evaluate function only if doing a line search
      double fval2 = (doLineSearch) ? fcnCaller(s0.Vec() + step) : 0;
      MinimumParameters p(s0.Vec() + step, fval2);

      // check that taking the full step does not deteriorate minimum
      // in that case do a line search
      if (doLineSearch && p.Fval() >= s0.Fval()) {
         print.Debug("Do a line search", fcn.NumOfCalls());
         MnLineSearch lsearch;
         auto pp = lsearch(fcn, s0.Parameters(), step, gdel, prec);

         if (std::fabs(pp.Y() - s0.Fval()) < prec.Eps()) {
            // std::cout<<"FumiliBuilder: no improvement"<<std::endl;
            break; // no improvement
         }
         p = MinimumParameters(s0.Vec() + pp.X() * step, pp.Y());
         print.Debug("New point after Line Search :", "\n  FVAL     ", p.Fval(), "\n  Parameter", p.Vec());
      }
      // use as scaling matrix diagonal of Hessian
      auto & H = s0.Error().Hessian();
      unsigned int n = (scaleTR) ?   H.Nrow() : 0;

      MnAlgebraicSymMatrix D(n);
      MnAlgebraicSymMatrix Dinv(n);
      MnAlgebraicSymMatrix Dinv2(n);
      MnAlgebraicVector stepScaled(n);
      // set the scaling matrix to the sqrt(diagoinal hessian)
      for (unsigned int i = 0; i < n; i++){
         double d  = std::sqrt(H(i,i));
         D(i,i) = d;
         Dinv(i,i) = 1./d;
         Dinv2(i,i) = 1./(d*d);
      }

      if (doTrustRegion) {
         // do simple trust region using some control delta and eta
         // compute norm of Newton step
         double norm = 0;
         if (scaleTR) {
            print.Debug("scaling Trust region with diagonal matrix D ",D);
            stepScaled = D * step;
            norm = sqrt(inner_product(stepScaled, stepScaled) );
         }
         else
            norm = sqrt(inner_product(step,step));
         // some conditions

         // double tr_radius = norm;
         // if (norm < 0.1 * delta) tr_radius = 0.1*delta;
         // else if (norm > delta) tr_radius = delta;

         // update new point using reduced step
         //step = (tr_radius/norm) * step;

         // if the proposed point (newton step) is inside the trust region radius accept it
         if (norm <= delta) {
            p = MinimumParameters(s0.Vec() + step, fcnCaller(s0.Vec() + step));
            print.Debug("Accept full Newton step - it is inside TR ",delta);
         } else {
            //step = - (delta/norm) * step;

            // if Newton step  is outside try to use the Cauchy point ?
            double normGrad2 = 0;
            double gHg = 0;
            if (scaleTR) {
               auto gScaled = Dinv * s0.Gradient().Grad();
               normGrad2 = inner_product(gScaled, gScaled);
               // compute D-1 H D-1 = H(i,j) * D(i,i) * D(j,j)
               MnAlgebraicSymMatrix Hscaled(n);
               for (unsigned int i = 0; i < n; i++) {
                  for (unsigned int j = 0; j <=i; j++) {
                     Hscaled(i,j) = H(i,j) * Dinv(i,i) * Dinv(j,j);
                  }
               }
               gHg = similarity(gScaled, Hscaled);
            } else {
               normGrad2 = inner_product(s0.Gradient().Grad(),s0.Gradient().Grad());
               gHg = similarity(s0.Gradient().Grad(), s0.Error().Hessian());
            }
            double normGrad = sqrt(normGrad2);
            // need to compute gTHg :


            print.Debug("computed gdel gHg and normGN",gdel,gHg,norm*norm, normGrad2);
            if (gHg <= 0.) {
               // Cauchy point is at the trust region boundary)
               step = - (delta/ normGrad) * s0.Gradient().Grad();
               if (scaleTR) step = Dinv2 * step;
               print.Debug("Use as new point the Cauchy  point - along gradient with norm=delta ", delta);
            } else {
               // Cauchy point can be inside the trust region
               double tau = std::min( (normGrad2*normGrad)/(gHg*delta), 1.0);

               MnAlgebraicVector stepC(step.size());
               stepC = -tau * (delta / normGrad) * s0.Gradient().Grad();
               if (scaleTR)
                  stepC = Dinv2 * stepC;

               print.Debug("Use as new point the Cauchy  point - along gradient with tau ", tau, "delta = ", delta);

               // compute dog -leg step solving quadratic equation a * t^2 + b * t + c = 0
               // where a = ||p_n - P_c ||^2
               //       b  = 2 * p_c * (P_n - P_c)
               //       c = || pc ||^2 - ||Delta||^2
               auto diffP = step - stepC;
               double a = inner_product(diffP, diffP);
               double b = 2. * inner_product(stepC, diffP);
               // norm cauchy point is tau*delta
               double c = (scaleTR) ? inner_product(stepC, stepC) - delta * delta : delta * delta * (tau * tau - 1.);

               print.Debug(" dogleg equation", a, b, c);
               // solution is
               double t = 0;
               // a cannot be negative or zero, otherwise point was accepted
               if (a <= 0) {
                  // case a is zero
                  print.Warn("a is equal to zero!  a = ", a);
                  print.Info(" delta ", delta, " tau ", tau, " gHg ", gHg, " normgrad2 ", normGrad2);
                  t = -b / c;
               } else {
                  double t1 = (-b + sqrt(b * b - 4. * a * c)) / (2.0 * a);
                  double t2 = (-b - sqrt(b * b - 4. * a * c)) / (2.0 * a);
                  // if b is positive solution is
                  print.Debug(" solution dogleg equation", t1, t2);
                  if (t1 >= 0 && t1 <= 1.)
                     t = t1;
                  else
                     t = t2;
               }
               step = stepC + t * diffP;
               // need to rescale point per D^-1 >
               print.Debug("New dogleg point is t = ", t);
            }
            print.Debug("New accepted step is ",step);

            p = MinimumParameters(s0.Vec() + step, fcnCaller(s0.Vec() + step));
            norm = delta;
            gdel = inner_product(step, s0.Gradient().Grad());
         }

         // compute real changes (require an evaluation)

         // expected change is gdel (can correct for Hessian )
         //double fcexp =  (-gdel - 0.5 * dot(step, hess(x) * step)

         double svs = 0.5 * similarity(step, s0.Error().Hessian());
         double rho = (p.Fval()-s0.Fval())/(gdel+svs);
         if (rho < 0.25) {
            delta = tr_factor_down * delta;
         } else {
            if (rho > 0.75 && norm == delta) {
                delta = std::min(tr_factor_up * delta, 100.*norm);  // 1000. is the delta max
            }
         }
         print.Debug("New point after Trust region :", "norm tr ",norm," rho ", rho," delta ", delta,
           "  FVAL    ", p.Fval(), "\n  Parameter", p.Vec());

      // check if we need to accept the point ?
         acceptNewPoint = (rho > eta);
         if (acceptNewPoint) {
               // we accept the point
               // we have already p = s0 + step
               print.Debug("Trust region: accept new point p = x + step since rho is larger than eta");
           }
           else {
             // we keep old point
             print.Debug("Trust region reject new point and repeat since rho is smaller than eta");
             p = MinimumParameters(s0.Vec(), s0.Fval());
           }

      }

      FunctionGradient g = s0.Gradient();

      if (acceptNewPoint || result.size() == 1) {

      print.Debug("Before Gradient - NCalls = ", fcn.NumOfCalls());

      g = gc(p, s0.Gradient());

      print.Debug("After Gradient - NCalls = ", fcn.NumOfCalls());

      }

      // move Error updator after Gradient since the Value is cached inside

      MinimumError e = ErrorUpdator().Update(s0, p, gc, lambda);

      // should I use here new error instead of old one (so.Error) ?
      edm = Estimator().Estimate(g, s0.Error());

      print.Debug("Updated new point:", "\n  FVAL     ", p.Fval(), "\n  Parameter", p.Vec(), "\n  Gradient", g.Vec(),
                  "\n  InvHessian", e.InvHessian(), "\n  Hessian", e.Hessian(), "\n  Edm", edm);

      if (edm < 0.) {
         print.Warn("Matrix not pos.def., Edm < 0");

         MnPosDef psdf;
         s0 = psdf(s0, prec);
         edm = Estimator().Estimate(g, s0.Error());
         if (edm < 0.) {
            result.push_back(s0);
            if (TraceIter())
               TraceIteration(result.size() - 1, result.back());
            return FunctionMinimum(seed, result, fcn.Up());
         }
      }

      // check lambda according to step
      if (doLineSearch) {
         if (p.Fval() < s0.Fval())
            // fcn is decreasing along the step
            lambda *= 0.1;
         else {
            lambda *= 10;
            // if close to minimum stop to avoid oscillations around minimum value
            if (edm < 0.1)
               break;
         }
      }

      print.Debug("finish iteration -", result.size(), "lambda =", lambda, "f1 =", p.Fval(), "f0 =", s0.Fval(),
                  "num of calls =", fcn.NumOfCalls(), "edm =", edm);

      result.push_back(MinimumState(p, e, g, edm, fcn.NumOfCalls()));
      if (TraceIter())
         TraceIteration(result.size() - 1, result.back());

      print.Info(MnPrint::Oneline(result.back(), result.size()));

      edm *= (1. + 3. * e.Dcovar());

      //}  // endif on acceptNewPoint


   } while (edm > edmval && fcn.NumOfCalls() < maxfcn);

   if (fcn.NumOfCalls() >= maxfcn) {
      print.Warn("Call limit exceeded",fcn.NumOfCalls(), maxfcn);

      return FunctionMinimum(seed, result, fcn.Up(), FunctionMinimum::MnReachedCallLimit);
   }

   if (edm > edmval) {
      if (edm < std::fabs(prec.Eps2() * result.back().Fval())) {
         print.Warn("Machine accuracy limits further improvement");

         return FunctionMinimum(seed, result, fcn.Up());
      } else if (edm < 10 * edmval) {
         return FunctionMinimum(seed, result, fcn.Up());
      } else {

         print.Warn("No convergence; Edm", edm, "is above tolerance", 10 * edmval);

         return FunctionMinimum(seed, result, fcn.Up(), FunctionMinimum::MnAboveMaxEdm);
      }
   }

   print.Debug("Exiting successfully", "Ncalls", fcn.NumOfCalls(), "FCN", result.back().Fval(), "Edm", edm, "Requested",
               edmval);

   return FunctionMinimum(seed, result, fcn.Up());
}

} // namespace Minuit2

} // namespace ROOT
