// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/MinimumSeedGenerator.h"
#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/ExternalInternalGradientCalculator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/MinimumBuilder.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/FCNGradientBase.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/MnPrint.h"

#include <iostream>
#include <chrono>

class TimingScope {
public:
   TimingScope(std::string const &label) : _begin{std::chrono::steady_clock::now()}, _label{label} {}
   ~TimingScope()
   {
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time difference for " << _label << " = "
                //<< std::chrono::duration_cast<std::chrono::microseconds>(end - _begin).count() << " [µs]" << std::endl;
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - _begin).count() << " [ms]" << std::endl;
   }

private:
   std::chrono::steady_clock::time_point _begin;
   const std::string _label;
};


namespace ROOT {

namespace Minuit2 {

// #include "Minuit2/MnUserParametersPrint.h"

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase &fcn, const std::vector<double> &par,
                                                   const std::vector<double> &err, unsigned int stra,
                                                   unsigned int maxfcn, double toler) const
{
   // minimize from FCNBase and std::vector of double's for parameter values and errors (step sizes)
   MnUserParameterState st(par, err);
   MnStrategy strategy(stra);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase &fcn, const std::vector<double> &par,
                                                   const std::vector<double> &err, unsigned int stra,
                                                   unsigned int maxfcn, double toler) const
{
   // minimize from FCNGradientBase (use analytical gradient provided in FCN)
   // and std::vector of double's for parameter values and errors (step sizes)
   MnUserParameterState st(par, err);
   MnStrategy strategy(stra);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

// move nrow before cov to avoid ambiguities when using default parameters
FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase &fcn, const std::vector<double> &par,
                                                   unsigned int nrow, const std::vector<double> &cov, unsigned int stra,
                                                   unsigned int maxfcn, double toler) const
{
   // minimize from FCNBase using std::vector for parameter error and
   // an std::vector of size n*(n+1)/2 for the covariance matrix  and n (rank of cov matrix)

   MnUserParameterState st(par, cov, nrow);
   MnStrategy strategy(stra);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase &fcn, const std::vector<double> &par,
                                                   unsigned int nrow, const std::vector<double> &cov, unsigned int stra,
                                                   unsigned int maxfcn, double toler) const
{
   // minimize from FCNGradientBase (use analytical gradient provided in FCN)
   // using std::vector for parameter error and
   // an std::vector of size n*(n+1)/2 for the covariance matrix  and n (rank of cov matrix)

   MnUserParameterState st(par, cov, nrow);
   MnStrategy strategy(stra);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase &fcn, const MnUserParameters &upar,
                                                   const MnStrategy &strategy, unsigned int maxfcn, double toler) const
{
   // minimize from FCNBase and MnUserParameters object

   MnUserParameterState st(upar);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase &fcn, const MnUserParameters &upar,
                                                   const MnStrategy &strategy, unsigned int maxfcn, double toler) const
{
   // minimize from FCNGradientBase (use analytical gradient provided in FCN)  and MnUserParameters object

   MnUserParameterState st(upar);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase &fcn, const MnUserParameters &upar,
                                                   const MnUserCovariance &cov, const MnStrategy &strategy,
                                                   unsigned int maxfcn, double toler) const
{
   // minimize from FCNBase and MnUserParameters and MnUserCovariance objects

   MnUserParameterState st(upar, cov);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase &fcn, const MnUserParameters &upar,
                                                   const MnUserCovariance &cov, const MnStrategy &strategy,
                                                   unsigned int maxfcn, double toler) const
{
   // minimize from FCNGradientBase (use analytical gradient provided in FCN)  and
   // MnUserParameters MnUserCovariance objects

   MnUserParameterState st(upar, cov);
   return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase &fcn, const MnUserParameterState &st,
                                                   const MnStrategy &strategy, unsigned int maxfcn, double toler) const
{
   // minimize from a FCNBase and a MnUserparameterState - interface used by all the previous ones
   // based on FCNBase. Create in this case a NumericalGradient calculator
   // Create the minuit FCN wrapper (MnUserFcn) containing the transformation (int<->ext)

   // need MnUserFcn for difference int-ext parameters
   MnUserFcn mfcn(fcn, st.Trafo());
   Numerical2PGradientCalculator gc(mfcn, st.Trafo(), strategy);

   unsigned int npar = st.VariableParameters();
   if (maxfcn == 0)
      maxfcn = 200 + 100 * npar + 5 * npar * npar;

   auto ts = std::make_unique<TimingScope>("Seeding");
   MinimumSeed mnseeds = SeedGenerator()(mfcn, gc, st, strategy);
   ts = std::make_unique<TimingScope>("Minimization");
   auto out = Minimize(mfcn, gc, mnseeds, strategy, maxfcn, toler);
   ts.reset();
   return out;
}

// use Gradient here
FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase &fcn, const MnUserParameterState &st,
                                                   const MnStrategy &strategy, unsigned int maxfcn, double toler) const
{
   // minimize from a FCNGradientBase and a MnUserParameterState -
   // interface based on FCNGradientBase (external/analytical gradients)
   // Create in this case an AnalyticalGradient calculator
   // Create the minuit FCN wrapper (MnUserFcn) containing the transformation (int<->ext)

   MnUserFcn mfcn(fcn, st.Trafo());
   std::unique_ptr<AnalyticalGradientCalculator> gc;
   if (fcn.gradParameterSpace() == GradientParameterSpace::Internal) {
        //        std::cout << "-- ModularFunctionMinimizer::Minimize: Internal parameter space" << std::endl;
        gc = std::unique_ptr<AnalyticalGradientCalculator>(new ExternalInternalGradientCalculator(fcn, st.Trafo()));
   } else {
        //        std::cout << "-- ModularFunctionMinimizer::Minimize: External parameter space" << std::endl;
        gc = std::make_unique<AnalyticalGradientCalculator>(fcn, st.Trafo());
   }

   unsigned int npar = st.VariableParameters();
   if (maxfcn == 0)
      maxfcn = 200 + 100 * npar + 5 * npar * npar;

   // compute seed (will use internally numerical gradient in case calculator does not implement g2 computations)
   auto ts = std::make_unique<TimingScope>("Seeding");
   MinimumSeed mnseeds = SeedGenerator()(mfcn, *gc, st, strategy);
   ts = std::make_unique<TimingScope>("Minimization");
   auto minimum = Minimize(mfcn, *gc, mnseeds, strategy, maxfcn, toler);
   ts.reset();

   return minimum;
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const MnFcn &mfcn, const GradientCalculator &gc,
                                                   const MinimumSeed &seed, const MnStrategy &strategy,
                                                   unsigned int maxfcn, double toler) const
{
   // Interface used by all the others for the minimization using the base MinimumBuilder class
   // According to the contained type of MinimumBuilder the right type will be used

   MnPrint print("ModularFunctionMinimizer");

   const MinimumBuilder &mb = Builder();
   // std::cout << typeid(&mb).Name() << std::endl;
   double effective_toler = toler * mfcn.Up(); // scale tolerance with Up()
   // avoid tolerance too smalls (than limits)
   double eps = MnMachinePrecision().Eps2();
   if (effective_toler < eps)
      effective_toler = eps;

   // check if maxfcn is already exhausted
   // case already reached call limit
   if (mfcn.NumOfCalls() >= maxfcn) {
      print.Warn("Stop before iterating - call limit already exceeded");

      return FunctionMinimum(seed, std::vector<MinimumState>(1, seed.State()), mfcn.Up(),
                             FunctionMinimum::MnReachedCallLimit);
   }

   return mb.Minimum(mfcn, gc, seed, strategy, maxfcn, effective_toler);
}

} // namespace Minuit2

} // namespace ROOT
