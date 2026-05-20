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
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnFcn.h"

namespace ROOT {

namespace Minuit2 {

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase &fcn, const MnUserParameterState &st,
                                                   const MnStrategy &strategy, unsigned int maxfcn, double toler) const
{
   // need wrapper for difference int-ext parameters
   MnFcn mfcn{fcn, st.Trafo()};
   std::unique_ptr<GradientCalculator> gc;

   if (!fcn.HasGradient()) {
      // minimize from a FCNBase and a MnUserparameterState - interface used by all the previous ones
      // based on FCNBase. Create in this case a NumericalGradient calculator
      gc = std::make_unique<Numerical2PGradientCalculator>(mfcn, st.Trafo(), strategy);
   } else if (fcn.gradParameterSpace() == GradientParameterSpace::Internal) {
      // minimize from a function with gradient and a MnUserParameterState -
      // interface based on function with gradient (external/analytical gradients)
      // Create in this case an AnalyticalGradient calculator
      gc = std::make_unique<ExternalInternalGradientCalculator>(fcn, st.Trafo());
   } else {
      gc = std::make_unique<AnalyticalGradientCalculator>(fcn, st.Trafo());
   }

   unsigned int npar = st.VariableParameters();
   if (maxfcn == 0)
      maxfcn = 200 + 100 * npar + 5 * npar * npar;

   // compute seed (will use internally numerical gradient in case calculator does not implement g2 computations)
   MinimumSeed mnseeds = SeedGenerator()(mfcn, *gc, st, strategy);
   return Minimize(mfcn, *gc, mnseeds, strategy, maxfcn, toler);
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
