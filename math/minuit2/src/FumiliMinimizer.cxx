// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnConfig.h"
#include "Minuit2/FumiliMinimizer.h"
#include "Minuit2/MinimumSeedGenerator.h"
#include "Minuit2/FumiliGradientCalculator.h"
#include "Minuit2/FumiliErrorUpdator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/MinimumBuilder.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/FumiliFCNBase.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {

namespace Minuit2 {

// for Fumili implement Minimize here because need downcast

FunctionMinimum FumiliMinimizer::Minimize(const FCNBase &fcn, const MnUserParameterState &st,
                                          const MnStrategy &strategy, unsigned int maxfcn, double toler) const
{
   MnPrint print("FumiliMinimizer::Minimize");

   // Minimize using Fumili. Create seed and Fumili gradient calculator.
   // The FCNBase passed must be a FumiliFCNBase type otherwise method will fail !

   MnUserFcn mfcn(fcn, st.Trafo());
   //  Numerical2PGradientCalculator gc(mfcn, st.Trafo(), strategy);

   unsigned int npar = st.VariableParameters();
   if (maxfcn == 0)
      maxfcn = 200 + 100 * npar + 5 * npar * npar;
   // FUMILI needs much less function calls
   maxfcn = int(0.1 * maxfcn);

   // MinimumSeed mnseeds = SeedGenerator()(mfcn, gc, st, strategy);

   // std::cout << "FCN type " << typeid(&fcn).Name() << std::endl;

   // Minimize using Fumili. Case of interface is a function with gradient.
   FumiliFCNBase *fumiliFcn = dynamic_cast<FumiliFCNBase *>(const_cast<FCNBase *>(&fcn));
   if (!fumiliFcn) {
      print.Error("Wrong FCN type; try to use default minimizer");
      return FunctionMinimum(MinimumSeed(MinimumState(0), st.Trafo()), fcn.Up());
   }

   FumiliGradientCalculator fgc(*fumiliFcn, st.Trafo(), npar);
   if (fcn.HasGradient()) {
      print.Debug("Using FumiliMinimizer with analytical gradients");
   } else {
      print.Debug("Using FumiliMinimizer with numerical gradients");
   }

   // compute initial values;
   const unsigned int n = st.VariableParameters();
   MnAlgebraicVector x(n);
   for (unsigned int i = 0; i < n; i++)
      x(i) = st.IntParameters()[i];
   double fcnmin = mfcn(x);
   MinimumParameters pa(x, fcnmin);
   FunctionGradient grad = fgc(pa);
   FumiliErrorUpdator errUpdator;
   MinimumError err = errUpdator.Update(MinimumState(0), pa, fgc, 0.);
   MinimumSeed mnseeds(MinimumState(pa, err, grad, -1., 1), st.Trafo());

   return ModularFunctionMinimizer::Minimize(mfcn, fgc, mnseeds, strategy, maxfcn, toler);
}


void FumiliMinimizer::SetMethod(const std::string & method) {
   if (method == "tr")
      fMinBuilder.SetMethod(FumiliBuilder::kTrustRegion);
   else if (method == "ls")
      fMinBuilder.SetMethod(FumiliBuilder::kLineSearch);
   else if (method == "trs")
      fMinBuilder.SetMethod(FumiliBuilder::kTrustRegionScaled);
}


} // namespace Minuit2

} // namespace ROOT
