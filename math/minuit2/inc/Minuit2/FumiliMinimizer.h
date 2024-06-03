// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FumiliMinimizer
#define ROOT_Minuit2_FumiliMinimizer

#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/MnSeedGenerator.h"
#include "Minuit2/FumiliBuilder.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class MinimumSeedGenerator;
class MinimumBuilder;
class MinimumSeed;
class MnFcn;
class FumiliFcnBase;
class GradientCalculator;
class MnUserParameterState;
class MnUserParameters;
class MnUserCovariance;
class MnStrategy;

//______________________________________________________________
/**

Instantiates the seed generator and Minimum builder for the
Fumili minimization method. Produces the Minimum via the
Minimize methods inherited from ModularFunctionMinimizer.

@author Andras Zsenei and Lorenzo Moneta, Creation date: 28 Sep 2004

@ingroup Minuit

*/

class FumiliMinimizer : public ModularFunctionMinimizer {

public:
   /**

      Constructor initializing the FumiliMinimizer by instantiatiating
      the SeedGenerator and MinimumBuilder for the Fumili minimization method.

      @see MnSeedGenerator

      @see FumiliBuilder

   */

   FumiliMinimizer() : fMinSeedGen(MnSeedGenerator()), fMinBuilder(FumiliBuilder()) {}

   ~FumiliMinimizer() override {}

   /**

      Accessor to the seed generator of the minimizer.

      @return A reference to the seed generator used by the minimizer

   */

   const MinimumSeedGenerator &SeedGenerator() const override { return fMinSeedGen; }

   /**

      Accessor to the Minimum builder of the minimizer.

      @return a reference to the Minimum builder.

   */

   const FumiliBuilder &Builder() const override { return fMinBuilder; }
   FumiliBuilder &Builder() override { return fMinBuilder; }

   // for Fumili

   FunctionMinimum Minimize(const FCNBase &, const MnUserParameterState &, const MnStrategy &, unsigned int maxfcn = 0,
                            double toler = 0.1, bool seedingOnly = false) const override;

   FunctionMinimum Minimize(const FCNGradientBase &, const MnUserParameterState &, const MnStrategy &,
                                    unsigned int maxfcn = 0, double toler = 0.1) const override;

   using ModularFunctionMinimizer::Minimize;

private:
   MnSeedGenerator fMinSeedGen;
   FumiliBuilder fMinBuilder;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_FumiliMinimizer
