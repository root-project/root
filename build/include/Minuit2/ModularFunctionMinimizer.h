// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_ModularFunctionMinimizer
#define ROOT_Minuit2_ModularFunctionMinimizer

#include "Minuit2/MnConfig.h"

#include "Minuit2/MnStrategy.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class MinimumSeedGenerator;
class MinimumBuilder;
class MinimumSeed;
class MnFcn;
class GradientCalculator;
class MnUserParameterState;
class MnUserParameters;
class MnUserCovariance;
class FCNBase;
class FumiliFCNBase;
class FunctionMinimum;

//_____________________________________________________________
/**
   Base common class providing the API for all the minimizer
   Various Minimize methods are provided varying on the type of
   FCN function passesd and on the objects used for the parameters
 */
class ModularFunctionMinimizer {

public:

   virtual ~ModularFunctionMinimizer() = default;

   virtual FunctionMinimum Minimize(const FCNBase &, const MnUserParameterState &, const MnStrategy & = MnStrategy{1},
                                    unsigned int maxfcn = 0, double toler = 0.1) const;

   virtual const MinimumSeedGenerator &SeedGenerator() const = 0;
   virtual const MinimumBuilder &Builder() const = 0;
   virtual MinimumBuilder &Builder() = 0;

public:
   virtual FunctionMinimum Minimize(const MnFcn &, const GradientCalculator &, const MinimumSeed &, const MnStrategy &,
                                    unsigned int, double) const;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_ModularFunctionMinimizer
