// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_SimplexMinimizer
#define ROOT_Minuit2_SimplexMinimizer

#include "Minuit2/MnConfig.h"
#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/SimplexBuilder.h"
#include "Minuit2/SimplexSeedGenerator.h"

namespace ROOT {

namespace Minuit2 {

//_____________________________________________________________
/**
   Class implementing the required methods for a minimization using Simplex.
   API is provided in the upper ROOT::Minuit2::ModularFunctionMinimizer class
 */

class SimplexMinimizer : public ModularFunctionMinimizer {

public:
   SimplexMinimizer() : fSeedGenerator(SimplexSeedGenerator()), fBuilder(SimplexBuilder()) {}

   ~SimplexMinimizer() override {}

   const MinimumSeedGenerator &SeedGenerator() const override { return fSeedGenerator; }
   const MinimumBuilder &Builder() const override { return fBuilder; }
   MinimumBuilder &Builder() override { return fBuilder; }

private:
   SimplexSeedGenerator fSeedGenerator;
   SimplexBuilder fBuilder;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_SimplexMinimizer
