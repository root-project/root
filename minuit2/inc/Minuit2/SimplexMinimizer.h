// @(#)root/minuit2:$Name:  $:$Id: SimplexMinimizer.h,v 1.3.2.3 2005/11/29 11:08:34 moneta Exp $
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

#include <vector>

namespace ROOT {

   namespace Minuit2 {


class SimplexMinimizer : public ModularFunctionMinimizer {

public:

  SimplexMinimizer() : fSeedGenerator(SimplexSeedGenerator()), 
		       fBuilder(SimplexBuilder()) {}

  ~SimplexMinimizer() {}

  const MinimumSeedGenerator& SeedGenerator() const {return fSeedGenerator;}
  const MinimumBuilder& Builder() const {return fBuilder;}

private:

  SimplexSeedGenerator fSeedGenerator;
  SimplexBuilder fBuilder;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_SimplexMinimizer
