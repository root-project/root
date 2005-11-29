// @(#)root/minuit2:$Name:  $:$Id: VariableMetricMinimizer.h,v 1.5.2.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_VariableMetricMinimizer
#define ROOT_Minuit2_VariableMetricMinimizer

#include "Minuit2/MnConfig.h"
#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/MnSeedGenerator.h"
#include "Minuit2/VariableMetricBuilder.h"

namespace ROOT {

   namespace Minuit2 {


/** Instantiates the SeedGenerator and MinimumBuilder for
    Variable Metric Minimization method.
 */

class VariableMetricMinimizer : public ModularFunctionMinimizer {

public:

  VariableMetricMinimizer() : fMinSeedGen(MnSeedGenerator()),
			      fMinBuilder(VariableMetricBuilder()) {}
  
  ~VariableMetricMinimizer() {}

  const MinimumSeedGenerator& SeedGenerator() const {return fMinSeedGen;}
  const MinimumBuilder& Builder() const {return fMinBuilder;}

private:

  MnSeedGenerator fMinSeedGen;
  VariableMetricBuilder fMinBuilder;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_VariableMetricMinimizer
