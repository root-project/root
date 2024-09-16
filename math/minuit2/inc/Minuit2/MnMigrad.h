// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnMigrad
#define ROOT_Minuit2_MnMigrad

#include "Minuit2/MnApplication.h"
#include "Minuit2/VariableMetricMinimizer.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class FCNBase;

//_____________________________________________________________________________
/**
   API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.;
    also used by MnMinos and MnContours;
 */

class MnMigrad : public MnApplication {

public:
   /// construct from FCNBase + MnUserParameterState + MnStrategy
   MnMigrad(const FCNBase &fcn, const MnUserParameterState &par, const MnStrategy &str = MnStrategy{1})
      : MnApplication(fcn, {par}, str), fMinimizer(VariableMetricMinimizer())
   {
   }

   /// Copy constructor, copy shares the reference to the same FCNBase in MnApplication
   MnMigrad(const MnMigrad &) = default;

   // Copy assignment deleted, since MnApplication has unassignable reference to FCNBase
   MnMigrad &operator=(const MnMigrad &) = delete;

   ModularFunctionMinimizer &Minimizer() override { return fMinimizer; }
   const ModularFunctionMinimizer &Minimizer() const override { return fMinimizer; }

private:
   VariableMetricMinimizer fMinimizer;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnMigrad
