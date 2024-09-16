// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnMinimize
#define ROOT_Minuit2_MnMinimize

#include "Minuit2/MnApplication.h"
#include "Minuit2/CombinedMinimizer.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class FCNBase;

/** API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.;
    also used by MnMinos and MnContours;
 */

class MnMinimize : public MnApplication {

public:
   /// construct from FCNBase + MnUserParameterState + MnStrategy
   MnMinimize(const FCNBase &fcn, const MnUserParameterState &par, const MnStrategy &str = MnStrategy{1})
      : MnApplication(fcn, {par}, str), fMinimizer(CombinedMinimizer())
   {
   }

   MnMinimize(const MnMinimize &migr)
      : MnApplication(migr.Fcnbase(), migr.State(), migr.Strategy(), migr.NumOfCalls()), fMinimizer(migr.fMinimizer)
   {
   }

   ModularFunctionMinimizer &Minimizer() override { return fMinimizer; }
   const ModularFunctionMinimizer &Minimizer() const override { return fMinimizer; }

private:
   CombinedMinimizer fMinimizer;

private:
   // forbidden assignment operator
   MnMinimize &operator=(const MnMinimize &) { return *this; }
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnMinimize
