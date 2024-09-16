// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnScan
#define ROOT_Minuit2_MnScan

#include "Minuit2/MnApplication.h"
#include "Minuit2/ScanMinimizer.h"

#include <vector>
#include <utility>

namespace ROOT {

namespace Minuit2 {

class FCNBase;

//_______________________________________________________________________
/**
    API class for minimization using a scan method to find the minimum;
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.;

 */

class MnScan : public MnApplication {

public:
   /// construct from FCNBase + MnUserParameterState + MnStrategy
   MnScan(const FCNBase &fcn, const MnUserParameterState &par, const MnStrategy &str = MnStrategy{})
      : MnApplication(fcn, MnUserParameterState(par), str), fMinimizer(ScanMinimizer())
   {
   }

   MnScan(const MnScan &migr)
      : MnApplication(migr.Fcnbase(), migr.State(), migr.Strategy(), migr.NumOfCalls()), fMinimizer(migr.fMinimizer)
   {
   }

   ModularFunctionMinimizer &Minimizer() override { return fMinimizer; }
   const ModularFunctionMinimizer &Minimizer() const override { return fMinimizer; }

   std::vector<std::pair<double, double>>
   Scan(unsigned int par, unsigned int maxsteps = 41, double low = 0., double high = 0.);

private:
   ScanMinimizer fMinimizer;

   /// forbidden assignment (const FCNBase& = )
   MnScan &operator=(const MnScan &) { return *this; }
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnScan
