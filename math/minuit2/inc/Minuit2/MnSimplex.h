// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnSimplex
#define ROOT_Minuit2_MnSimplex

#include "Minuit2/MnApplication.h"
#include "Minuit2/SimplexMinimizer.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class FCNBase;

//_________________________________________________________________________
/**
    API class for minimization using the Simplex method, which does not need and use
    the derivatives of the function, but only function values.
    More information on the minimization method is available
    <A HREF="http://seal.web.cern.ch/mathlibs/documents/minuit/mntutorial.pdf">here</A>.

    It allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.;
 */

class MnSimplex : public MnApplication {

public:
   /// construct from FCNBase + MnUserParameterState + MnStrategy
   MnSimplex(const FCNBase &fcn, const MnUserParameterState &par, const MnStrategy &str = MnStrategy{1})
      : MnApplication(fcn, MnUserParameterState(par), str), fMinimizer(SimplexMinimizer())
   {
   }

   MnSimplex(const MnSimplex &migr)
      : MnApplication(migr.Fcnbase(), migr.State(), migr.Strategy(), migr.NumOfCalls()), fMinimizer(migr.fMinimizer)
   {
   }

   ModularFunctionMinimizer &Minimizer() override { return fMinimizer; }
   const ModularFunctionMinimizer &Minimizer() const override { return fMinimizer; }

private:
   SimplexMinimizer fMinimizer;

private:
   // forbidden assignment of migrad (const FCNBase& = )
   MnSimplex &operator=(const MnSimplex &) { return *this; }
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnSimplex
