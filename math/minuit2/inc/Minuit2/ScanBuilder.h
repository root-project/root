// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_ScanBuilder
#define ROOT_Minuit2_ScanBuilder

#include "Minuit2/MinimumBuilder.h"

namespace ROOT {

namespace Minuit2 {

class FunctionMinimum;
class MnFcn;
class MinimumSeed;

/** Performs a minimization using the simplex method of Nelder and Mead
    (ref. Comp. J. 7, 308 (1965)).
 */

class ScanBuilder : public MinimumBuilder {

public:
   ScanBuilder() {}

   ~ScanBuilder() override {}

   FunctionMinimum Minimum(const MnFcn &, const GradientCalculator &, const MinimumSeed &, const MnStrategy &,
                                   unsigned int, double) const override;

private:
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_ScanBuilder
