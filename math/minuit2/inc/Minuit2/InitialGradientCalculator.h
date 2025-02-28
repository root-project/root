// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_InitialGradientCalculator
#define ROOT_Minuit2_InitialGradientCalculator

#include "Minuit2/GradientCalculator.h"

namespace ROOT {

namespace Minuit2 {

class MnUserTransformation;

FunctionGradient calculateInitialGradient(const MinimumParameters &, const MnUserTransformation &, double errorDef);

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_InitialGradientCalculator
