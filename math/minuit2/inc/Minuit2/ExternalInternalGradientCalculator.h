// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_ExternalInternalGradientCalculator
#define ROOT_Minuit2_ExternalInternalGradientCalculator

#include "Minuit2/AnalyticalGradientCalculator.h"

namespace ROOT {

namespace Minuit2 {

class FCNGradientBase;
class MnUserTransformation;

/// Similar to the AnalyticalGradientCalculator, the ExternalInternalGradientCalculator
/// supplies Minuit with an externally calculated gradient. The main difference is that
/// ExternalInternalGradientCalculator expects that the external gradient calculator does
/// things in Minuit2-internal parameter space, which means many int2ext and ext2int
/// transformation steps are not necessary. This avoids loss of precision in some cases,
/// where trigonometrically transforming parameters back and forth can lose a few bits of
/// floating point precision on every pass.

class ExternalInternalGradientCalculator : public AnalyticalGradientCalculator {

public:
   ExternalInternalGradientCalculator(const FCNGradientBase &fcn, const MnUserTransformation &state)
      : AnalyticalGradientCalculator(fcn, state)
   {
   }

   ~ExternalInternalGradientCalculator() override {}

   FunctionGradient operator()(const MinimumParameters &) const override;

   FunctionGradient operator()(const MinimumParameters &, const FunctionGradient &) const override;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_ExternalInternalGradientCalculator
