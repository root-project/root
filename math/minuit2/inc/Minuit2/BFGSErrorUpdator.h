// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_BFGSErrorUpdator
#define ROOT_Minuit2_BFGSErrorUpdator

#include "Minuit2/MinimumErrorUpdator.h"

namespace ROOT {

namespace Minuit2 {

/**
   Update of the covariance matrix for the Variable Metric minimizer (MIGRAD)
 */
class BFGSErrorUpdator : public MinimumErrorUpdator {

public:
   BFGSErrorUpdator() {}

   ~BFGSErrorUpdator() override {}

   MinimumError Update(const MinimumState &, const MinimumParameters &, const FunctionGradient &) const override;

private:
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_BFGSErrorUpdator
