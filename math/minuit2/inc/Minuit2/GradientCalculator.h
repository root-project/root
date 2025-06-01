// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_GradientCalculator
#define ROOT_Minuit2_GradientCalculator

#include "Minuit2/MnMatrixfwd.h"

namespace ROOT {

namespace Minuit2 {

class MinimumParameters;
class FunctionGradient;

/**
   interface class for gradient calculators
 */
class GradientCalculator {

public:
   virtual ~GradientCalculator() {}

   virtual FunctionGradient operator()(const MinimumParameters &) const = 0;

   virtual FunctionGradient operator()(const MinimumParameters &, const FunctionGradient &) const = 0;

   virtual bool Hessian(const MinimumParameters &, MnAlgebraicSymMatrix &) const { return false;}

   virtual bool G2(const MinimumParameters &, MnAlgebraicVector &) const { return false;}

   static bool SetParallelOMP(bool doParallel = true);

protected:
   static inline bool fDoParallelOMP = false; ///< flag to indicate if parallel OpenMP processing is used
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_GradientCalculator
