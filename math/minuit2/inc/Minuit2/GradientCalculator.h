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

   /**
    * Enable parallelization of gradient calculation using OpenMP.
    * This is different from the default parallel mechanism elsewhere (IMT, threads, TBB, ...).
    * It can only be used to minimise thread-safe functions in Minuit2.
    * \param doParallel true to enable, false to disable
    * \note If OPENMP is not available, this function has no effect
    */
   static void DoParallelOMP(bool doParallel = true) { fDoParallelOMP = doParallel; }

protected:
   static inline bool fDoParallelOMP = false; ///< flag to indicate if parallel OpenMP processing is used
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_GradientCalculator
