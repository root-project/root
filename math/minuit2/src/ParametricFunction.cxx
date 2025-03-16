// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/ParametricFunction.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumParameters.h"

namespace ROOT {

namespace Minuit2 {

std::vector<double> ParametricFunction::GetGradient(std::vector<double> const &x) const
{
   // calculate the numerical gradient (using Numerical2PGradientCalculator)

   // LM:  this I believe is not very efficient
   MnFcn mfcn(*this);

   MnStrategy strategy(1);

   // ????????? how does it know the transformation????????
   std::vector<double> err(x.size());
   err.assign(x.size(), 0.1);
   // need to use parameters
   MnUserParameterState st(x, err);

   Numerical2PGradientCalculator gc(mfcn, st.Trafo(), strategy);
   MnAlgebraicVector xVec{x};
   FunctionGradient g = gc(MinimumParameters{xVec, MnFcnCaller{mfcn}(xVec)});
   const MnAlgebraicVector &grad = g.Vec();
   assert(grad.size() == x.size());
   //  std::cout << "Param Function Gradient " << grad << std::endl;
   return {grad.Data(), grad.Data() + grad.size()};
}

} // namespace Minuit2

} // namespace ROOT
