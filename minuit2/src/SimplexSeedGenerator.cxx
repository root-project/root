// @(#)root/minuit2:$Name:  $:$Id: SimplexSeedGenerator.cpp,v 1.2.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/SimplexSeedGenerator.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/VariableMetricEDMEstimator.h"

namespace ROOT {

   namespace Minuit2 {


MinimumSeed SimplexSeedGenerator::operator()(const MnFcn& fcn, const GradientCalculator&, const MnUserParameterState& st, const MnStrategy& stra) const {

  unsigned int n = st.VariableParameters();
  const MnMachinePrecision& prec = st.Precision();

  // initial starting values
  MnAlgebraicVector x(n);
  for(unsigned int i = 0; i < n; i++) x(i) = st.IntParameters()[i];
  double fcnmin = fcn(x);
  MinimumParameters pa(x, fcnmin);
  InitialGradientCalculator igc(fcn, st.Trafo(), stra);
  FunctionGradient dgrad = igc(pa);
  MnAlgebraicSymMatrix mat(n);
  double dcovar = 1.;
  for(unsigned int i = 0; i < n; i++)	
    mat(i,i) = (fabs(dgrad.G2()(i)) > prec.Eps2() ? 1./dgrad.G2()(i) : 1.);
  MinimumError err(mat, dcovar);
  double edm = VariableMetricEDMEstimator().Estimate(dgrad, err);
  MinimumState state(pa, err, dgrad, edm, fcn.NumOfCalls());
  
  return MinimumSeed(state, st.Trafo());		     
}

MinimumSeed SimplexSeedGenerator::operator()(const MnFcn& fcn, const AnalyticalGradientCalculator& gc, const MnUserParameterState& st, const MnStrategy& stra) const {
  return (*this)(fcn, (const GradientCalculator&)(gc), st, stra);
}

  }  // namespace Minuit2

}  // namespace ROOT
