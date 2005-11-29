// @(#)root/minuit2:$Name:  $:$Id: ModularFunctionMinimizer.cpp,v 1.23.2.4 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/MinimumSeedGenerator.h"
#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/MinimumBuilder.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnUserParameters.h"
#include "Minuit2/MnUserCovariance.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/FCNGradientBase.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/FumiliFCNBase.h"
#include "Minuit2/FumiliGradientCalculator.h"

namespace ROOT {

   namespace Minuit2 {


// #include "Minuit2/MnUserParametersPrint.h"

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(par, err);
  MnStrategy strategy(stra);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}
  
FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra, unsigned int maxfcn, double toler) const {
  MnUserParameterState st(par, err);
  MnStrategy strategy(stra);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}

// move nrow before cov to avoid ambiguities when using default parameters
FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase& fcn, const std::vector<double>& par, unsigned int nrow, const std::vector<double>& cov, unsigned int stra, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(par, cov, nrow);
  MnStrategy strategy(stra);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}
 
FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase& fcn, const std::vector<double>& par, unsigned int nrow, const std::vector<double>& cov, unsigned int stra, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(par, cov, nrow);
  MnStrategy strategy(stra);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase& fcn, const MnUserParameters& upar, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase& fcn, const MnUserParameters& upar, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase& fcn, const MnUserParameters& upar, const MnUserCovariance& cov, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar, cov);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}

FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase& fcn, const MnUserParameters& upar, const MnUserCovariance& cov, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserParameterState st(upar, cov);
  return Minimize(fcn, st, strategy, maxfcn, toler);
}



FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNBase& fcn, const MnUserParameterState& st, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  // neeed MnUsserFcn for diference int-ext parameters
  MnUserFcn mfcn(fcn, st.Trafo() );
  Numerical2PGradientCalculator gc(mfcn, st.Trafo(), strategy);

  unsigned int npar = st.VariableParameters();
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;
  MinimumSeed mnseeds = SeedGenerator()(mfcn, gc, st, strategy);

  return Minimize(mfcn, gc, mnseeds, strategy, maxfcn, toler);
}


// use Gradient here 
FunctionMinimum ModularFunctionMinimizer::Minimize(const FCNGradientBase& fcn, const MnUserParameterState& st, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  MnUserFcn mfcn(fcn, st.Trafo());
  AnalyticalGradientCalculator gc(fcn, st.Trafo());

  unsigned int npar = st.VariableParameters();
  if(maxfcn == 0) maxfcn = 200 + 100*npar + 5*npar*npar;

  MinimumSeed mnseeds = SeedGenerator()(mfcn, gc, st, strategy);

  return Minimize(mfcn, gc, mnseeds, strategy, maxfcn, toler);
}

// function that actually do the work 

FunctionMinimum ModularFunctionMinimizer::Minimize(const MnFcn& mfcn, const GradientCalculator& gc, const MinimumSeed& seed, const MnStrategy& strategy, unsigned int maxfcn, double toler) const {

  const MinimumBuilder & mb = Builder();
  //std::cout << typeid(&mb).Name() << std::endl;
  return mb.Minimum(mfcn, gc, seed, strategy, maxfcn, toler*mfcn.Up());
}




  }  // namespace Minuit2

}  // namespace ROOT
