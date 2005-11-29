// @(#)root/minuit2:$Name:  $:$Id: VariableMetricEDMEstimator.cpp,v 1.3.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/VariableMetricEDMEstimator.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumError.h"

namespace ROOT {

   namespace Minuit2 {


double similarity(const LAVector&, const LASymMatrix&);

double VariableMetricEDMEstimator::Estimate(const FunctionGradient& g, const MinimumError& e) const {

  if(e.InvHessian().size()  == 1) 
    return 0.5*g.Grad()(0)*g.Grad()(0)*e.InvHessian()(0,0);

  double rho = similarity(g.Grad(), e.InvHessian());
  return 0.5*rho;
}

  }  // namespace Minuit2

}  // namespace ROOT
