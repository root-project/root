// @(#)root/minuit2:$Name:  $:$Id: MnStrategy.cpp,v 1.3.6.3 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnStrategy.h"

namespace ROOT {

   namespace Minuit2 {


//default strategy
MnStrategy::MnStrategy() {
  SetMediumStrategy();
}

//user defined strategy (0, 1, >=2)
MnStrategy::MnStrategy(unsigned int stra) {
  if(stra == 0) SetLowStrategy();
  else if(stra == 1) SetMediumStrategy();
  else SetHighStrategy();
}

void MnStrategy::SetLowStrategy() {
  fStrategy = 0;
  SetGradientNCycles(2);
  SetGradientStepTolerance(0.5);
  SetGradientTolerance(0.1);
  SetHessianNCycles(3);
  SetHessianStepTolerance(0.5);
  SetHessianG2Tolerance(0.1);
  SetHessianGradientNCycles(1);
}

void MnStrategy::SetMediumStrategy() {
  fStrategy = 1;
  SetGradientNCycles(3);
  SetGradientStepTolerance(0.3);
  SetGradientTolerance(0.05);
  SetHessianNCycles(5);
  SetHessianStepTolerance(0.3);
  SetHessianG2Tolerance(0.05);
  SetHessianGradientNCycles(2);
}

void MnStrategy::SetHighStrategy() {
  fStrategy = 2;
  SetGradientNCycles(5);
  SetGradientStepTolerance(0.1);
  SetGradientTolerance(0.02);
  SetHessianNCycles(7);
  SetHessianStepTolerance(0.1);
  SetHessianG2Tolerance(0.02);
  SetHessianGradientNCycles(6);
}

  }  // namespace Minuit2

}  // namespace ROOT
