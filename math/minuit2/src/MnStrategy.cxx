// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnStrategy.h"

namespace ROOT {

namespace Minuit2 {

MnStrategy::MnStrategy() : fHessCFDG2(0), fHessForcePosDef(1), fStoreLevel(1)
{
   // default strategy
   SetMediumStrategy();
}

MnStrategy::MnStrategy(unsigned int stra) : fHessCFDG2(0), fHessForcePosDef(1), fStoreLevel(1)
{
   // user defined strategy (0, 1, 2, >=3)
   if (stra == 0)
      SetLowStrategy();
   else if (stra == 1)
      SetMediumStrategy();
   else if (stra == 2)
      SetHighStrategy();
   else
      SetVeryHighStrategy();
}

void MnStrategy::SetLowStrategy()
{
   // set low strategy (0) values
   fStrategy = 0;
   SetGradientNCycles(2);
   SetGradientStepTolerance(0.5);
   SetGradientTolerance(0.1);
   SetHessianNCycles(3);
   SetHessianStepTolerance(0.5);
   SetHessianG2Tolerance(0.1);
   SetHessianGradientNCycles(1);
   SetHessianCentralFDMixedDerivatives(0);
}

void MnStrategy::SetMediumStrategy()
{
   // set minimum strategy (1) the default
   fStrategy = 1;
   SetGradientNCycles(3);
   SetGradientStepTolerance(0.3);
   SetGradientTolerance(0.05);
   SetHessianNCycles(5);
   SetHessianStepTolerance(0.3);
   SetHessianG2Tolerance(0.05);
   SetHessianGradientNCycles(2);
   SetHessianCentralFDMixedDerivatives(0);
}

void MnStrategy::SetHighStrategy()
{
   // set high strategy (2)
   fStrategy = 2;
   SetGradientNCycles(5);
   SetGradientStepTolerance(0.1);
   SetGradientTolerance(0.02);
   SetHessianNCycles(7);
   SetHessianStepTolerance(0.1);
   SetHessianG2Tolerance(0.02);
   SetHessianGradientNCycles(6);
   SetHessianCentralFDMixedDerivatives(0);
}

void MnStrategy::SetVeryHighStrategy()
{
    // set very high strategy (3)
    fStrategy = 3;
    SetGradientNCycles(5);
    SetGradientStepTolerance(0.1);
    SetGradientTolerance(0.02);
    SetHessianNCycles(7);
    SetHessianStepTolerance(0.);
    SetHessianG2Tolerance(0.);
    SetHessianGradientNCycles(6);
    SetHessianCentralFDMixedDerivatives(1);
    SetHessianForcePosDef(0);
}

} // namespace Minuit2

} // namespace ROOT
