// @(#)root/minuit2:$Name:  $:$Id: ScanBuilder.cpp,v 1.1.6.4 2005/11/29 11:08:35 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/ScanBuilder.h"
#include "Minuit2/MnParameterScan.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MnFcn.h"

namespace ROOT {

   namespace Minuit2 {


FunctionMinimum ScanBuilder::Minimum(const MnFcn& mfcn, const GradientCalculator&, const MinimumSeed& seed, const MnStrategy&, unsigned int, double) const {
  
  MnAlgebraicVector x = seed.Parameters().Vec();
  MnUserParameterState upst(seed.State(), mfcn.Up(), seed.Trafo());
  MnParameterScan Scan(mfcn.Fcn(), upst.Parameters(), seed.Fval());
  double amin = Scan.Fval();
  unsigned int n = seed.Trafo().VariableParameters();
  MnAlgebraicVector dirin(n);
  for(unsigned int i = 0; i < n; i++) {
    unsigned int ext = seed.Trafo().ExtOfInt(i);
    Scan(ext);
    if(Scan.Fval() < amin) {
      amin = Scan.Fval();
      x(i) = seed.Trafo().Ext2int(ext, Scan.Parameters().Value(ext));
    }
    dirin(i) = sqrt(2.*mfcn.Up()*seed.Error().InvHessian()(i,i));
  }

  MinimumParameters mp(x, dirin, amin);
  MinimumState st(mp, 0., mfcn.NumOfCalls());

  return FunctionMinimum(seed, std::vector<MinimumState>(1, st), mfcn.Up());
}

  }  // namespace Minuit2

}  // namespace ROOT
