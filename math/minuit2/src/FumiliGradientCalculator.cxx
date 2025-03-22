// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/FumiliGradientCalculator.h"
#include "Minuit2/FumiliFCNBase.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FumiliChi2FCN.h"
#include "Minuit2/FumiliMaximumLikelihoodFCN.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnFcn.h"

namespace ROOT {

namespace Minuit2 {

FumiliGradientCalculator::FumiliGradientCalculator(const FumiliFCNBase &fcn, const MnUserTransformation &trafo, int n)
   : AnalyticalGradientCalculator(fcn, trafo),
   fFcn(fcn),
   fHessian(MnAlgebraicSymMatrix(n))
{
}


FunctionGradient FumiliGradientCalculator::operator()(const MinimumParameters &par) const
{

   // Calculate gradient and Hessian for Fumili using the gradient and Hessian provided
   // by the FCN Fumili function
   // Need to apply internal to external for parameters and the external to int transformation
   // for the return gradient and Hessian

   MnPrint print("FumiliGradientCalculator");

   int nvar = par.Vec().size();
   std::vector<double> extParam = fTransformation(par.Vec());

   // eval Gradient
   FumiliFCNBase &fcn = const_cast<FumiliFCNBase &>(fFcn);

   fcn.EvaluateAll(extParam);

   MnAlgebraicVector v(nvar);
   MnAlgebraicSymMatrix h(nvar);

   const std::vector<double> &fcn_gradient = fFcn.Gradient();
   assert(fcn_gradient.size() == extParam.size());

   // transform gradient and Hessian from external to internal
   std::vector<double> deriv(nvar);
   std::vector<unsigned int> extIndex(nvar);
   for (int i = 0; i < nvar; ++i) {
      extIndex[i] = fTransformation.ExtOfInt(i);
      deriv[i] = 1;
      if (fTransformation.Parameter(extIndex[i]).HasLimits())
         deriv[i] = fTransformation.DInt2Ext(i, par.Vec()(i));

      v(i) = fcn_gradient[extIndex[i]] * deriv[i];

      for (int j = 0; j <= i; ++j) {
         h(i, j) = deriv[i] * deriv[j] * fFcn.Hessian(extIndex[i], extIndex[j]);
      }
   }


   print.Debug([&](std::ostream &os) {
      // compare Fumili with Minuit gradient
      os << "Comparison of Fumili Gradient and standard (numerical) Minuit Gradient (done only when debugging enabled)" << std::endl;
      int plevel = MnPrint::SetGlobalLevel(MnPrint::GlobalLevel()-1);
      Numerical2PGradientCalculator gc(MnFcn{fFcn, fTransformation}, fTransformation, MnStrategy(1));
      FunctionGradient grd2 = gc(par);
      os << "Fumili Gradient:" << v << std::endl;
      os << "Minuit Gradient" << grd2.Vec() << std::endl;
      os << "Fumili Hessian:  " << h << std::endl;
      os << "Numerical g2 " << grd2.G2() << std::endl;
      MnPrint::SetGlobalLevel(plevel);
   });

   // store calculated Hessian
   fHessian = h;
   // compute also g2 from diagonal Hessian
   MnAlgebraicVector g2(nvar);
   G2(par,g2);

   return FunctionGradient(v,g2);
}

FunctionGradient FumiliGradientCalculator::operator()(const MinimumParameters &par, const FunctionGradient &) const

{
   // Needed for interface of base class.
   return this->operator()(par);
}
bool FumiliGradientCalculator::G2(const MinimumParameters &par, MnAlgebraicVector & g2) const
{
   unsigned int n = par.Vec().size();
   if (fHessian.Nrow() != n || g2.size() != n) {
      assert(false);
      return false;
   }
   for (unsigned int i = 0; i < n ; i++) {
      g2(i) = fHessian(i,i);
   }
   return true;
}

bool FumiliGradientCalculator::Hessian(const MinimumParameters &par, MnAlgebraicSymMatrix & h) const
{
   unsigned int n = par.Vec().size();
   if (fHessian.Nrow() != n ) {
      assert(false);
      return false;
   }
   h = fHessian;
   return true;
}

} // namespace Minuit2

} // namespace ROOT
