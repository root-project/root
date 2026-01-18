// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/MnMatrix.h"
#include "Minuit2/MnPrint.h"

#include <cassert>

namespace ROOT {
namespace Minuit2 {

FunctionGradient AnalyticalGradientCalculator::operator()(const MinimumParameters &par) const
{
   // evaluate analytical gradient. take care of parameter transformations

   std::vector<double> grad = fGradFunc.Gradient(fTransformation(par.Vec()));
   assert(grad.size() == fTransformation.Parameters().size());

   MnAlgebraicVector v(par.Vec().size());
   for (unsigned int i = 0; i < par.Vec().size(); i++) {
      unsigned int ext = fTransformation.ExtOfInt(i);
      if (fTransformation.Parameter(ext).HasLimits()) {
         double dd = fTransformation.DInt2Ext(i, par.Vec()(i));
         v(i) = dd * grad[ext];
      } else {
         v(i) = grad[ext];
      }
   }

   MnPrint print("AnalyticalGradientCalculator");
   print.Debug("User given gradient in Minuit2", v);

   // in case we can compute Hessian do not waste re-computing G2 here
   if (!CanComputeG2() || CanComputeHessian())
      return FunctionGradient(v);

   // compute G2 if possible
   MnAlgebraicVector g2(par.Vec().size());
   if (!this->G2(par, g2)) {
      print.Error("Error computing G2");
      return FunctionGradient(v);
   }
   return FunctionGradient(v,g2);
}

FunctionGradient AnalyticalGradientCalculator::operator()(const MinimumParameters &par, const FunctionGradient &) const
{
   // needed from base class
   return (*this)(par);
}

// G2 can be computed directly without Hessian or via the Hessian
bool AnalyticalGradientCalculator::CanComputeG2() const {
   return fGradFunc.HasG2() || fGradFunc.HasHessian();
}

bool AnalyticalGradientCalculator::CanComputeHessian() const {
   return fGradFunc.HasHessian();
}


bool AnalyticalGradientCalculator::Hessian(const MinimumParameters &par, MnAlgebraicSymMatrix & hmat) const
{
   // compute  Hessian using external gradient
   unsigned int n = par.Vec().size();
   assert(hmat.size() == n *(n+1)/2);
   // compute external Hessian and then transform
   std::vector<double> extHessian = fGradFunc.Hessian(fTransformation(par.Vec()));
   if (extHessian.empty()) {
      MnPrint print("AnalyticalGradientCalculator::Hessian");
      print.Info("FCN cannot compute Hessian matrix");
      return false;
   }
   unsigned int next = sqrt(extHessian.size());
   // we need now to transform the matrix from external to internal coordinates
   for (unsigned int i = 0; i < n; i++) {
      unsigned int iext = fTransformation.ExtOfInt(i);
      double dxdi = 1.;
      if (fTransformation.Parameters()[iext].HasLimits()) {
         dxdi = fTransformation.DInt2Ext(i, par.Vec()(i));
      }
      for (unsigned int j = i; j < n; j++) {
         double dxdj = 1.;
         unsigned int jext = fTransformation.ExtOfInt(j);
         if (fTransformation.Parameters()[jext].HasLimits()) {
            dxdj = fTransformation.DInt2Ext(j, par.Vec()(j));
         }
         hmat(i, j) = dxdi * extHessian[iext*next+ jext] * dxdj;
      }
   }
   return true;
}

bool AnalyticalGradientCalculator::G2(const MinimumParameters &par, MnAlgebraicVector &g2) const
{
   // compute G2 using external calculator if available; otherwise fall back to Hessian diagonal
   const unsigned int n = par.Vec().size(); // n is size of internal parameters
   assert(g2.size() == n);

   MnPrint print("AnalyticalGradientCalculator::G2");

   // --- Preferred path: direct G2 from FCN ---
   if (fGradFunc.HasG2()) {
      std::vector<double> extG2 = fGradFunc.G2(fTransformation(par.Vec()));
      if (extG2.empty()) {
         print.Info("FCN cannot compute the 2nd derivatives vector (G2)");
         return false;
      }
      assert(extG2.size() == fTransformation.Parameters().size());
      for (unsigned int i = 0; i < n; i++) {
         const unsigned int iext = fTransformation.ExtOfInt(i);
         if (fTransformation.Parameters()[iext].HasLimits()) {
            const double dxdi = fTransformation.DInt2Ext(i, par.Vec()(i));
            g2(i) = dxdi * dxdi * extG2[iext];
         } else {
            g2(i) = extG2[iext];
         }
      }
      return true;
   }

   // --- Fallback: use Hessian diagonal if FCN provides Hessian ---
   if (!fGradFunc.HasHessian()) {
      print.Info("FCN cannot compute the 2nd derivatives vector (G2) or the Hessian");
      return false;
   }

   std::vector<double> extHessian = fGradFunc.Hessian(fTransformation(par.Vec()));
   if (extHessian.empty()) {
      print.Info("FCN cannot compute Hessian matrix (needed to derive G2)");
      return false;
   }

   // FCNBase::Hessian is expected to return a flat nExt*nExt matrix (row-major).
   const unsigned int nExt = static_cast<unsigned int>(std::lround(std::sqrt(static_cast<double>(extHessian.size()))));
   if (nExt * nExt != extHessian.size()) {
      print.Error("Unexpected Hessian size; cannot derive G2 from Hessian diagonal");
      return false;
   }
   // Sanity check against transformation parameter count
   assert(nExt == fTransformation.Parameters().size());

   for (unsigned int i = 0; i < n; i++) {
      const unsigned int iext = fTransformation.ExtOfInt(i);
      const double diag = extHessian[iext * nExt + iext];
      if (fTransformation.Parameters()[iext].HasLimits()) {
         const double dxdi = fTransformation.DInt2Ext(i, par.Vec()(i));
         g2(i) = dxdi * dxdi * diag;
      } else {
         g2(i) = diag;
      }
   }
   return true;
}

} // namespace Minuit2

} // namespace ROOT
