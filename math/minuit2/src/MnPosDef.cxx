// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnPosDef.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnPrint.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace ROOT {

namespace Minuit2 {

namespace {

// Return true if the symmetric matrix p is certainly accepted by the eigenvalue test below, i.e. if its smallest
// eigenvalue pmin satisfies pmin > epspdf * max(|pmax|, 1). This avoids the full O(n^3) eigen decomposition with
// mneigen in the common case where the matrix is well inside the positive-definite region. A false return value
// is not a verdict: the caller then falls back to the eigenvalue computation.
//
// The test is a Cholesky decomposition of p - shift * 1. Its success proves that all eigenvalues of p are larger
// than shift. The Gershgorin bound gives max(|pmax|, 1) <= bound, so it suffices that shift >= epspdf * bound.
// The shift also contains margins for the convergence tolerance of mneigen (relative precision 1e-6 on the
// tridiagonal matrix, whose elements are bounded by the Gershgorin bound too) and for the rounding errors of the
// Cholesky decomposition of a matrix with unit diagonal, so that a successful test always leads to the same
// decision as the eigenvalues that mneigen would have computed.
bool IsCertainlyPosDef(const MnAlgebraicSymMatrix &p, double epspdf)
{
   const unsigned int n = p.Nrow();

   double bound = 1.;
   for (unsigned int i = 0; i < n; i++) {
      double rowSum = 0.;
      for (unsigned int j = 0; j < n; j++)
         rowSum += std::abs(p(i, j));
      bound = std::max(bound, rowSum);
   }
   const double shift = (epspdf + 1.e-5 + (n + 1.) * n * std::numeric_limits<double>::epsilon()) * bound;

   // Cholesky decomposition in the packed storage of the matrix: element (i, j) with j <= i is at i * (i + 1) / 2 + j.
   std::vector<double> l(p.Data(), p.Data() + p.size());
   for (unsigned int i = 0; i < n; i++) {
      double *li = l.data() + i * (i + 1) / 2;
      for (unsigned int j = 0; j <= i; j++) {
         const double *lj = l.data() + j * (j + 1) / 2;
         double sum = li[j];
         if (i == j)
            sum -= shift;
         for (unsigned int k = 0; k < j; k++)
            sum -= li[k] * lj[k];
         if (i == j) {
            if (!(sum > 0.))
               return false;
            li[i] = std::sqrt(sum);
         } else {
            li[j] = sum / lj[j];
         }
      }
   }
   return true;
}

} // namespace

MinimumState MnPosDef::operator()(const MinimumState &st, const MnMachinePrecision &prec) const
{
   // interface from minimum state
   MinimumError err = (*this)(st.Error(), prec);
   return MinimumState(st.Parameters(), err, st.Gradient(), st.Edm(), st.NFcn());
}

MinimumError MnPosDef::operator()(const MinimumError &e, const MnMachinePrecision &prec) const
{
   MnPrint print("MnPosDef");

   // make error matrix positive defined returning a new corrected minimum error state

   MnAlgebraicSymMatrix err(e.InvHessian());
   if (err.size() == 1 && err(0, 0) < prec.Eps()) {
      err(0, 0) = 1.;
      return MinimumError(err, MinimumError::MnMadePosDef);
   }
   if (err.size() == 1 && err(0, 0) > prec.Eps()) {
      return e;
   }
   //   std::cout<<"MnPosDef init matrix= "<<err<<std::endl;

   double epspdf = std::max(1.e-12, prec.Eps2()); // should this lower bound be configurable?
   double dgmin = err(0, 0);

   for (unsigned int i = 0; i < err.Nrow(); i++) {
      if (err(i, i) <= 0 /* prec.Eps2() */)
         print.Warn("non-positive diagonal element in covariance matrix[", i, "] =", err(i, i));

      if (err(i, i) < dgmin)
         dgmin = err(i, i);
   }
   double dg = 0.;
   if (dgmin <= 0) {
      // dg = 1. + epspdf - dgmin;
      dg = 0.5 + epspdf - dgmin;
      //     dg = 0.5*(1. + epspdf - dgmin);

      print.Warn("Added to diagonal of Error matrix a value", dg);

      // std::cout << "Error matrix " << err << std::endl;
   }

   MnAlgebraicVector s(err.Nrow());
   MnAlgebraicSymMatrix p(err.Nrow());
   for (unsigned int i = 0; i < err.Nrow(); i++) {
      err(i, i) += dg;
      if (err(i, i) < 0.)
         err(i, i) = 1.;
      s(i) = 1. / std::sqrt(err(i, i));
      for (unsigned int j = 0; j <= i; j++) {
         p(i, j) = err(i, j) * s(i) * s(j);
      }
   }

   // std::cout<<"MnPosDef p: "<<p<<std::endl;
   if (IsCertainlyPosDef(p, epspdf))
      return MinimumError(err, e.Dcovar());

   MnAlgebraicVector eval = eigenvalues(p);
   double pmin = eval(0);
   double pmax = eval(eval.size() - 1);
   // std::cout<<"pmin= "<<pmin<<" pmax= "<<pmax<<std::endl;
   pmax = std::max(std::fabs(pmax), 1.);
   if (pmin > epspdf * pmax)
      return MinimumError(err, e.Dcovar());

   double pAdd = 0.001 * pmax - pmin;

   for (unsigned int i = 0; i < err.Nrow(); i++)
      err(i, i) *= (1. + pAdd);

   print.Debug([&](std::ostream &os) {
      os << "Eigenvalues:";
      for (unsigned i = 0; i < err.Nrow(); ++i)
         os << "\n  " << eval(i);
   });

   //   std::cout<<"MnPosDef final matrix: "<<err<<std::endl;

   print.Warn("Matrix forced pos-def by adding to diagonal", pAdd);

   return MinimumError(err, MinimumError::MnMadePosDef);
}

} // namespace Minuit2

} // namespace ROOT
