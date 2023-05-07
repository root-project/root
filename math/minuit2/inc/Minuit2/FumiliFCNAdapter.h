// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FumiliFCNAdapter
#define ROOT_Minuit2_FumiliFCNAdapter

#include "Minuit2/FumiliFCNBase.h"

#include "Math/FitMethodFunction.h"

#include "Minuit2/MnPrint.h"

// #ifndef ROOT_Math_Util
// #include "Math/Util.h"
// #endif

#include <cmath>
#include <cassert>
#include <vector>

namespace ROOT {

namespace Minuit2 {

/**


template wrapped class for adapting to FumiliFCNBase signature

@author Lorenzo Moneta

@ingroup Minuit

*/

template <class Function>
class FumiliFCNAdapter : public FumiliFCNBase {

public:
   //   typedef ROOT::Math::FitMethodFunction Function;
   typedef typename Function::Type_t Type_t;

   FumiliFCNAdapter(const Function &f, unsigned int ndim, double up = 1.) : FumiliFCNBase(ndim), fFunc(f), fUp(up) {}

   double operator()(std::vector<double> const &v) const override { return fFunc.operator()(&v[0]); }
   double operator()(const double *v) const { return fFunc.operator()(v); }
   double Up() const override { return fUp; }

   void SetErrorDef(double up) override { fUp = up; }


   /**
       evaluate gradient hessian and function value needed by Fumili
     */
   void EvaluateAll(std::vector<double> const &v) override;

private:
   const Function &fFunc;
   double fUp;
};

template <class Function>
void FumiliFCNAdapter<Function>::EvaluateAll(std::vector<double> const &v)
{
   MnPrint print("FumiliFCNAdapter");

   // evaluate all elements
   unsigned int npar = Dimension();
   if (npar != v.size())
      print.Error("npar", npar, "v.size()", v.size());
   assert(npar == v.size());
   // must distinguish case of likelihood or LS

   std::vector<double> &grad = Gradient();
   std::vector<double> &hess = Hessian();
   // reset
   assert(grad.size() == npar);
   grad.assign(npar, 0.0);
   hess.assign(hess.size(), 0.0);

   unsigned int ndata = fFunc.NPoints();

   std::vector<double> gf(npar);
   std::vector<double> h(hess.size());

   // loop on the data points

   // if FCN is of type least-square
   if (fFunc.Type() == Function::kLeastSquare) {
      print.Debug("Chi2 FCN: Evaluate gradient and Hessian");

      for (unsigned int i = 0; i < ndata; ++i) {
         // calculate data element and gradient (no need to compute Hessian)
         double fval = fFunc.DataElement(&v.front(), i, &gf[0]);

         for (unsigned int j = 0; j < npar; ++j) {
            grad[j] += 2. * fval * gf[j];
            for (unsigned int k = j; k < npar; ++k) {
               int idx = j + k * (k + 1) / 2;
               hess[idx] += 2.0 * gf[j] * gf[k];
            }
         }
      }
   } else if (fFunc.Type() == Function::kLogLikelihood) {
      print.Debug("LogLikelihood FCN: Evaluate gradient and Hessian");
      for (unsigned int i = 0; i < ndata; ++i) {

         // calculate data element and gradient: returns derivative of log(pdf)
         fFunc.DataElement(&v.front(), i, &gf[0]);

         for (unsigned int j = 0; j < npar; ++j) {
            double gfj = gf[j];
            grad[j] -= gfj; // need a minus sign since is a NLL
            for (unsigned int k = j; k < npar; ++k) {
               int idx = j + k * (k + 1) / 2;
               hess[idx] += gfj * gf[k];
            }
         }
      }
   } else if (fFunc.Type() == Function::kPoissonLikelihood) {
      print.Debug("Poisson Likelihood FCN: Evaluate gradient and Hessian");
      // for Poisson need Hessian computed in DataElement since one needs the bin expected value ad bin observed value
      for (unsigned int i = 0; i < ndata; ++i) {
         // calculate data element and gradient
         fFunc.DataElement(&v.front(), i, gf.data(), h.data());
         for (size_t j = 0; j < npar; ++j) {
            grad[j] += gf[j];
            for (unsigned int k = j; k < npar; ++k) {
               int idx = j + k * (k + 1) / 2;
               hess[idx] += h[idx];
            }
         }
      }
   } else {
      print.Error("Type of fit method is not supported, it must be chi2 or log-likelihood or Poisson Likelihood");
   }
}

} // end namespace Minuit2

} // end namespace ROOT

#endif // ROOT_Minuit2_FCNAdapter
