// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNGradAdapter
#define ROOT_Minuit2_FCNGradAdapter

#include "Minuit2/FCNGradientBase.h"
#include "Minuit2/MnPrint.h"

#include <vector>
#include <functional>

namespace ROOT {

namespace Minuit2 {

/**


template wrapped class for adapting to FCNBase signature a IGradFunction

@author Lorenzo Moneta

@ingroup Minuit

*/

template <class Function>
class FCNGradAdapter : public FCNGradientBase {

public:
   FCNGradAdapter(const Function &f, double up = 1.) : fFunc(f), fUp(up), fGrad(std::vector<double>(fFunc.NDim())) {}

   ~FCNGradAdapter() override {}

   double operator()(const std::vector<double> &v) const override { return fFunc.operator()(&v[0]); }
   double operator()(const double *v) const { return fFunc.operator()(v); }

   double Up() const override { return fUp; }

   std::vector<double> Gradient(const std::vector<double> &v) const override
   {
      fFunc.Gradient(&v[0], &fGrad[0]);
      return fGrad;
   }
   std::vector<double> GradientWithPrevResult(const std::vector<double> &v, double *previous_grad, double *previous_g2,
                                              double *previous_gstep) const override
   {
      fFunc.GradientWithPrevResult(&v[0], &fGrad[0], previous_grad, previous_g2, previous_gstep);
      return fGrad;
   }
   // forward interface
   // virtual double operator()(int npar, double* params,int iflag = 4) const;
   bool CheckGradient() const override { return false; }

   GradientParameterSpace gradParameterSpace() const override {
      if (fFunc.returnsInMinuit2ParameterSpace()) {
         return GradientParameterSpace::Internal;
      } else {
         return GradientParameterSpace::External;
      }
   }

   /// return second derivatives (diagonal of the Hessian matrix)
   std::vector<double> G2(const std::vector<double> & x) const override {
      if (fG2Func)
         return fG2Func(x);
      if (fHessianFunc) {
         unsigned int n = fFunc.NDim();
         if (fG2Vec.empty() ) fG2Vec.resize(n);
         if (fHessian.empty() ) fHessian.resize(n*n);
         fHessianFunc(x,fHessian.data());
         if (!fHessian.empty()) {
            // get diagonal element of h
            for (unsigned int i = 0; i < n; i++)
               fG2Vec[i] = fHessian[i*n+i];
         }
         else fG2Vec.clear();
      }
      else
         if (!fG2Vec.empty()) fG2Vec.clear();
      return fG2Vec;
   }

   /// compute Hessian. Return Hessian as a std::vector of size(n*n)
   std::vector<double> Hessian(const std::vector<double> & x ) const override {
      unsigned int n = fFunc.NDim();
      if (fHessianFunc) {
         if (fHessian.empty() ) fHessian.resize(n * n);
         bool ret = fHessianFunc(x,fHessian.data());
         if (!ret) {
            fHessian.clear();
            fHessianFunc = nullptr;
         }
      } else {
         fHessian.clear();
      }

      return fHessian;
   }

   bool HasG2() const override {
      return bool(fG2Func);
   }
   bool HasHessian() const override {
      return bool(fHessianFunc);
   }

   template<class Func>
   void SetG2Function(Func f) { fG2Func = f;}

   template<class Func>
   void SetHessianFunction(Func f) { fHessianFunc = f;}

   void SetErrorDef(double up) override { fUp = up; }

private:
   const Function &fFunc;
   double fUp;
   mutable std::vector<double> fGrad;
   mutable std::vector<double> fHessian;
   mutable std::vector<double> fG2Vec;

   std::function<std::vector<double>(const std::vector<double> &)> fG2Func;
   mutable std::function<bool(const std::vector<double> &, double *)> fHessianFunc;
};

} // end namespace Minuit2

} // end namespace ROOT

#endif // ROOT_Minuit2_FCNGradAdapter
