// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNAdapter
#define ROOT_Minuit2_FCNAdapter

#include "Minuit2/FCNBase.h"

#include <ROOT/RSpan.hxx>

#include <vector>
#include <functional>

namespace ROOT {

namespace Minuit2 {

/**


Wrapper class for std::function objects, adapting to FCNBase signature.

@author Lorenzo Moneta

@ingroup Minuit

*/

class FCNAdapter : public FCNBase {

public:
   FCNAdapter(std::function<double(double const *)> f, double up = 1.) : fUp(up), fFunc(std::move(f)) {}

   bool HasGradient() const override { return bool(fGradFunc); }
   bool HasG2() const override { return bool(fG2Func); }
   bool HasHessian() const override { return bool(fHessianFunc); }

   double operator()(std::vector<double> const &v) const override { return fFunc(v.data()); }

   double Up() const override { return fUp; }

   std::vector<double> Gradient(std::vector<double> const &v) const override
   {
      std::vector<double> output(v.size());
      fGradFunc(v.data(), output.data());
      return output;
   }

   /// return second derivatives (diagonal of the Hessian matrix)
   std::vector<double> G2(std::vector<double> const &x) const override
   {
      std::vector<double> output;
      if (fG2Func)
         return fG2Func(x);
      if (fHessianFunc) {
         std::size_t n = x.size();
         output.resize(n);
         if (fHessian.empty())
            fHessian.resize(n * n);
         fHessianFunc(x, fHessian.data());
         if (!fHessian.empty()) {
            // get diagonal element of h
            for (unsigned int i = 0; i < n; i++)
               output[i] = fHessian[i * n + i];
         }
      }
      return output;
   }

   /// compute Hessian. Return Hessian as a std::vector of size(n*n)
   std::vector<double> Hessian(std::vector<double> const &x) const override
   {
      std::vector<double> output;
      if (fHessianFunc) {
         std::size_t n = x.size();
         output.resize(n * n);
         bool ret = fHessianFunc(x, output.data());
         if (!ret) {
            output.clear();
            fHessianFunc = nullptr;
         }
      }

      return output;
   }

   void SetGradientFunction(std::function<void(double const *, double *)> f) { fGradFunc = std::move(f); }
   void SetG2Function(std::function<std::vector<double>(std::vector<double> const &)> f) { fG2Func = std::move(f); }
   void SetHessianFunction(std::function<bool(std::vector<double> const &, double *)> f)
   {
      fHessianFunc = std::move(f);
   }

   void SetErrorDef(double up) override { fUp = up; }

private:
   using Function = std::function<double(double const *)>;
   using GradFunction = std::function<void(double const *, double *)>;
   using G2Function = std::function<std::vector<double>(std::vector<double> const &)>;
   using HessianFunction = std::function<bool(std::vector<double> const &, double *)>;

   double fUp = 1.;
   mutable std::vector<double> fHessian;

   Function fFunc;
   GradFunction fGradFunc;
   G2Function fG2Func;
   mutable HessianFunction fHessianFunc;
};

} // end namespace Minuit2

} // end namespace ROOT

#endif // ROOT_Minuit2_FCNAdapter
