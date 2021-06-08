// @(#)root/minuit2:$Id$
// Authors: L. Moneta, E.G.P. Bos   2006-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNGradAdapter
#define ROOT_Minuit2_FCNGradAdapter

#include "Minuit2/FCNGradientBase.h"
#include "Minuit2/MnPrint.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

/**

template wrapped class for adapting to FCNBase signature a IGradFunction

@author Lorenzo Moneta, Patrick Bos

@ingroup Minuit

*/


template <class Function>
class FCNGradAdapter : public FCNGradientBase {

public:
   FCNGradAdapter(const Function &f, double up = 1.) : fFunc(f), fUp(up), fGrad(std::vector<double>(fFunc.NDim())),
   fG2(fFunc.hasG2ndDerivative() ? std::vector<double>(fFunc.NDim()) : std::vector<double>(0)),
   fGStep(fFunc.hasGStepSize()   ? std::vector<double>(fFunc.NDim()) : std::vector<double>(0)) {}

   ~FCNGradAdapter() {}


   double operator()(const std::vector<double> &v) const override { return fFunc.operator()(&v[0]); }
   double operator()(const double *v) const { return fFunc.operator()(v); }

   double Up() const override { return fUp; }

   std::vector<double> Gradient(const std::vector<double> &v) const override
   {
      fFunc.Gradient(&v[0], &fGrad[0]);

      MnPrint("FCNGradAdapter").Debug([&](std::ostream &os) {
         os << "gradient in FCNAdapter = {";
         for (unsigned int i = 0; i < fGrad.size(); ++i)
            os << fGrad[i] << (i == fGrad.size() - 1 ? '}' : '\t');
      });
      return fGrad;
   }
   // forward interface
   // virtual double operator()(int npar, double* params,int iflag = 4) const;
   bool CheckGradient() const override { return false; }

   std::vector<double> G2ndDerivative(const std::vector<double>& v) const override {
      fFunc.G2ndDerivative(v.data(), fG2.data());
      return fG2;
   };

   std::vector<double> GStepSize(const std::vector<double>& v) const override {
      fFunc.GStepSize(v.data(), fGStep.data());
      return fGStep;
   };

   bool hasG2ndDerivative() const override {
      return fFunc.hasG2ndDerivative();
   }

   bool hasGStepSize() const override {
      return fFunc.hasGStepSize();
   }

   GradientParameterSpace gradParameterSpace() const override {
      if (fFunc.returnsInMinuit2ParameterSpace()) {
         return GradientParameterSpace::Internal;
      } else {
         return GradientParameterSpace::External;
      }
   }

private:
   const Function &fFunc;
   double fUp;
   mutable std::vector<double> fGrad;
   mutable std::vector<double> fG2;
   mutable std::vector<double> fGStep;
};

} // end namespace Minuit2

} // end namespace ROOT

#endif // ROOT_Minuit2_FCNGradAdapter
