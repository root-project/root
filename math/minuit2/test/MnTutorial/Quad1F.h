// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/FCNGradientBase.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class Quad1F : public FCNGradientBase {

public:
   Quad1F() : fErrorDef(1.) {}

   ~Quad1F() {}

   double operator()(const std::vector<double> &par) const override
   {

      double x = par[0];

      return (x * x);
   }

   std::vector<double> Gradient(const std::vector<double> &par) const override
   {

      double x = par[0];

      return (std::vector<double>(1, 2. * x));
   }

   // G2ndDerivative and GStepSize will not be used since the default hasG2ndDerivative
   // and hasGStepSize functions that return false are not overridden, but these have to
   // be defined, since they are pure virtual functions in FCNGradientBase
   std::vector<double> G2ndDerivative(const std::vector<double>&) const override {
      std::vector<double> g(0);
      return g;
   }
   std::vector<double> GStepSize(const std::vector<double>&) const override {
      std::vector<double> g(0);
      return g;
   }

   void SetErrorDef(double up) override { fErrorDef = up; }

   double Up() const override { return fErrorDef; }

   const FCNBase *Base() const { return this; }

private:
   double fErrorDef;
};

} // namespace Minuit2

} // namespace ROOT
