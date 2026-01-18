// Tests for Minuit2, covering former bugs and regressions
//
// Author: Jonas Rembser, CERN 01/2026


#include <Minuit2/FCNBase.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnPrint.h>

#include <gtest/gtest.h>

#include <cassert>
#include <vector>
#include <cmath>
#include <iostream>

class QuadraticFCN : public ROOT::Minuit2::FCNBase {
public:
   double operator()(const std::vector<double> &p) const override
   {
      const double x = p[0];
      const double y = p[1];
      const double z = p[2];
      // clang-format off
      return x*x + 10*y*y + 100*z*z + 2*x*y + 4*x*z + 8*y*z;
      // clang-format on
   }

   bool HasGradient() const override { return true; }
   bool HasHessian() const override { return true; }
   bool HasG2() const override { return false; }

   std::vector<double> Gradient(const std::vector<double> &p) const override
   {
      // clang-format off
      return {   2 * p[0] + 2 * p[1] + 4 * p[2],
                20 * p[1] + 2 * p[0] + 8 * p[2],
               200 * p[2] + 4 * p[0] + 8 * p[1] };
      // clang-format on
   }

   std::vector<double> Hessian(const std::vector<double> &) const override
   {
      // Row-major 3x3
      // clang-format off
      return { 2,  2,   4,
               2, 20,   8,
               4,  8, 200 };
      // clang-format on
   }

   double Up() const override { return 1.0; }
};

// This test guards against an indexing bug in AnalyticalGradientCalculator::Hessian.
// When parameters are fixed or otherwise constrained, Minuit2's internal parameter
// indices no longer match the external FCN parameter indices. The analytical Hessian
// provided by the FCN is defined in *external* parameter space and must therefore be
// indexed using external indices (iext, jext), not internal ones (i, j).
//
// Historically, the Hessian transformation incorrectly used internal indices when
// accessing the external Hessian matrix, which silently produced wrong second
// derivatives whenever the internal↔external mapping was non-identity.
//
// Fixing parameter "x" breaks the identity mapping, so this test fails if external
// indices are not used correctly and passes only when the Hessian transformation
// is implemented properly.
TEST(Minuit2, HessianExternalIndexing)
{
   using namespace ROOT::Minuit2;

   QuadraticFCN fcn;

   MnUserParameters upar;
   upar.Add("x", 1.0, 0.1);
   upar.Add("y", 2.0, 0.1);
   upar.Add("z", 3.0, 0.1);

   // Fix x to break the internal/external identity mapping
   upar.Fix("x");

   MnMigrad migrad(fcn, upar);
   FunctionMinimum min = migrad();

   const auto &h = min.Error().Hessian();

   // Internal parameters are (y, z)
   // Expected Hessian:
   // [20  8]
   // [ 8 200]
   //
   // Packed: [20, 8, 200]

   EXPECT_DOUBLE_EQ(h(0, 0), 20.0);
   EXPECT_DOUBLE_EQ(h(0, 1), 8.0);
   EXPECT_DOUBLE_EQ(h(1, 1), 200.0);
}
