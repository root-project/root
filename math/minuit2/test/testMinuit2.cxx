// Tests for Minuit2, covering former bugs and regressions
//
// Author: Jonas Rembser, CERN 01/2026

#include <Minuit2/FCNBase.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnPrint.h>

#include <gtest/gtest.h>

class QuadraticFCNBase : public ROOT::Minuit2::FCNBase {
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
   bool HasG2() const override { return false; }

   std::vector<double> Gradient(const std::vector<double> &p) const override
   {
      // clang-format off
      return {
         2 * p[0] + 2 * p[1] + 4 * p[2],
         20 * p[1] + 2 * p[0] + 8 * p[2],
         200 * p[2] + 4 * p[0] + 8 * p[1]
      };
      // clang-format on
   }

   double Up() const override { return 1.0; }
};

class QuadraticFCNWithHessian : public QuadraticFCNBase {
public:
   bool HasHessian() const override { return true; }

   std::vector<double> Hessian(const std::vector<double> &) const override
   {
      // Row-major 3x3 Hessian in *external* parameter space
      // clang-format off
      return {
         2,   2,   4,
         2,  20,   8,
         4,   8, 200
      };
      // clang-format on
   }
};

class QuadraticFCNNoHessian : public QuadraticFCNBase {
public:
   bool HasHessian() const override { return false; }
};

// ----------------------------------------------------------------------
// Shared test helper
// ----------------------------------------------------------------------

template <typename FCN>
void RunHessianExternalIndexingTest(const FCN &fcn, double tolerance)
{
   using namespace ROOT::Minuit2;

   MnUserParameters upar;
   upar.Add("x", 1.0, 0.1);
   upar.Add("y", 2.0, 0.1);
   upar.Add("z", 3.0, 0.1);

   // Break internal/external identity mapping
   upar.Fix("x");

   MnMigrad migrad(fcn, upar);
   FunctionMinimum min = migrad();

   const auto &hessian = min.Error().Hessian();

   // Only floating parameters must appear in the Hessian
   ASSERT_EQ(hessian.Nrow(), 2);

   // Internal parameters are (y, z)
   //
   // Expected Hessian:
   // [20   8]
   // [ 8 200]

   EXPECT_NEAR(hessian(0, 0), 20.0, tolerance);
   EXPECT_NEAR(hessian(0, 1), 8.0, tolerance);
   EXPECT_NEAR(hessian(1, 1), 200.0, tolerance);
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

// Check that for both the numeric an analytical case, Hessians are only
// computed for the floating parameters.

TEST(Minuit2, HessianExternalIndexing_Numeric)
{
   QuadraticFCNNoHessian fcn;

   // Numeric second derivatives are not exact: relax tolerance
   RunHessianExternalIndexingTest(fcn, /*tolerance=*/1e-6);
}

TEST(Minuit2, HessianExternalIndexing_Analytic)
{
   QuadraticFCNWithHessian fcn;

   // The tolerance is non-zero to account for floating point effects
   RunHessianExternalIndexingTest(fcn, /*tolerance=*/1e-12);
}
