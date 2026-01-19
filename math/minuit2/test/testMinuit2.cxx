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

class RosenbrockFCNWithoutG2 : public ROOT::Minuit2::FCNBase {

public:
   static int &G2CallCounter()
   {
      static int counter = 0;
      return counter;
   }

   double Up() const override { return 1.0; }

   // Rosenbrock: (1-x)^2 + 100*(y-x^2)^2
   double operator()(std::vector<double> const &p) const override
   {
      const double x = p.at(0);
      const double y = p.at(1);
      const double a = (1.0 - x);
      const double b = (y - x * x);
      return a * a + 100.0 * b * b;
   }

   bool HasGradient() const override { return true; }
   bool HasHessian() const override { return true; }

   // we claim we do NOT provide G2
   bool HasG2() const override { return false; }

   std::vector<double> Gradient(std::vector<double> const &p) const override
   {
      const double x = p.at(0);
      const double y = p.at(1);

      // gradient[x] = 400*x*(x^2 - y) + 2*x - 2
      // gradient[y] = -200*x^2 + 200*y
      std::vector<double> g(2);
      g[0] = 400.0 * x * (x * x - y) + 2.0 * x - 2.0;
      g[1] = -200.0 * x * x + 200.0 * y;
      return g;
   }

   // hxx = 1200*x^2 - 400*y + 2
   // hxy = -400*x
   // hyy = 200
   std::vector<double> Hessian(std::vector<double> const &p) const override
   {
      const double x = p.at(0);
      const double y = p.at(1);

      const double hxx = 1200.0 * x * x - 400.0 * y + 2.0;
      const double hxy = -400.0 * x;
      const double hyy = 200.0;

      std::vector<double> h(2 * 2);
      h[0] = hxx; // (0,0)
      h[1] = hxy; // (0,1)
      h[2] = hyy; // (1,1)
      h[3] = hxy; // (1,0)
      return h;
   }

   std::vector<double> G2(std::vector<double> const &p) const override
   {
      ++G2CallCounter();
      std::vector<double> out(p.size());

      auto hessian = Hessian(p);
      for (std::size_t i = 0; i < p.size(); ++i) {
         out[i] = hessian[i * p.size() + i];
      }

      return out;
   }
};

// Verify that the G2() method is not called when we don't claim to implement
// it in the FCN (HasG2() returns false).
//
// The test needs to use the Rosenbrock function to create some setup where the
// initial Hessian estimate is not positive definite, which triggers Minuit to
// go into the negative G2 line search, where accidental G2() calls used to
// happen.
//
// This covers GitHub issue https://github.com/root-project/root/issues/20913
TEST(Minuit2, NoG2CallsWhenFCHasNoG2)
{
   RosenbrockFCNWithoutG2 fcn;

   ROOT::Minuit2::MnUserParameters upar;
   // initial values and step size
   upar.Add("x", 0.0, 0.1);
   upar.Add("y", 1.0, 0.1);

   ROOT::Minuit2::MnMigrad migrad(fcn, upar);
   ROOT::Minuit2::FunctionMinimum min = migrad();

   // The counter needs to be static, because Minuit2 will internally clone the
   // function.
   EXPECT_EQ(RosenbrockFCNWithoutG2::G2CallCounter(), 0)
      << "The G2() method was called, even though we claim to not implement it!";
}
