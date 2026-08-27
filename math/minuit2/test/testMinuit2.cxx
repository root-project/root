// Tests for Minuit2, covering former bugs and regressions
//
// Author: Jonas Rembser, CERN 01/2026

#include <Minuit2/FCNBase.h>
#include <Minuit2/MnMigrad.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/MnUserParameterState.h>
#include <Minuit2/MnUserTransformation.h>
#include <Minuit2/AnalyticalGradientCalculator.h>
#include <Minuit2/MinimumParameters.h>
#include <Minuit2/FunctionGradient.h>
#include <Minuit2/MnMatrix.h>
#include <Minuit2/FunctionMinimum.h>
#include <Minuit2/MnPrint.h>

#include <gtest/gtest.h>

#include <cmath>

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

// Least-squares fit of a simplified, piecewise binding-kinetics model with an
// optional analytical Hessian. The model gradient and Hessian are exact. The
// two rate parameters (ka, kd) have a physical lower limit of zero and are
// started far below the solution.
//
// When parameters have limits, the internal<->external parameter transformation
// is non-linear, so converting the user-provided external Hessian (and G2) to
// internal coordinates requires an extra curvature term proportional to the
// external gradient. Without it, the seed covariance is corrupted and, for this
// configuration, Migrad silently diverges to a completely wrong minimum while
// still reporting validity.
class BindingKineticsFCN : public ROOT::Minuit2::FCNBase {
public:
   explicit BindingKineticsFCN(bool hasHessian) : fHasHessian(hasHessian) {}

   double Up() const override { return 1.0; }
   bool HasGradient() const override { return true; }
   bool HasHessian() const override { return fHasHessian; }

   double operator()(std::vector<double> const &p) const override
   {
      double chi2 = 0.0;
      for (std::size_t i = 0; i < fX.size(); ++i) {
         const double r = (fY[i] - Model(fX[i], p)) / fErr;
         chi2 += r * r;
      }
      return chi2;
   }

   std::vector<double> Gradient(std::vector<double> const &p) const override
   {
      std::vector<double> grad(2, 0.0);
      for (std::size_t i = 0; i < fX.size(); ++i) {
         const double r = (fY[i] - Model(fX[i], p)) / fErr;
         const auto g = ModelGradient(fX[i], p);
         for (int j = 0; j < 2; ++j)
            grad[j] -= 2.0 * r * g[j] / fErr;
      }
      return grad;
   }

   std::vector<double> Hessian(std::vector<double> const &p) const override
   {
      std::vector<double> hess(4, 0.0);
      for (std::size_t i = 0; i < fX.size(); ++i) {
         const double r = (fY[i] - Model(fX[i], p)) / fErr;
         const auto g = ModelGradient(fX[i], p);
         const auto h = ModelHessian(fX[i], p);
         for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
               hess[j * 2 + k] -= 2.0 / fErr * (r * h[j * 2 + k] - g[j] * g[k] / fErr);
      }
      return hess;
   }

private:
   static double Model(double x, std::vector<double> const &p)
   {
      const double ka = p[0];
      const double kd = p[1];
      return x < 100 ? 1.0 - std::exp(x * (-ka - kd))
                     : (1.0 - std::exp(-100.0 * ka - 100.0 * kd)) * std::exp(-kd * (x - 100.0));
   }

   static std::vector<double> ModelGradient(double x, std::vector<double> const &p)
   {
      const double ka = p[0];
      const double kd = p[1];
      const double g0 = x < 100 ? x * std::exp(-x * (ka + kd)) : 100.0 * std::exp(-100.0 * ka - kd * x);
      const double g1 = x < 100 ? x * std::exp(-x * (ka + kd))
                                : ((1.0 - std::exp(100.0 * ka + 100.0 * kd)) * (x - 100.0) + 100.0) *
                                     std::exp(-100.0 * ka - kd * (x - 100.0) - 100.0 * kd);
      return {g0, g1};
   }

   static std::vector<double> ModelHessian(double x, std::vector<double> const &p)
   {
      const double ka = p[0];
      const double kd = p[1];
      const double h00 = x < 100 ? -x * x * std::exp(-x * (ka + kd)) : -10000.0 * std::exp(-100.0 * ka - kd * x);
      const double h01 = x < 100 ? -x * x * std::exp(-x * (ka + kd)) : -100.0 * x * std::exp(-100.0 * ka - kd * x);
      const double h11 =
         x < 100 ? -x * x * std::exp(-x * (ka + kd))
                 : (-200.0 * x - (1.0 - std::exp(100.0 * ka + 100.0 * kd)) * (x - 100.0) * (x - 100.0) + 10000.0) *
                      std::exp(-100.0 * ka - kd * (x - 100.0) - 100.0 * kd);
      return {h00, h01, h01, h11};
   }

   bool fHasHessian;

   static constexpr double fErr = 0.001;

   // clang-format off
   inline static const std::vector<double> fX = {
      0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
      160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300};

   // Generated with Model(x, {0.01, 0.01}) plus normal noise of width fErr
   inline static const std::vector<double> fY = {
      0.0014677825891482553, 0.18188013378785522, 0.32992558356212337, 0.45041553188395367,
      0.54794601921519193,   0.63195112024476019, 0.69853633449106323, 0.75286490803859663,
      0.7981616977365642,    0.83331949155031482, 0.86616863624896234, 0.78174922906099609,
      0.70818268046233734,   0.64271424905745012, 0.57898830590860717, 0.52425958636511283,
      0.47459845673511991,   0.43080972773016701, 0.38834996655153947, 0.35178861510658521,
      0.31925150166578198,   0.28732404660033928, 0.26252634588235924, 0.23540772239857896,
      0.21266927896262663,   0.19394038874520433, 0.17446775399099632, 0.15841619132882989,
      0.14478862098494658,   0.13111385920544755, 0.1156468064187659};
   // clang-format on
};

// When a parameter has limits, the internal<->external parameter transformation
// q -> x(q) is non-linear, so the Hessian and G2 (second derivatives) in
// internal coordinates carry an extra diagonal term beyond (dx/dq)^2 H_ext:
//
//   H_int(i,i) = (dx/dq)^2 H_ext(i,i) + (d^2x/dq^2) g_ext(i)
//
// The analytical-gradient path used to drop the (d^2x/dq^2) g_ext term. At a
// point with a non-zero gradient (i.e. away from the minimum) and a parameter
// close to its limit, this term dominates, which corrupted the Migrad seed and
// could make the minimization silently diverge to a wrong but "valid" result.
//
// This is checked deterministically (no dependence on the chaotic convergence
// behaviour) by comparing the analytical internal Hessian/G2 against a central
// finite difference of the calculator's own internal gradient.
//
// Covers GitHub issue https://github.com/root-project/root/issues/22692
TEST(Minuit2, AnalyticalHessianLimitTransformation)
{
   using namespace ROOT::Minuit2;

   BindingKineticsFCN fcn(/*hasHessian=*/true);

   MnUserParameters upar;
   // Start far from the solution and close to the lower limit: here the external
   // gradient is large and dx/dq is tiny, so the missing curvature term matters.
   upar.Add("ka", 1e-7, 1e-8);
   upar.Add("kd", 1e-7, 1e-8);
   upar.SetLowerLimit(0, 0.0);
   upar.SetLowerLimit(1, 0.0);

   const MnUserParameterState state(upar);
   const MnUserTransformation &trafo = state.Trafo();
   AnalyticalGradientCalculator agc(fcn, trafo);

   const unsigned int n = 2;

   MnAlgebraicVector q(n);
   for (unsigned int i = 0; i < n; ++i)
      q(i) = state.IntParameters()[i];

   // Internal gradient computed by the calculator at an internal point.
   auto internalGradient = [&](const MnAlgebraicVector &qq) { return agc(MinimumParameters(qq, 0.0)).Grad(); };

   // Analytical internal Hessian.
   MnAlgebraicSymMatrix hAnalytic(n);
   ASSERT_TRUE(agc.Hessian(MinimumParameters(q, 0.0), hAnalytic));

   // Analytical internal G2 (diagonal of the Hessian) via the dedicated method.
   MnAlgebraicVector g2Analytic(n);
   ASSERT_TRUE(agc.G2(MinimumParameters(q, 0.0), g2Analytic));

   // Ground truth: H_int(i,j) = d g_int(i) / d q_j by central finite difference.
   const double h = 1e-6;
   for (unsigned int j = 0; j < n; ++j) {
      MnAlgebraicVector qp(q);
      MnAlgebraicVector qm(q);
      qp(j) += h;
      qm(j) -= h;
      const MnAlgebraicVector gp = internalGradient(qp);
      const MnAlgebraicVector gm = internalGradient(qm);
      for (unsigned int i = 0; i < n; ++i) {
         const double fd = (gp(i) - gm(i)) / (2.0 * h);
         // Relative tolerance: the second derivatives span several orders of
         // magnitude, and dropping the curvature term changes them by ~100%.
         const double tol = 1e-3 * std::max(std::abs(fd), 1.0);
         EXPECT_NEAR(hAnalytic(i, j), fd, tol) << "Hessian mismatch at (" << i << "," << j << ")";
         if (i == j) {
            EXPECT_NEAR(g2Analytic(i), fd, tol) << "G2 mismatch at " << i;
         }
      }
   }
}
