// Tests for RooIntegralMorph
// Author: Jonas Rembser, CERN 08/2026

#include <RooAddPdf.h>
#include <RooArgSet.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooIntegralMorph.h>
#include <RooPolynomial.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooGlobalFunc.h>

#include <TMath.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

/// Kolmogorov-Smirnov distance between the pdf (normalized on [-xr, xr]) and
/// a true Gaussian with given mean and sigma, using a trapezoidal c.d.f. scan.
double ksDistanceToGaussian(RooAbsPdf &pdf, RooRealVar &x, double mu, double sigma, double xr, int npts = 40001)
{
   RooArgSet nset{x};
   const double h = 2 * xr / (npts - 1);

   auto trueCdf = [&](double xv) {
      const double z = std::sqrt(2.0) * sigma;
      return 0.5 * (std::erf((xv - mu) / z) - std::erf((-xr - mu) / z));
   };

   std::vector<double> vals(npts);
   for (int i = 0; i < npts; ++i) {
      x.setVal(-xr + i * h);
      vals[i] = pdf.getVal(nset);
   }

   double acc = 0.0;
   double ks = 0.0;
   const double tnorm = trueCdf(xr);
   for (int i = 1; i < npts; ++i) {
      acc += 0.5 * (vals[i - 1] + vals[i]) * h;
   }
   double cdf = 0.0;
   for (int i = 1; i < npts; ++i) {
      cdf += 0.5 * (vals[i - 1] + vals[i]) * h;
      ks = std::max(ks, std::abs(cdf / acc - trueCdf(-xr + i * h) / tnorm));
   }
   return ks;
}

/// Invert a monotone c.d.f. with safeguarded Newton iterations.
double quantile(double y, const std::function<double(double)> &cdf, const std::function<double(double)> &pdf,
                double xlo, double xhi)
{
   double a = xlo;
   double b = xhi;
   double x = 0.5 * (a + b);
   for (int i = 0; i < 200; ++i) {
      const double c = cdf(x) - y;
      (c > 0 ? b : a) = x;
      const double f = pdf(x);
      double xNew = f > 0 ? x - c / f : 0.5 * (a + b);
      if (!(xNew > a && xNew < b)) {
         xNew = 0.5 * (a + b);
      }
      if (std::abs(xNew - x) < 1e-15 * (xhi - xlo)) {
         return xNew;
      }
      x = xNew;
   }
   return x;
}

/// Kolmogorov-Smirnov distance between a pdf (normalized on [xlo, xhi]) and
/// the exact integral morph of two shapes given by their analytic normalized
/// c.d.f.s and p.d.f.s. The exact morphed c.d.f. is known parametrically: at
/// x(y) = alpha * x1(y) + (1 - alpha) * x2(y) it takes the value y, where
/// x1, x2 are the quantile functions of the input shapes.
double ksDistanceToTrueMorph(RooAbsPdf &pdf, RooRealVar &x, double alpha, const std::function<double(double)> &cdf1,
                             const std::function<double(double)> &pdf1, const std::function<double(double)> &cdf2,
                             const std::function<double(double)> &pdf2)
{
   RooArgSet nset{x};
   const double xlo = x.getMin();
   const double xhi = x.getMax();

   const int npts = 40001;
   const double yMin = 1e-9;
   const double yMax = 1. - 1e-9;

   // Sample the morph pdf at the exact morphed quantile positions and build
   // its c.d.f. with the trapezoidal rule on that grid.
   std::vector<double> xMorph(npts);
   std::vector<double> vals(npts);
   for (int i = 0; i < npts; ++i) {
      const double y = yMin + i * (yMax - yMin) / (npts - 1);
      const double x1 = quantile(y, cdf1, pdf1, xlo, xhi);
      const double x2 = quantile(y, cdf2, pdf2, xlo, xhi);
      xMorph[i] = alpha * x1 + (1 - alpha) * x2;
      x.setVal(xMorph[i]);
      vals[i] = pdf.getVal(nset);
   }

   double acc = 0.0;
   for (int i = 1; i < npts; ++i) {
      acc += 0.5 * (vals[i - 1] + vals[i]) * (xMorph[i] - xMorph[i - 1]);
   }
   double cdf = 0.0;
   double ks = 0.0;
   for (int i = 1; i < npts; ++i) {
      cdf += 0.5 * (vals[i - 1] + vals[i]) * (xMorph[i] - xMorph[i - 1]);
      const double yTrue = (yMin + i * (yMax - yMin) / (npts - 1) - yMin) / (yMax - yMin);
      ks = std::max(ks, std::abs(cdf / acc - yTrue));
   }
   return ks;
}

} // namespace

/// Morphing between two Gaussians with different means and widths must
/// reproduce a Gaussian with linearly-interpolated mean and width, since the
/// quantile function of a Gaussian is linear in (mu, sigma). This is the
/// N_{mu,sigma} benchmark from Baak et al., NIM A 771 (2015) 39-48, where the
/// original implementation was reported to be limited to KS distances of about
/// 1e-3 by implementation defects.
TEST(RooIntegralMorph, GaussianMorphAccuracy)
{
   const double xr = 10.0;

   for (double a : {0.25, 0.5, 0.75}) {
      RooRealVar x{"x", "x", -xr, xr};
      x.setBins(1000, "cache");
      RooRealVar alpha{"alpha", "alpha", a, 0.0, 1.0};
      RooGaussian g1{"g1", "", x, RooFit::RooConst(2.0), RooFit::RooConst(1.5)};
      RooGaussian g2{"g2", "", x, RooFit::RooConst(-2.0), RooFit::RooConst(0.5)};
      RooIntegralMorph morph{"morph", "", g1, g2, x, alpha};

      // True morphed shape: Gaussian with interpolated mean and width
      const double mu = a * 2.0 + (1 - a) * (-2.0);
      const double sigma = a * 1.5 + (1 - a) * 0.5;

      // With 1000 cache bins the achievable accuracy is limited by the
      // interpolation of the cache histogram at about 1e-7 ... 1e-6.
      EXPECT_LT(ksDistanceToGaussian(morph, x, mu, sigma, xr), 5e-6) << "alpha = " << a;
   }
}

namespace {

/// The endpoint shapes of the morphing test from the rf705 tutorial (and the
/// former stressRooFit test 705): a narrow Gaussian in the left half of the
/// range and a falling polynomial, on x in [-20, 20]. The exact morphed shape
/// has no closed form, but the analytic c.d.f.s of the endpoints are enough to
/// compute it to high precision (see ksDistanceToTrueMorph).
struct GaussPolySetup {
   double xr = 20.0;
   double mean = -10.0;
   double sigma = 2.0;
   double a1 = -0.03;
   double a2 = -0.001;

   double gaussCdfRaw(double xv) const { return 0.5 * (1.0 + std::erf((xv - mean) / (sigma * std::sqrt(2.0)))); }
   double gaussNorm() const { return gaussCdfRaw(xr) - gaussCdfRaw(-xr); }
   double polyPrim(double xv) const { return xv + 0.5 * a1 * xv * xv + a2 * xv * xv * xv / 3.0; }
   double polyNorm() const { return polyPrim(xr) - polyPrim(-xr); }

   std::function<double(double)> cdf1() const
   {
      return [*this](double xv) { return (gaussCdfRaw(xv) - gaussCdfRaw(-xr)) / gaussNorm(); };
   }
   std::function<double(double)> pdf1() const
   {
      return [*this](double xv) {
         const double z = (xv - mean) / sigma;
         return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2 * TMath::Pi())) / gaussNorm();
      };
   }
   std::function<double(double)> cdf2() const
   {
      return [*this](double xv) { return (polyPrim(xv) - polyPrim(-xr)) / polyNorm(); };
   }
   std::function<double(double)> pdf2() const
   {
      return [*this](double xv) { return (1.0 + a1 * xv + a2 * xv * xv) / polyNorm(); };
   }
};

} // namespace

/// Morphing between a Gaussian and a polynomial shape, checked against the
/// exact morphed shape computed from the analytic c.d.f.s of the endpoints.
/// Together with the following tests, this replaces the former stressRooFit
/// test 705, which compared against stored reference results and therefore
/// could not distinguish expected numerical changes from real regressions.
TEST(RooIntegralMorph, GaussPolyMorphAccuracy)
{
   GaussPolySetup s;

   for (double a : {0.125, 0.5, 0.875, 0.95}) {
      RooRealVar x{"x", "x", -s.xr, s.xr};
      x.setBins(1000, "cache");
      RooRealVar alpha{"alpha", "alpha", a, 0.0, 1.0};
      RooGaussian g1{"g1", "", x, RooFit::RooConst(s.mean), RooFit::RooConst(s.sigma)};
      RooPolynomial g2{"g2", "", x, {RooFit::RooConst(s.a1), RooFit::RooConst(s.a2)}};
      RooIntegralMorph morph{"morph", "", g1, g2, x, alpha};

      // The accuracy at 1000 cache bins is limited by the histogram
      // representation of the sharply falling tails to about 1e-4 (verified to
      // be identical to an ideal RooHistPdf built from the exact morph pdf
      // sampled at the same bin centers).
      const double ks = ksDistanceToTrueMorph(morph, x, a, s.cdf1(), s.pdf1(), s.cdf2(), s.pdf2());
      EXPECT_LT(ks, 3e-4) << "alpha = " << a;
   }
}

/// With setCacheAlpha(true), the pdf is cached in two dimensions (x, alpha)
/// and interpolated in alpha. Scanning the pdf over a grid in x and alpha must
/// agree with the exact morphed shape within the coarser alpha sampling. This
/// covers the two-dimensional pdf scan and the alpha-cache machinery of the
/// former stressRooFit test 705.
TEST(RooIntegralMorph, AlphaCacheScan)
{
   GaussPolySetup s;

   RooRealVar x{"x", "x", -s.xr, s.xr};
   x.setBins(1000, "cache");
   RooRealVar alpha{"alpha", "alpha", 0.5, 0.0, 1.0};
   alpha.setBins(50, "cache");
   RooGaussian g1{"g1", "", x, RooFit::RooConst(s.mean), RooFit::RooConst(s.sigma)};
   RooPolynomial g2{"g2", "", x, {RooFit::RooConst(s.a1), RooFit::RooConst(s.a2)}};
   RooIntegralMorph morph{"morph", "", g1, g2, x, alpha};
   morph.setCacheAlpha(true);

   for (double a : {0.125, 0.375, 0.625, 0.875}) {
      alpha.setVal(a);
      const double ks = ksDistanceToTrueMorph(morph, x, a, s.cdf1(), s.pdf1(), s.cdf2(), s.pdf2());
      // Measured about 1e-4 at this binning; well below the 1e-3 level of the
      // implementation defects fixed in the same commit that added this test.
      EXPECT_LT(ks, 5e-4) << "alpha = " << a;
   }
}

/// Generating a toy dataset from the morph pdf and fitting it back must
/// recover the true alpha, with the alpha cache enabled as in a realistic
/// fitting application. This covers the toy fit of the former stressRooFit
/// test 705.
TEST(RooIntegralMorph, GenerateAndFit)
{
   GaussPolySetup s;

   RooRealVar x{"x", "x", -s.xr, s.xr};
   x.setBins(1000, "cache");
   RooRealVar alpha{"alpha", "alpha", 0.8, 0.0, 1.0};
   alpha.setBins(50, "cache");
   RooGaussian g1{"g1", "", x, RooFit::RooConst(s.mean), RooFit::RooConst(s.sigma)};
   RooPolynomial g2{"g2", "", x, {RooFit::RooConst(s.a1), RooFit::RooConst(s.a2)}};
   RooIntegralMorph morph{"morph", "", g1, g2, x, alpha};

   RooRandom::randomGenerator()->SetSeed(12345);
   std::unique_ptr<RooDataSet> data{morph.generate(x, 1000)};

   morph.setCacheAlpha(true);

   for (std::string backend : {"cpu"}) {
      alpha.setVal(0.5);
      alpha.setError(0.0);
      std::unique_ptr<RooFitResult> res{
         morph.fitTo(*data, RooFit::EvalBackend(backend), RooFit::PrintLevel(-1), RooFit::Save())};
      EXPECT_EQ(res->status(), 0) << backend;
      EXPECT_NEAR(alpha.getVal(), 0.8, 5. * alpha.getError()) << backend;
      EXPECT_LT(alpha.getError(), 0.1) << backend;
      EXPECT_GT(alpha.getError(), 1e-4) << backend;
   }
}

/// Morphing from a bimodal shape must yield a smooth, normalized and
/// non-negative pdf (regions of near-zero density stress the c.d.f inversion).
TEST(RooIntegralMorph, BimodalEndpoint)
{
   const double xr = 10.0;
   RooRealVar x{"x", "x", -xr, xr};
   x.setBins(1000, "cache");
   RooRealVar alpha{"alpha", "alpha", 0.5, 0.0, 1.0};
   RooGaussian ga{"ga", "", x, RooFit::RooConst(-4.0), RooFit::RooConst(0.4)};
   RooGaussian gb{"gb", "", x, RooFit::RooConst(4.0), RooFit::RooConst(0.4)};
   RooAddPdf p1{"p1", "", {ga, gb}, {RooFit::RooConst(0.5)}};
   RooGaussian p2{"p2", "", x, RooFit::RooConst(0.0), RooFit::RooConst(2.0)};
   RooIntegralMorph morph{"morph", "", p1, p2, x, alpha};

   RooArgSet nset{x};
   const int npts = 10001;
   const double h = 2 * xr / (npts - 1);
   double integral = 0.0;
   double prev = 0.0;
   for (int i = 0; i < npts; ++i) {
      x.setVal(-xr + i * h);
      const double v = morph.getVal(nset);
      ASSERT_FALSE(std::isnan(v));
      ASSERT_GE(v, 0.0);
      if (i > 0) {
         integral += 0.5 * (prev + v) * h;
      }
      prev = v;
   }
   EXPECT_NEAR(integral, 1.0, 1e-6);
}
