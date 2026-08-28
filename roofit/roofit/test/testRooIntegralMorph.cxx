// Tests for RooIntegralMorph
// Author: Jonas Rembser, CERN 08/2026

#include <RooAddPdf.h>
#include <RooArgSet.h>
#include <RooConstVar.h>
#include <RooGaussian.h>
#include <RooIntegralMorph.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

#include <cmath>
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
