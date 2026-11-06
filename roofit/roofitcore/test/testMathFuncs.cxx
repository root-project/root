// Tests for the functions in RooFit::Detail::MathFuncs
// Authors: Jonas Rembser, CERN 11/2026

#include <RooFit/Detail/MathFuncs.h>

#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooRealVar.h>

#include <TRandom3.h>

#include <gtest/gtest.h>

#include <vector>

// Compare the FFT-based circular convolution with the direct sum, for
// power-of-two lengths and for lengths that need zero-padding.
TEST(MathFuncs, FFTConvolveAdd)
{
   TRandom3 rng{1337};
   for (unsigned int n : {1u, 2u, 3u, 7u, 8u, 16u, 100u, 1023u}) {
      std::vector<double> in1(n);
      std::vector<double> in2(n);
      for (unsigned int k = 0; k < n; ++k) {
         in1[k] = rng.Uniform(-1.0, 1.0);
         in2[k] = rng.Uniform(-1.0, 1.0);
      }
      // The function adds to the output
      std::vector<double> out(n, 1.0);
      RooFit::Detail::MathFuncs::fftConvolveAdd(n, in1.data(), in2.data(), out.data());

      for (unsigned int k = 0; k < n; ++k) {
         double ref = 1.0;
         for (unsigned int j = 0; j < n; ++j) {
            ref += in1[j] * in2[(k + n - j) % n];
         }
         EXPECT_NEAR(out[k], ref, 1e-12 * n) << "n = " << n << ", k = " << k;
      }
   }
}

// The interpolation must reproduce RooDataHist::weight(), including the
// mirroring at the boundaries.
TEST(MathFuncs, Interpolate1dPolynomial)
{
   const int nBins = 10;
   const double low = -2.0;
   const double high = 3.0;
   RooRealVar x{"x", "x", low, high};
   x.setBins(nBins);
   RooDataHist dh{"dh", "dh", x};

   TRandom3 rng{42};
   std::vector<double> vals(nBins);
   for (int i = 0; i < nBins; ++i) {
      vals[i] = rng.Uniform(0.5, 2.0);
      dh.set(i, vals[i], 0.0);
   }

   RooArgSet coords{x};
   for (int intOrder : {0, 1, 2, 3, 4}) {
      for (double val = low; val < high; val += 0.0137) {
         x.setVal(val);
         const double ref = dh.weight(coords, intOrder, /*correctForBinSize=*/false, /*cdfBoundaries=*/false);
         const double res =
            RooFit::Detail::MathFuncs::interpolate1dPolynomial(low, high, val, nBins, vals.data(), intOrder);
         EXPECT_NEAR(res, ref, 1e-12) << "intOrder = " << intOrder << ", x = " << val;
      }
   }
}
