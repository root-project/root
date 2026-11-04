// Tests for the functions in RooFit::Detail::MathFuncs
// Authors: Jonas Rembser, CERN 11/2026

#include <RooFit/Detail/MathFuncs.h>

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
