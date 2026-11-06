// Tests for the RooFFTConvPdf
// Authors: Jonas Rembser, CERN 11/2026

#include <RooArgSet.h>
#include <RooFFTConvPdf.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooRealVar.h>

#include <TMath.h>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace {

struct GaussConvCase {
   double low;
   double high;
   int nBins;
   RooFFTConvPdf::BufStrat strategy;
   bool zeroShifts;
};

std::string gaussConvCaseName(testing::TestParamInfo<GaussConvCase> const &info)
{
   auto const &c = info.param;
   const char *strat = c.strategy == RooFFTConvPdf::Extend ? "Extend"
                       : c.strategy == RooFFTConvPdf::Flat ? "Flat"
                                                           : "Mirror";
   auto num = [](double val) {
      std::string out = std::to_string(static_cast<int>(std::lround(std::abs(val) * 10)));
      return (val < 0 ? "m" : "") + out;
   };
   return std::string{strat} + "_" + std::to_string(c.nBins) + "bins_" + num(c.low) + "_" + num(c.high) +
          (c.zeroShifts ? "_zeroShifts" : "");
}

} // namespace

class FFTConvGaussTest : public testing::TestWithParam<GaussConvCase> {};

// Compare the FFT convolution of two Gaussians with the analytic result, a
// Gaussian with added means and variances, for all buffer strategies, with and
// without the default shifts (see RooFFTConvPdf::setShift()), even and odd
// numbers of bins, ranges with and without zero, and a range not aligned with
// the bin width. Most of these came out shifted by several bins before the
// readout index of the convolution output was fixed, and even numbers of bins
// by half a bin before the sampling shift of the second pdf was corrected.
TEST_P(FFTConvGaussTest, CompareToAnalyticConvolution)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
   auto const &c = GetParam();

   RooRealVar x{"x", "x", c.low, c.high};
   x.setBins(c.nBins, "cache");

   // The first pdf sits off-center in the range, the second one is a
   // resolution function centered at zero.
   const double mean1 = 0.5 * (c.low + c.high) - 3.0;
   RooRealVar m1{"m1", "m1", mean1};
   RooRealVar s1{"s1", "s1", 1.0, 0.1, 10.0};
   RooRealVar m2{"m2", "m2", 0.0};
   RooRealVar s2{"s2", "s2", 2.0, 0.1, 10.0};
   RooGaussian g1{"g1", "g1", x, m1, s1};
   RooGaussian g2{"g2", "g2", x, m2, s2};

   RooFFTConvPdf conv{"conv", "conv", x, g1, g2};
   conv.setBufferStrategy(c.strategy);
   if (c.zeroShifts) {
      conv.setShift(0.0, 0.0);
   }

   const double sigma = std::sqrt(s1.getVal() * s1.getVal() + s2.getVal() * s2.getVal());
   const double norm = 0.5 * (std::erf((c.high - mean1) / (sigma * std::sqrt(2.0))) -
                              std::erf((c.low - mean1) / (sigma * std::sqrt(2.0))));
   auto ref = [&](double xVal) {
      return std::exp(-0.5 * std::pow((xVal - mean1) / sigma, 2)) / (sigma * std::sqrt(2.0 * TMath::Pi())) / norm;
   };

   // The convolution is exact at the bin centers. What remains is the
   // interpolation error of the cache in between (below 1e-7 here) and,
   // without the default shifts, the Flat and Mirror approximation of the
   // resolution function in the buffers (below 1e-5).
   const double tol = 2e-5;

   RooArgSet normSet{x};
   for (double xVal = mean1 - 4.0 * sigma; xVal <= mean1 + 4.0 * sigma; xVal += 0.25 * sigma) {
      x.setVal(xVal);
      EXPECT_NEAR(conv.getVal(normSet), ref(xVal), tol) << "at x = " << xVal;
   }
}

INSTANTIATE_TEST_SUITE_P(RooFFTConvPdf, FFTConvGaussTest,
                         testing::Values(
                            // range containing zero, lower boundary aligned with the bins
                            GaussConvCase{-10.0, 30.0, 1000, RooFFTConvPdf::Extend, false},
                            GaussConvCase{-10.0, 30.0, 1000, RooFFTConvPdf::Flat, false},
                            GaussConvCase{-10.0, 30.0, 1000, RooFFTConvPdf::Mirror, false},
                            GaussConvCase{-10.0, 30.0, 1001, RooFFTConvPdf::Extend, false},
                            GaussConvCase{-10.0, 30.0, 1001, RooFFTConvPdf::Flat, false},
                            GaussConvCase{-10.0, 30.0, 1001, RooFFTConvPdf::Mirror, false},
                            // without shifts, the resolution function is sampled where it
                            // is, which only works if it fits into the range
                            GaussConvCase{-10.0, 30.0, 1000, RooFFTConvPdf::Extend, true},
                            GaussConvCase{-10.0, 30.0, 1000, RooFFTConvPdf::Flat, true},
                            GaussConvCase{-10.0, 30.0, 1000, RooFFTConvPdf::Mirror, true},
                            GaussConvCase{-10.0, 30.0, 1001, RooFFTConvPdf::Extend, true},
                            // range not containing zero
                            GaussConvCase{2.0, 42.0, 1000, RooFFTConvPdf::Extend, false},
                            GaussConvCase{2.0, 42.0, 1000, RooFFTConvPdf::Flat, false},
                            GaussConvCase{2.0, 42.0, 1000, RooFFTConvPdf::Mirror, false},
                            GaussConvCase{2.0, 42.0, 1001, RooFFTConvPdf::Extend, false},
                            // range boundaries not aligned with multiples of the bin width
                            GaussConvCase{-3.7, 36.3, 1000, RooFFTConvPdf::Extend, false},
                            GaussConvCase{-3.7, 36.3, 1000, RooFFTConvPdf::Flat, false},
                            GaussConvCase{-3.7, 36.3, 1000, RooFFTConvPdf::Mirror, false},
                            GaussConvCase{-3.7, 36.3, 1001, RooFFTConvPdf::Extend, false}),
                         gaussConvCaseName);
