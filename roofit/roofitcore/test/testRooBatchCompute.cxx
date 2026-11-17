// Tests for the RooBatchCompute CPU library, in particular the optional
// multi-threaded evaluation of large batches.
// Authors: Jonas Rembser, CERN 2026

#include <RooBatchCompute.h>
#include <RooNaNPacker.h>

#include <ROOT/TestSupport.hxx>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

RooBatchCompute::Config makeConfig(int nThreads)
{
   RooBatchCompute::Config cfg;
   cfg.setNThreads(nThreads);
   return cfg;
}

// Large enough to engage the multi-threaded path, and deliberately not a
// multiple of the internal chunk size.
constexpr std::size_t nEvents = 200001;

std::vector<double> makeRandomVector(std::size_t n, double lo, double hi, unsigned int seed)
{
   std::mt19937 gen{seed};
   std::uniform_real_distribution<double> dist{lo, hi};
   std::vector<double> out(n);
   for (auto &val : out) {
      val = dist(gen);
   }
   return out;
}

} // namespace

class RooBatchComputeMT : public ::testing::Test {
protected:
   void SetUp() override
   {
      RooBatchCompute::initCPU();
      // The tests sweep over several thread counts, but ROOT's global thread
      // pool gets its size from the first parallel evaluation in the process
      // and warns about later requests for a different concurrency.
      _diags.optionalDiag(kWarning, "RTaskArenaWrapper", "There's already an active task arena",
                          /*matchFullMessage=*/false);
   }

   ROOT::TestSupport::CheckDiagsRAII _diags;
};

/// The parallel compute path must produce bitwise identical kernel outputs
/// for any number of threads, because the computation is element-wise.
TEST_F(RooBatchComputeMT, ComputeGaussian)
{
   std::vector<double> x = makeRandomVector(nEvents, -10., 10., 100);
   // Scalar parameters must be broadcast to bufferSize-wide arrays, like
   // RooFit::EvalContext does for the compute functions.
   std::vector<double> mean(RooBatchCompute::bufferSize, 1.0);
   std::vector<double> sigma(RooBatchCompute::bufferSize, 2.0);

   auto run = [&](int nThreads) {
      std::vector<double> output(nEvents);
      RooBatchCompute::compute(makeConfig(nThreads), RooBatchCompute::Gaussian, output, {x, mean, sigma});
      return output;
   };

   auto serial = run(1);
   for (int nThreads : {2, 4, 7}) {
      auto parallel = run(nThreads);
      EXPECT_EQ(serial, parallel) << "with " << nThreads << " threads";
   }
}

/// The NormalizedPdf computer uses the extra arguments to report evaluation
/// error counts. In the parallel path, each task works on a private copy of
/// the extra arguments and the deltas get merged, so the counts must come out
/// the same as in the serial evaluation.
TEST_F(RooBatchComputeMT, ComputeNormalizedPdfExtraArgsMerge)
{
   std::vector<double> rawVal = makeRandomVector(nEvents, 0., 10., 101);
   std::vector<double> normVal = makeRandomVector(nEvents, 0.5, 2., 102);

   // Sprinkle in evaluation errors: negative normalization terms (type 0
   // errors), spread over the full range so several chunks see some.
   for (std::size_t i = 0; i < nEvents; i += 1000) {
      normVal[i] = -1.0;
   }

   auto run = [&](int nThreads) {
      std::vector<double> output(nEvents);
      std::vector<double> extraArgs{0.0, 0.0, 0.0};
      RooBatchCompute::compute(makeConfig(nThreads), RooBatchCompute::NormalizedPdf, output, {rawVal, normVal},
                               extraArgs);
      return extraArgs;
   };

   auto serial = run(1);
   EXPECT_GT(serial[0], 0.0); // the test setup must actually produce errors
   for (int nThreads : {2, 4, 7}) {
      auto parallel = run(nThreads);
      EXPECT_EQ(serial, parallel) << "with " << nThreads << " threads";
   }
}

/// The parallel reductions combine fixed-size chunks in a fixed order, so the
/// result must be bitwise independent of the number of threads (though it may
/// differ from the serial result by rounding).
TEST_F(RooBatchComputeMT, ReduceSum)
{
   std::vector<double> input = makeRandomVector(nEvents, -1., 1., 103);

   const double serial = RooBatchCompute::reduceSum(makeConfig(1), input.data(), input.size());
   const double ref = RooBatchCompute::reduceSum(makeConfig(2), input.data(), input.size());
   for (int nThreads : {3, 4, 7}) {
      const double parallel = RooBatchCompute::reduceSum(makeConfig(nThreads), input.data(), input.size());
      EXPECT_EQ(ref, parallel) << "with " << nThreads << " threads";
   }
   EXPECT_NEAR(serial, ref, std::abs(serial) * 1e-14);
}

TEST_F(RooBatchComputeMT, ReduceNLL)
{
   std::vector<double> probas = makeRandomVector(nEvents, 0.01, 1., 104);
   std::vector<double> weights(nEvents, 1.0);
   // Some zero weights, which are skipped in the reduction.
   for (std::size_t i = 0; i < nEvents; i += 100) {
      weights[i] = 0.0;
   }

   auto run = [&](int nThreads) { return RooBatchCompute::reduceNLL(makeConfig(nThreads), probas, weights, {}); };

   auto serial = run(1);
   auto ref = run(2);
   for (int nThreads : {3, 4, 7}) {
      auto parallel = run(nThreads);
      EXPECT_EQ(ref.nllSum, parallel.nllSum) << "with " << nThreads << " threads";
      EXPECT_EQ(ref.nllSumCarry, parallel.nllSumCarry) << "with " << nThreads << " threads";
   }
   EXPECT_NEAR(serial.nllSum, ref.nllSum, std::abs(serial.nllSum) * 1e-14);

   // The counters must be exact in any mode.
   probas[123] = 0.0;
   probas[181818] = 0.0;
   for (int nThreads : {1, 4}) {
      auto out = run(nThreads);
      EXPECT_EQ(out.nNonPositiveValues, 2u) << "with " << nThreads << " threads";
   }
}

/// Reducing the logarithms of the probabilities gives the same result and the
/// same evaluation error counters as reducing the probabilities.
TEST_F(RooBatchComputeMT, ReduceNLLFromLogarithms)
{
   std::vector<double> probas = makeRandomVector(nEvents, 0.01, 1., 105);
   std::vector<double> weights = makeRandomVector(nEvents, 0.5, 2., 106);
   auto logOf = [](std::vector<double> const &values) {
      std::vector<double> out(values.size());
      std::transform(values.begin(), values.end(), out.begin(), [](double x) { return std::log(x); });
      return out;
   };

   for (int nThreads : {1, 4}) {
      auto ref = RooBatchCompute::reduceNLL(makeConfig(nThreads), probas, weights, {});
      auto out = RooBatchCompute::reduceNLL(makeConfig(nThreads), logOf(probas), weights, {}, true);
      EXPECT_NEAR(out.nllSum, ref.nllSum, 1e-12 * std::abs(ref.nllSum)) << "with " << nThreads << " threads";
   }

   probas[123] = 0.0;
   probas[181818] = std::numeric_limits<double>::quiet_NaN();
   for (int nThreads : {1, 4}) {
      auto out = RooBatchCompute::reduceNLL(makeConfig(nThreads), logOf(probas), weights, {}, true);
      EXPECT_EQ(out.nNonPositiveValues, 1u) << "with " << nThreads << " threads";
      EXPECT_EQ(out.nNaNValues, 1u) << "with " << nThreads << " threads";
   }
}

/// The LogProdPdf computer sums the factors given as logarithms and
/// multiplies the others, without underflow for many small factors.
TEST_F(RooBatchComputeMT, ComputeLogProdPdf)
{
   constexpr std::size_t n = 1000;
   constexpr std::size_t nLogFactors = 200;
   constexpr std::size_t nFactors = 400;

   std::vector<std::vector<double>> factors;
   std::vector<double> ref(n, 0.0);
   for (std::size_t k = 0; k < nFactors; ++k) {
      factors.push_back(makeRandomVector(n, 1e-3, 1e-2, 200 + k));
      for (std::size_t i = 0; i < n; ++i) {
         ref[i] += std::log(factors[k][i]);
         if (k < nLogFactors) {
            factors[k][i] = std::log(factors[k][i]);
         }
      }
   }
   // A zero and a negative factor.
   factors[nLogFactors][0] = 0.0;
   factors[nLogFactors][1] = -0.5;

   std::vector<std::span<const double>> vars(factors.begin(), factors.end());
   std::vector<double> output(n);
   std::array<double, 1> extraArgs{static_cast<double>(nLogFactors)};
   RooBatchCompute::compute(makeConfig(1), RooBatchCompute::LogProdPdf, output, vars, extraArgs);

   EXPECT_EQ(output[0], -std::numeric_limits<double>::infinity());
   EXPECT_TRUE(std::isnan(output[1]));
   for (std::size_t i = 2; i < n; ++i) {
      EXPECT_NEAR(output[i], ref[i], 1e-12 * std::abs(ref[i])) << "for event " << i;
   }
}

/// The logarithm of a product stays exact if a single factor is tiny, or if
/// the first factors multiply to a subnormal number or overflow and later
/// factors bring the product back into range. A NaN factor keeps its payload.
TEST_F(RooBatchComputeMT, ComputeLogProdPdfExtremeFactors)
{
   constexpr std::size_t nFactors = 207;
   std::vector<std::vector<double>> factors(nFactors, std::vector<double>(4, 1.0));
   for (std::size_t k = 0; k < 50; ++k) {
      factors[k][0] = 0x1p-5;
   }
   factors[50][0] = 1e-247;
   for (std::size_t k = 0; k < nFactors; ++k) {
      factors[k][1] = k < 107 ? 1e-3 : 1e3;
      factors[k][2] = k < 107 ? 1e3 : 1e-3;
   }
   factors[50][3] = RooNaNPacker::packFloatIntoNaN(3.f);
   const std::array<double, 3> ref{50 * std::log(0x1p-5) + std::log(1e-247), 107 * std::log(1e-3) + 100 * std::log(1e3),
                                   107 * std::log(1e3) + 100 * std::log(1e-3)};

   std::vector<std::span<const double>> vars(factors.begin(), factors.end());
   std::vector<double> output(4);
   std::array<double, 1> extraArgs{0.0};
   RooBatchCompute::compute(makeConfig(1), RooBatchCompute::LogProdPdf, output, vars, extraArgs);

   for (std::size_t i = 0; i < ref.size(); ++i) {
      EXPECT_NEAR(output[i], ref[i], 1e-12 * std::abs(ref[i])) << "for event " << i;
   }
   EXPECT_TRUE(std::isnan(output[3]));
// Like in testNaNPacker, payloads don't survive arithmetic in this configuration.
#if !defined(_MSC_VER) || defined(_WIN64) || defined(NDEBUG) || defined(R__ENABLE_BROKEN_WIN_TESTS)
   EXPECT_EQ(RooNaNPacker::unpackNaN(output[3]), 3.f);
#endif
}
