// Tests for the RooBatchCompute CPU library, in particular the optional
// multi-threaded evaluation of large batches.
// Authors: Jonas Rembser, CERN 2026

#include <RooBatchCompute.h>

#include <gtest/gtest.h>

#include <cmath>
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
   void SetUp() override { RooBatchCompute::initCPU(); }
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
