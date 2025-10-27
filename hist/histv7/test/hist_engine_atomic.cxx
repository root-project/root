#include "hist_test.hxx"

TEST(RHistEngine, FillAtomic)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine({axis});

   engine.FillAtomic(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.FillAtomic(i);
   }
   engine.FillAtomic(100);

   EXPECT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 1);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engine.GetBinContent(index), 1);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 1);

   // Instantiate further bin content types to make sure they work.
   RHistEngine<long> engineL({axis});
   engineL.FillAtomic(1);

   RHistEngine<long long> engineLL({axis});
   engineLL.FillAtomic(1);

   RHistEngine<float> engineF({axis});
   engineF.FillAtomic(1);

   RHistEngine<double> engineD({axis});
   engineD.FillAtomic(1);
}

TEST(RHistEngine, StressFillAtomic)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;

   // Fill a single bin, to maximize contention.
   RHistEngine<int> engine(1, {0, 1});
   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NFillsPerThread; i++) {
         engine.FillAtomic(0.5);
      }
   });

   EXPECT_EQ(engine.GetBinContent(0), NFills);
}

TEST(RHistEngine, FillAtomicTuple)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine({axis});

   engine.FillAtomic(std::make_tuple(-100));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.FillAtomic(std::make_tuple(i));
   }
   engine.FillAtomic(std::make_tuple(100));

   EXPECT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 1);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engine.GetBinContent(index), 1);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 1);
}

TEST(RHistEngine, FillAtomicInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   RHistEngine<int> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.FillAtomic(1));
   EXPECT_THROW(engine1.FillAtomic(1, 2), std::invalid_argument);

   EXPECT_THROW(engine2.FillAtomic(1), std::invalid_argument);
   EXPECT_NO_THROW(engine2.FillAtomic(1, 2));
   EXPECT_THROW(engine2.FillAtomic(1, 2, 3), std::invalid_argument);
}

TEST(RHistEngine, FillAtomicWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine({axis});

   engine.FillAtomic(-100, RWeight(0.25));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.FillAtomic(i, RWeight(0.1 + i * 0.03));
   }
   engine.FillAtomic(100, RWeight(0.75));

   EXPECT_FLOAT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 0.25);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_FLOAT_EQ(engine.GetBinContent(index), 0.1 + index.GetIndex() * 0.03);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 0.75);

   // Instantiate further bin content types to make sure they work.
   RHistEngine<double> engineD({axis});
   engineD.FillAtomic(1, RWeight(0.8));
}

TEST(RHistEngine, StressFillAtomicWeight)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;
   static constexpr double Weight = 0.5;

   // Fill a single bin, to maximize contention.
   RHistEngine<float> engine(1, {0, 1});
   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NFillsPerThread; i++) {
         engine.FillAtomic(0.5, RWeight(Weight));
      }
   });

   EXPECT_EQ(engine.GetBinContent(0), NFills * Weight);
}

TEST(RHistEngine, FillAtomicTupleWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine({axis});

   engine.FillAtomic(std::make_tuple(-100), RWeight(0.25));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.FillAtomic(std::make_tuple(i), RWeight(0.1 + i * 0.03));
   }
   engine.FillAtomic(std::make_tuple(100), RWeight(0.75));

   EXPECT_FLOAT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 0.25);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_FLOAT_EQ(engine.GetBinContent(index), 0.1 + index.GetIndex() * 0.03);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 0.75);
}

TEST(RHistEngine, FillAtomicWeightInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   RHistEngine<float> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.FillAtomic(1, RWeight(1)));
   EXPECT_THROW(engine1.FillAtomic(1, 2, RWeight(1)), std::invalid_argument);

   EXPECT_THROW(engine2.FillAtomic(1, RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(engine2.FillAtomic(1, 2, RWeight(1)));
   EXPECT_THROW(engine2.FillAtomic(1, 2, 3, RWeight(1)), std::invalid_argument);
}

TEST(RHistEngine, FillAtomicTupleWeightInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   RHistEngine<float> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.FillAtomic(std::make_tuple(1), RWeight(1)));
   EXPECT_THROW(engine1.FillAtomic(std::make_tuple(1, 2), RWeight(1)), std::invalid_argument);

   EXPECT_THROW(engine2.FillAtomic(std::make_tuple(1), RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(engine2.FillAtomic(std::make_tuple(1, 2), RWeight(1)));
   EXPECT_THROW(engine2.FillAtomic(std::make_tuple(1, 2, 3), RWeight(1)), std::invalid_argument);
}

TEST(RHistEngine_RBinWithError, FillAtomic)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engine({axis});

   for (std::size_t i = 0; i < Bins; i++) {
      engine.FillAtomic(i);
   }

   for (auto index : axis.GetNormalRange()) {
      auto &bin = engine.GetBinContent(index);
      EXPECT_EQ(bin.fSum, 1);
      EXPECT_EQ(bin.fSum2, 1);
   }
}

TEST(RHistEngine_RBinWithError, StressFillAtomic)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;

   // Fill a single bin, to maximize contention.
   RHistEngine<RBinWithError> engine(1, {0, 1});
   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NFillsPerThread; i++) {
         engine.FillAtomic(0.5);
      }
   });

   EXPECT_EQ(engine.GetBinContent(0).fSum, NFills);
   EXPECT_EQ(engine.GetBinContent(0).fSum2, NFills);
}

TEST(RHistEngine_RBinWithError, FillAtomicWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engine({axis});

   for (std::size_t i = 0; i < Bins; i++) {
      engine.FillAtomic(i, RWeight(0.1 + i * 0.03));
   }

   for (auto index : axis.GetNormalRange()) {
      auto &bin = engine.GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST(RHistEngine_RBinWithError, StressFillAtomicWeight)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;
   static constexpr double Weight = 0.5;

   // Fill a single bin, to maximize contention.
   RHistEngine<RBinWithError> engine(1, {0, 1});
   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NFillsPerThread; i++) {
         engine.FillAtomic(0.5, RWeight(Weight));
      }
   });

   EXPECT_EQ(engine.GetBinContent(0).fSum, NFills * Weight);
   EXPECT_EQ(engine.GetBinContent(0).fSum2, NFills * Weight * Weight);
}
