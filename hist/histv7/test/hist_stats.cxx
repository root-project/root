#include "hist_test.hxx"

#include <cmath>
#include <stdexcept>

TEST(RHistStats, Constructor)
{
   RHistStats stats(1);
   EXPECT_EQ(stats.GetNDimensions(), 1);

   stats = RHistStats(2);
   EXPECT_EQ(stats.GetNDimensions(), 2);

   EXPECT_THROW(RHistStats(0), std::invalid_argument);
}

TEST(RHistStats, GetDimensionStats)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   {
      const auto &dimensionStats = stats.GetDimensionStats(/*=0*/);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, 190);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, 2470);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, 36100);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, 562666);
   }
   {
      const auto &dimensionStats = stats.GetDimensionStats(1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, 2 * 190);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, 4 * 2470);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, 8 * 36100);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, 16 * 562666);
   }
   {
      const auto &dimensionStats = stats.GetDimensionStats(2);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, 2470);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, 562666);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, 152455810);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, 44940730666);
   }
}

TEST(RHistStats, GetDimensionStatsWeighted)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   {
      const auto &dimensionStats = stats.GetDimensionStats(/*=0*/);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, 93.1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, 1330.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, 20489.98);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, 330265.6);
   }
   {
      const auto &dimensionStats = stats.GetDimensionStats(1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, 2 * 93.1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, 4 * 1330.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, 8 * 20489.98);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, 16 * 330265.6);
   }
   {
      const auto &dimensionStats = stats.GetDimensionStats(2);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, 1330.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, 330265.6);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, 93164182.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, 28108731464.8);
   }
}

TEST(RHistStats, Add)
{
   RHistStats statsA(2);
   RHistStats statsB(2);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      statsA.Fill(i, 2 * i);
      statsB.Fill(2 * i, 3 * i, RWeight(0.1 + 0.03 * i));
   }

   statsA.Add(statsB);

   ASSERT_EQ(statsA.GetNEntries(), 2 * Entries);
   EXPECT_DOUBLE_EQ(statsA.GetSumW(), 27.7);
   EXPECT_DOUBLE_EQ(statsA.GetSumW2(), 23.563);

   {
      const auto &dimensionStats = statsA.GetDimensionStats(/*=0*/);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX, 376.2);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX2, 7790);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX3, 200019.84);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX4, 5846915.6);
   }
   {
      const auto &dimensionStats = statsA.GetDimensionStats(1);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX, 659.3);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX2, 21850);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX3, 842029.46);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX4, 35754169.6);
   }
}

TEST(RHistStats, AddDifferent)
{
   RHistStats statsA(2);
   RHistStats statsB(3);

   EXPECT_THROW(statsA.Add(statsB), std::invalid_argument);
}

TEST(RHistStats, AddAtomic)
{
   RHistStats statsA(2);
   RHistStats statsB(2);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      statsA.Fill(i, 2 * i);
      statsB.Fill(2 * i, 3 * i, RWeight(0.1 + 0.03 * i));
   }

   statsA.AddAtomic(statsB);

   ASSERT_EQ(statsA.GetNEntries(), 2 * Entries);
   EXPECT_DOUBLE_EQ(statsA.GetSumW(), 27.7);
   EXPECT_DOUBLE_EQ(statsA.GetSumW2(), 23.563);

   {
      const auto &dimensionStats = statsA.GetDimensionStats(/*=0*/);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX, 376.2);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX2, 7790);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX3, 200019.84);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX4, 5846915.6);
   }
   {
      const auto &dimensionStats = statsA.GetDimensionStats(1);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX, 659.3);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX2, 21850);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX3, 842029.46);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX4, 35754169.6);
   }
}

TEST(RHistStats, StressAddAtomic)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NAddsPerThread = 10000;
   static constexpr std::size_t NAdds = NThreads * NAddsPerThread;
   static constexpr double X = 1.5;
   static constexpr double Weight = 0.5;

   // Use a single dimension, to maximize contention.
   RHistStats statsA(1);
   RHistStats statsB(1);
   statsB.Fill(X, RWeight(Weight));

   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NAddsPerThread; i++) {
         statsA.AddAtomic(statsB);
      }
   });

   EXPECT_EQ(statsA.GetNEntries(), NAdds);
   EXPECT_DOUBLE_EQ(statsA.GetSumW(), NAdds * Weight);
   EXPECT_DOUBLE_EQ(statsA.GetSumW2(), NAdds * Weight * Weight);

   const auto &dimensionStats = statsA.GetDimensionStats();
   EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, NAdds * Weight * X);
   EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, NAdds * Weight * X * X);
   EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, NAdds * Weight * X * X * X);
   EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, NAdds * Weight * X * X * X * X);
}

TEST(RHistStats, AddAtomicDifferent)
{
   RHistStats statsA(2);
   RHistStats statsB(3);

   EXPECT_THROW(statsA.AddAtomic(statsB), std::invalid_argument);
}

TEST(RHistStats, Clear)
{
   RHistStats stats(2);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i);
   }

   stats.Clear();

   EXPECT_EQ(stats.GetNEntries(), 0);
   EXPECT_DOUBLE_EQ(stats.GetSumW(), 0);
   EXPECT_DOUBLE_EQ(stats.GetSumW2(), 0);

   for (std::size_t dim = 0; dim < 2; dim++) {
      const auto &dimensionStats = stats.GetDimensionStats(dim);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX, 0);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX2, 0);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX3, 0);
      EXPECT_FLOAT_EQ(dimensionStats.fSumWX4, 0);
   }
}

TEST(RHistStats, ComputeNEffectiveEntries)
{
   RHistStats stats(1);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeNEffectiveEntries(), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(1);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.GetSumW(), Entries);
   EXPECT_DOUBLE_EQ(stats.GetSumW2(), Entries);
   EXPECT_DOUBLE_EQ(stats.ComputeNEffectiveEntries(), Entries);
}

TEST(RHistStats, ComputeNEffectiveEntriesWeighted)
{
   RHistStats stats(1);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeNEffectiveEntries(), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(1, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.GetSumW(), 7.7);
   EXPECT_DOUBLE_EQ(stats.GetSumW2(), 3.563);
   // Cross-checked with TH1
   EXPECT_DOUBLE_EQ(stats.ComputeNEffectiveEntries(), 16.640471512770137);
}

TEST(RHistStats, ComputeMean)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeMean(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeMean(1), 0);
   EXPECT_EQ(stats.ComputeMean(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.ComputeMean(/*=0*/), 9.5);
   EXPECT_DOUBLE_EQ(stats.ComputeMean(1), 19);
   EXPECT_DOUBLE_EQ(stats.ComputeMean(2), 123.5);
}

TEST(RHistStats, ComputeMeanWeighted)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeMean(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeMean(1), 0);
   EXPECT_EQ(stats.ComputeMean(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1
   EXPECT_DOUBLE_EQ(stats.ComputeMean(/*=0*/), 12.090909090909090);
   EXPECT_DOUBLE_EQ(stats.ComputeMean(1), 24.181818181818180);
   EXPECT_DOUBLE_EQ(stats.ComputeMean(2), 172.72727272727272);
}

TEST(RHistStats, ComputeVariance)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeVariance(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeVariance(1), 0);
   EXPECT_EQ(stats.ComputeVariance(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.ComputeVariance(/*=0*/), 33.25);
   EXPECT_DOUBLE_EQ(stats.ComputeVariance(1), 133);
   EXPECT_DOUBLE_EQ(stats.ComputeVariance(2), 12881.05);
}

TEST(RHistStats, ComputeVarianceWeighted)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeVariance(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeVariance(1), 0);
   EXPECT_EQ(stats.ComputeVariance(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1::GetStdDev squared, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeVariance(/*=0*/), 26.5371900);
   EXPECT_FLOAT_EQ(stats.ComputeVariance(1), 106.148760);
   EXPECT_FLOAT_EQ(stats.ComputeVariance(2), 13056.9256);
}

TEST(RHistStats, ComputeStdDev)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeStdDev(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeStdDev(1), 0);
   EXPECT_EQ(stats.ComputeStdDev(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.ComputeStdDev(/*=0*/), std::sqrt(33.25));
   EXPECT_DOUBLE_EQ(stats.ComputeStdDev(1), std::sqrt(133));
   EXPECT_DOUBLE_EQ(stats.ComputeStdDev(2), std::sqrt(12881.05));
}

TEST(RHistStats, ComputeStdDevWeighted)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeStdDev(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeStdDev(1), 0);
   EXPECT_EQ(stats.ComputeStdDev(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeStdDev(/*=0*/), 5.15142602);
   EXPECT_FLOAT_EQ(stats.ComputeStdDev(1), 10.3028520);
   EXPECT_FLOAT_EQ(stats.ComputeStdDev(2), 114.266905);
}

TEST(RHistStats, ComputeSkewness)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeSkewness(1), 0);
   EXPECT_EQ(stats.ComputeSkewness(2), 0);

   stats.Fill(0, 0, 0);
   ASSERT_EQ(stats.GetNEntries(), 1);
   // With one entry, the variance is 0 and we define skewness to be 0 as well.
   EXPECT_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeSkewness(1), 0);
   EXPECT_EQ(stats.ComputeSkewness(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 1; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_DOUBLE_EQ(stats.ComputeSkewness(1), 0);
   // Cross-checked with TH1 and SciPy, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeSkewness(2), 0.66125456);
}

TEST(RHistStats, ComputeSkewnessWeighted)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeSkewness(1), 0);
   EXPECT_EQ(stats.ComputeSkewness(2), 0);

   stats.Fill(0, 0, 0, RWeight(0.1));
   ASSERT_EQ(stats.GetNEntries(), 1);
   // With one entry, the variance is 0 and we define skewness to be 0 as well.
   EXPECT_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeSkewness(1), 0);
   EXPECT_EQ(stats.ComputeSkewness(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 1; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeSkewness(/*=0*/), -0.50554999);
   EXPECT_FLOAT_EQ(stats.ComputeSkewness(1), -0.50554999);
   EXPECT_FLOAT_EQ(stats.ComputeSkewness(2), 0.12072240);
}

TEST(RHistStats, ComputeKurtosis)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeKurtosis(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeKurtosis(1), 0);
   EXPECT_EQ(stats.ComputeKurtosis(2), 0);

   stats.Fill(0, 0, 0);
   ASSERT_EQ(stats.GetNEntries(), 1);
   // With one entry, the variance is 0 and we define kurtosis to be 0 as well.
   EXPECT_EQ(stats.ComputeKurtosis(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeKurtosis(1), 0);
   EXPECT_EQ(stats.ComputeKurtosis(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 1; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1 and SciPy, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(/*=0*/), -1.2060150);
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(1), -1.2060150);
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(2), -0.84198253);
}

TEST(RHistStats, ComputeKurtosisWeighted)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeKurtosis(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeKurtosis(1), 0);
   EXPECT_EQ(stats.ComputeKurtosis(2), 0);

   stats.Fill(0, 0, 0, RWeight(0.1));
   ASSERT_EQ(stats.GetNEntries(), 1);
   // With one entry, the variance is 0 and we define kurtosis to be 0 as well.
   EXPECT_EQ(stats.ComputeKurtosis(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeKurtosis(1), 0);
   EXPECT_EQ(stats.ComputeKurtosis(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 1; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(/*=0*/), -0.74828797);
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(1), -0.74828797);
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(2), -1.2483086);
}

TEST(RHistStats, ComputeSkewnessKurtosisVar0)
{
   RHistStats stats(1);
   stats.Fill(1, RWeight(0.1));
   ASSERT_EQ(stats.GetNEntries(), 1);
   EXPECT_EQ(stats.ComputeVariance(), 0);
   EXPECT_EQ(stats.ComputeSkewness(), 0);
   EXPECT_EQ(stats.ComputeKurtosis(), 0);
}

TEST(RHistStats, FillInvalidNumberOfArguments)
{
   RHistStats stats1(1);
   RHistStats stats2(2);

   EXPECT_NO_THROW(stats1.Fill(1));
   EXPECT_THROW(stats1.Fill(1, 2), std::invalid_argument);

   EXPECT_THROW(stats2.Fill(1), std::invalid_argument);
   EXPECT_NO_THROW(stats2.Fill(1, 2));
   EXPECT_THROW(stats2.Fill(1, 2, 3), std::invalid_argument);
}

TEST(RHistStats, FillWeightInvalidNumberOfArguments)
{
   RHistStats stats1(1);
   RHistStats stats2(2);

   EXPECT_NO_THROW(stats1.Fill(1, RWeight(1)));
   EXPECT_THROW(stats1.Fill(1, 2, RWeight(1)), std::invalid_argument);

   EXPECT_THROW(stats2.Fill(1, RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(stats2.Fill(1, 2, RWeight(1)));
   EXPECT_THROW(stats2.Fill(1, 2, 3, RWeight(1)), std::invalid_argument);
}

TEST(RHistStats, FillTupleWeightInvalidNumberOfArguments)
{
   RHistStats stats1(1);
   RHistStats stats2(2);

   EXPECT_NO_THROW(stats1.Fill(std::make_tuple(1), RWeight(1)));
   EXPECT_THROW(stats1.Fill(std::make_tuple(1, 2), RWeight(1)), std::invalid_argument);

   EXPECT_THROW(stats2.Fill(std::make_tuple(1), RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(stats2.Fill(std::make_tuple(1, 2), RWeight(1)));
   EXPECT_THROW(stats2.Fill(std::make_tuple(1, 2, 3), RWeight(1)), std::invalid_argument);
}

TEST(RHistStats, Scale)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i, RWeight(0.1 + 0.03 * i));
   }

   static constexpr double Factor = 0.8;
   stats.Scale(Factor);

   // The number of entries should not have changed.
   EXPECT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.GetSumW(), Factor * 7.7);
   EXPECT_DOUBLE_EQ(stats.GetSumW2(), Factor * Factor * 3.563);

   {
      const auto &dimensionStats = stats.GetDimensionStats(/*=0*/);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, Factor * 93.1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, Factor * 1330.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, Factor * 20489.98);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, Factor * 330265.6);
   }
   {
      const auto &dimensionStats = stats.GetDimensionStats(1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, Factor * 2 * 93.1);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, Factor * 4 * 1330.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, Factor * 8 * 20489.98);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, Factor * 16 * 330265.6);
   }
   {
      const auto &dimensionStats = stats.GetDimensionStats(2);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX, Factor * 1330.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX2, Factor * 330265.6);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX3, Factor * 93164182.0);
      EXPECT_DOUBLE_EQ(dimensionStats.fSumWX4, Factor * 28108731464.8);
   }
}
