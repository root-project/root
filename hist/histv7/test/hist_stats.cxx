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

TEST(RHistStats, ComputeSkewness)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeSkewness(1), 0);
   EXPECT_EQ(stats.ComputeSkewness(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   EXPECT_DOUBLE_EQ(stats.ComputeSkewness(/*=0*/), 0);
   EXPECT_DOUBLE_EQ(stats.ComputeSkewness(1), 0);
   // Cross-checked with TH1 and SciPy, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeSkewness(2), 0.66125456);
}

TEST(RHistStats, ComputeKurtosis)
{
   RHistStats stats(3);
   ASSERT_EQ(stats.GetNEntries(), 0);
   EXPECT_EQ(stats.ComputeKurtosis(/*=0*/), 0);
   EXPECT_EQ(stats.ComputeKurtosis(1), 0);
   EXPECT_EQ(stats.ComputeKurtosis(2), 0);

   static constexpr std::size_t Entries = 20;
   for (std::size_t i = 0; i < Entries; i++) {
      stats.Fill(i, 2 * i, i * i);
   }

   ASSERT_EQ(stats.GetNEntries(), Entries);
   // Cross-checked with TH1 and SciPy, numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(/*=0*/), -1.2060150);
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(1), -1.2060150);
   EXPECT_FLOAT_EQ(stats.ComputeKurtosis(2), -0.84198253);
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
