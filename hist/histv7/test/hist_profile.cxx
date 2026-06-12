#include "hist_test.hxx"

#include <cstdint>
#include <iterator>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

static_assert(std::is_nothrow_move_constructible_v<RProfile>);
static_assert(std::is_nothrow_move_assignable_v<RProfile>);

TEST(RProfile, Constructor)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis regularAxis(Bins, {0, Bins});

   // The most generic constructor takes a vector of axis objects.
   const std::vector<RAxisVariant> axes = {regularAxis, regularAxis};
   RProfile profile(axes);
   EXPECT_EQ(profile.GetNDimensions(), 2);
   const auto &engine = profile.GetEngine();
   EXPECT_EQ(engine.GetNDimensions(), 2);
   const auto &stats = profile.GetStats();
   EXPECT_EQ(stats.GetNDimensions(), 3);
   EXPECT_EQ(profile.GetAxes().size(), 2);
   // Both axes include underflow and overflow bins.
   EXPECT_EQ(profile.GetTotalNBins(), (Bins + 2) * (Bins + 2));

   // Test other constructors, including move-assignment.
   profile = RProfile(Bins, {0, Bins});
   ASSERT_EQ(profile.GetNDimensions(), 1);
   auto *regular = profile.GetAxes()[0].GetRegularAxis();
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), Bins);
   EXPECT_EQ(regular->GetLow(), 0);
   EXPECT_EQ(regular->GetHigh(), Bins);
   // std::make_pair will take the types of the arguments, std::size_t in this case.
   profile = RProfile(Bins, std::make_pair(0, Bins));
   EXPECT_EQ(profile.GetNDimensions(), 1);

   // Brace-enclosed initializer list
   profile = RProfile({regularAxis});
   EXPECT_EQ(profile.GetNDimensions(), 1);
   profile = RProfile({regularAxis, regularAxis});
   EXPECT_EQ(profile.GetNDimensions(), 2);

   // Templated constructors
   profile = RProfile(regularAxis);
   EXPECT_EQ(profile.GetNDimensions(), 1);
   profile = RProfile(regularAxis, regularAxis);
   EXPECT_EQ(profile.GetNDimensions(), 2);
   profile = RProfile(regularAxis, regularAxis, regularAxis);
   EXPECT_EQ(profile.GetNDimensions(), 3);
}

TEST(RProfile, GetFullMultiDimRange)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   std::mt19937 gen;
   std::uniform_real_distribution<double> distX(0, Bins);
   std::uniform_real_distribution<double> distV(Bins, 2 * Bins);
   std::uniform_real_distribution<double> distW(0.8, 1.2);
   static constexpr std::uint64_t Entries = 1000;
   for (std::uint64_t i = 0; i < Entries; i++) {
      profile.Fill(distX(gen), distV(gen), RWeight(distW(gen)));
   }
   ASSERT_EQ(profile.GetNEntries(), Entries);

   auto range = profile.GetFullMultiDimRange();
   EXPECT_EQ(std::distance(range.begin(), range.end()), Bins + 2);

   double sumW = 0;
   double sumValues = 0;
   for (auto &&indices : range) {
      auto &bin = profile.GetBinContent(indices);
      sumW += bin.fSum;
      sumValues += bin.fSumValues;
   }
   // Numerical differences with EXPECT_DOUBLE_EQ
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW(), sumW);
   EXPECT_FLOAT_EQ(profile.GetStats().GetDimensionStats(1).fSumWX, sumValues);
}

TEST(RProfile, SetBinContent)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});

   RProfile::RProfileBin bin;
   bin.fSumValues = 42;

   {
      RProfile profile(axis);
      ASSERT_FALSE(profile.GetStats().IsTainted());
      profile.SetBinContent(RBinIndex(1), bin);
      EXPECT_EQ(profile.GetBinContent(RBinIndex(1)).fSumValues, 42);
      EXPECT_TRUE(profile.GetStats().IsTainted());
      EXPECT_THROW(profile.GetNEntries(), std::logic_error);
   }

   {
      RProfile profile(axis);
      ASSERT_FALSE(profile.GetStats().IsTainted());
      const std::array<RBinIndex, 1> indices = {2};
      profile.SetBinContent(indices, bin);
      EXPECT_EQ(profile.GetBinContent(indices).fSumValues, 42);
      EXPECT_TRUE(profile.GetStats().IsTainted());
      EXPECT_THROW(profile.GetNEntries(), std::logic_error);
   }
}

TEST(RProfile, Add)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RProfile profileA(axis);
   RProfile profileB(axis);

   profileA.Fill(8.5, 23.0);
   profileB.Fill(9.5, 25.0);

   profileA.Add(profileB);

   EXPECT_EQ(profileA.GetNEntries(), 2);
   EXPECT_EQ(profileA.GetBinContent(RBinIndex(8)).fSumValues, 23.0);
   EXPECT_EQ(profileA.GetBinContent(RBinIndex(9)).fSumValues, 25.0);
}

TEST(RProfile, AddAtomic)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RProfile profileA(axis);
   RProfile profileB(axis);

   profileA.Fill(8.5, 23.0);
   profileB.Fill(9.5, 25.0);

   profileA.AddAtomic(profileB);

   EXPECT_EQ(profileA.GetNEntries(), 2);
   EXPECT_EQ(profileA.GetBinContent(RBinIndex(8)).fSumValues, 23.0);
   EXPECT_EQ(profileA.GetBinContent(RBinIndex(9)).fSumValues, 25.0);
}

TEST(RProfile, StressAddAtomic)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NAddsPerThread = 10000;
   static constexpr std::size_t NAdds = NThreads * NAddsPerThread;

   // Fill a single bin, to maximize contention.
   const RRegularAxis axis(1, {0, 1});
   RProfile profileA(axis);
   RProfile profileB(axis);
   profileB.Fill(0.5, 23.0);

   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NAddsPerThread; i++) {
         profileA.AddAtomic(profileB);
      }
   });

   EXPECT_EQ(profileA.GetNEntries(), NAdds);
   EXPECT_EQ(profileA.GetBinContent(0).fSumValues, 23.0 * NAdds);
}

TEST(RProfile, AddExceptionSafety)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis regularAxis(Bins, {0, Bins});
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories);

   RProfile profileA({regularAxis, regularAxis});
   RProfile profileB({regularAxis, categoricalAxis});

   profileA.Fill(1.5, 2.5, 23.0);
   ASSERT_EQ(profileA.GetNEntries(), 1);
   ASSERT_EQ(profileA.GetBinContent(RBinIndex(1), RBinIndex(2)).fSumValues, 23.0);
   profileB.Fill(1.5, "b", 25.0);

   EXPECT_THROW(profileA.Add(profileB), std::invalid_argument);
   EXPECT_THROW(profileA.AddAtomic(profileB), std::invalid_argument);

   // Verify exception safety. Only the original entry should be there.
   EXPECT_EQ(profileA.GetNEntries(), 1);
   EXPECT_EQ(profileA.GetBinContent(RBinIndex(1), RBinIndex(2)).fSumValues, 23.0);
   EXPECT_EQ(profileA.GetStats().GetSumW(), 1);
   EXPECT_EQ(profileA.GetStats().GetSumW2(), 1);
   EXPECT_EQ(profileA.GetStats().GetDimensionStats(0).fSumWX, 1.5);
   EXPECT_EQ(profileA.GetStats().GetDimensionStats(1).fSumWX, 2.5);
}

TEST(RProfile, Clear)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   profile.Fill(8.5, 23.0);
   profile.Fill(9.5, 25.0);

   profile.Clear();

   EXPECT_EQ(profile.GetNEntries(), 0);
   EXPECT_EQ(profile.GetBinContent(RBinIndex(8)).fSumValues, 0);
   EXPECT_EQ(profile.GetBinContent(RBinIndex(9)).fSumValues, 0);
}

TEST(RProfile, Clone)
{
   static constexpr std::size_t Bins = 20;
   RProfile profileA(Bins, {0, Bins});

   profileA.Fill(8.5, 23.0);

   RProfile profileB = profileA.Clone();
   ASSERT_EQ(profileB.GetNDimensions(), 1);
   ASSERT_EQ(profileB.GetTotalNBins(), Bins + 2);

   EXPECT_EQ(profileB.GetNEntries(), 1);
   EXPECT_EQ(profileB.GetBinContent(8).fSumValues, 23.0);

   // Check that we can continue filling the clone.
   profileB.Fill(9.5, 25.0);

   EXPECT_EQ(profileA.GetNEntries(), 1);
   EXPECT_EQ(profileB.GetNEntries(), 2);
   EXPECT_EQ(profileA.GetBinContent(9).fSumValues, 0);
   EXPECT_EQ(profileB.GetBinContent(9).fSumValues, 25.0);
}

TEST(RProfile, Fill)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   profile.Fill(8.5, 23.0);
   profile.Fill(std::make_tuple(9.5), 25.0);

   auto &bin8 = profile.GetBinContent(RBinIndex(8));
   EXPECT_EQ(bin8.fSumValues, 23.0);
   EXPECT_EQ(bin8.fSumValues2, 529.0);
   EXPECT_EQ(bin8.fSum, 1.0);
   EXPECT_EQ(bin8.fSum2, 1.0);
   std::array<RBinIndex, 1> indices = {9};
   auto &bin9 = profile.GetBinContent(indices);
   EXPECT_EQ(bin9.fSumValues, 25.0);
   EXPECT_EQ(bin9.fSumValues2, 625.0);
   EXPECT_EQ(bin9.fSum, 1.0);
   EXPECT_EQ(bin9.fSum2, 1.0);

   EXPECT_EQ(profile.GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile.ComputeNEffectiveEntries(), 2);
   EXPECT_FLOAT_EQ(profile.ComputeMean(0), 9);
   EXPECT_FLOAT_EQ(profile.ComputeStdDev(0), 0.5);
   EXPECT_FLOAT_EQ(profile.ComputeMean(1), 24.0);
   EXPECT_FLOAT_EQ(profile.ComputeStdDev(1), 1.0);
}

TEST(RProfile, FillInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RProfile profile1(axis);
   ASSERT_EQ(profile1.GetNDimensions(), 1);
   RProfile profile2(axis, axis);
   ASSERT_EQ(profile2.GetNDimensions(), 2);

   EXPECT_NO_THROW(profile1.Fill(1, 2));
   EXPECT_THROW(profile1.Fill(1, 2, 3), std::invalid_argument);

   EXPECT_THROW(profile2.Fill(1, 2), std::invalid_argument);
   EXPECT_NO_THROW(profile2.Fill(1, 2, 3));
   EXPECT_THROW(profile2.Fill(1, 2, 3, 4), std::invalid_argument);
}

TEST(RProfile, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   profile.Fill(8.5, 23.0, RWeight(0.8));
   profile.Fill(std::make_tuple(9.5), 25.0, RWeight(0.9));

   auto &bin8 = profile.GetBinContent(RBinIndex(8));
   EXPECT_FLOAT_EQ(bin8.fSumValues, 18.4);
   EXPECT_FLOAT_EQ(bin8.fSumValues2, 423.2);
   EXPECT_FLOAT_EQ(bin8.fSum, 0.8);
   EXPECT_FLOAT_EQ(bin8.fSum2, 0.64);
   std::array<RBinIndex, 1> indices = {9};
   auto &bin9 = profile.GetBinContent(indices);
   EXPECT_FLOAT_EQ(bin9.fSumValues, 22.5);
   EXPECT_FLOAT_EQ(bin9.fSumValues2, 562.5);
   EXPECT_FLOAT_EQ(bin9.fSum, 0.9);
   EXPECT_FLOAT_EQ(bin9.fSum2, 0.81);

   EXPECT_EQ(profile.GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW(), 1.7);
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW2(), 1.45);
   // Cross-checked with TH1
   EXPECT_FLOAT_EQ(profile.ComputeNEffectiveEntries(), 1.9931034);
   EXPECT_FLOAT_EQ(profile.ComputeMean(0), 9.0294118);
   EXPECT_FLOAT_EQ(profile.ComputeStdDev(0), 0.49913420);
}

TEST(RProfile, FillWeightInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RProfile profile1(axis);
   ASSERT_EQ(profile1.GetNDimensions(), 1);
   RProfile profile2(axis, axis);
   ASSERT_EQ(profile2.GetNDimensions(), 2);

   EXPECT_THROW(profile1.Fill(1, RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(profile1.Fill(1, 2, RWeight(1)));
   EXPECT_THROW(profile1.Fill(1, 2, 3, RWeight(1)), std::invalid_argument);

   EXPECT_THROW(profile2.Fill(1, 2, RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(profile2.Fill(1, 2, 3, RWeight(1)));
   EXPECT_THROW(profile2.Fill(1, 2, 3, 4, RWeight(1)), std::invalid_argument);
}

TEST(RProfile, FillCategorical)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   RProfile profile({axis});

   profile.Fill("b", 23.0);
   profile.Fill(std::make_tuple("c"), 25.0);

   auto &bin1 = profile.GetBinContent(RBinIndex(1));
   EXPECT_EQ(bin1.fSumValues, 23.0);
   EXPECT_EQ(bin1.fSumValues2, 529.0);
   EXPECT_EQ(bin1.fSum, 1.0);
   EXPECT_EQ(bin1.fSum2, 1.0);
   std::array<RBinIndex, 1> indices = {2};
   auto &bin2 = profile.GetBinContent(indices);
   EXPECT_EQ(bin2.fSumValues, 25.0);
   EXPECT_EQ(bin2.fSumValues2, 625.0);
   EXPECT_EQ(bin2.fSum, 1.0);
   EXPECT_EQ(bin2.fSum2, 1.0);

   EXPECT_EQ(profile.GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile.ComputeNEffectiveEntries(), 2);
}

TEST(RProfile, FillCategoricalWeight)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   RProfile profile({axis});

   profile.Fill("b", 23.0, RWeight(0.8));
   profile.Fill(std::make_tuple("c"), 25.0, RWeight(0.9));

   auto &bin1 = profile.GetBinContent(RBinIndex(1));
   EXPECT_FLOAT_EQ(bin1.fSumValues, 18.4);
   EXPECT_FLOAT_EQ(bin1.fSumValues2, 423.2);
   EXPECT_FLOAT_EQ(bin1.fSum, 0.8);
   EXPECT_FLOAT_EQ(bin1.fSum2, 0.64);
   std::array<RBinIndex, 1> indices = {2};
   auto &bin2 = profile.GetBinContent(indices);
   EXPECT_FLOAT_EQ(bin2.fSumValues, 22.5);
   EXPECT_FLOAT_EQ(bin2.fSumValues2, 562.5);
   EXPECT_FLOAT_EQ(bin2.fSum, 0.9);
   EXPECT_FLOAT_EQ(bin2.fSum2, 0.81);

   EXPECT_EQ(profile.GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW(), 1.7);
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW2(), 1.45);
   // Cross-checked with TH1
   EXPECT_FLOAT_EQ(profile.ComputeNEffectiveEntries(), 1.9931034);
}

TEST(RProfile, FillExceptionSafety)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RProfile profile({axis, axis});

   profile.Fill(1.5, 2.5, 3.5);
   ASSERT_EQ(profile.GetNEntries(), 1);
   ASSERT_EQ(profile.GetBinContent(RBinIndex(1), RBinIndex(2)).fSumValues, 3.5);

   EXPECT_THROW(profile.Fill(1.5, "b", 3.5), std::invalid_argument);
   EXPECT_THROW(profile.Fill(std::make_tuple(1.5, "b"), 3.5), std::invalid_argument);
   EXPECT_THROW(profile.Fill(1.5, "b", 3.5, RWeight(1)), std::invalid_argument);
   EXPECT_THROW(profile.Fill(std::make_tuple(1.5, "b"), 3.5, RWeight(1)), std::invalid_argument);

   // Verify exception safety. Only the first entry should be there.
   EXPECT_EQ(profile.GetNEntries(), 1);
   EXPECT_EQ(profile.GetBinContent(RBinIndex(1), RBinIndex(2)).fSumValues, 3.5);
   EXPECT_EQ(profile.GetStats().GetSumW(), 1);
   EXPECT_EQ(profile.GetStats().GetSumW2(), 1);
   EXPECT_EQ(profile.GetStats().GetDimensionStats(0).fSumWX, 1.5);
   EXPECT_EQ(profile.GetStats().GetDimensionStats(1).fSumWX, 2.5);
   EXPECT_EQ(profile.GetStats().GetDimensionStats(2).fSumWX, 3.5);
}

TEST(RProfile, FillForward)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RProfile profile(axis, axis);
   CopyArgument value(23.0);

   std::tuple<CopyArgument, CopyArgument> args(1.5, 2.5);
   profile.Fill(args, value);
   profile.Fill(args, value, RWeight(0.5));
   EXPECT_EQ(profile.GetNEntries(), 2);
   EXPECT_EQ(profile.GetBinContent(1, 2).fSumValues, 34.5);

   ASSERT_FALSE(CopyArgument::HasBeenCopied());

   CopyArgument arg1(3.5), arg2(4.5);
   profile.Fill(arg1, arg2, value);
   profile.Fill(arg1, arg2, value, RWeight(0.5));
   EXPECT_EQ(profile.GetNEntries(), 4);
   EXPECT_EQ(profile.GetBinContent(3, 4).fSumValues, 34.5);

   ASSERT_FALSE(CopyArgument::HasBeenCopied());
}

TEST(RProfile, Scale)
{
   static constexpr std::size_t Bins = 20;
   RProfile profile(Bins, {0, Bins});

   profile.Fill(8.5, 23.0, RWeight(0.8));
   profile.Fill(9.5, 25.0, RWeight(0.9));

   static constexpr double Factor = 0.8;
   profile.Scale(Factor);

   EXPECT_FLOAT_EQ(profile.GetBinContent(8).fSumValues, Factor * 0.8 * 23.0);
   EXPECT_FLOAT_EQ(profile.GetBinContent(9).fSumValues, Factor * 0.9 * 25.0);

   EXPECT_EQ(profile.GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW(), Factor * 1.7);
   EXPECT_FLOAT_EQ(profile.GetStats().GetSumW2(), Factor * Factor * 1.45);
   // Cross-checked with TH1 - unchanged compared to FillWeight because the factor cancels out.
   EXPECT_FLOAT_EQ(profile.ComputeNEffectiveEntries(), 1.9931034);
   EXPECT_FLOAT_EQ(profile.ComputeMean(), 9.0294118);
   EXPECT_FLOAT_EQ(profile.ComputeStdDev(), 0.49913420);
}
