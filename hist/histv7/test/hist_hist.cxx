#include "hist_test.hxx"

#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

// Mostly, RHist = RHistEngine + RHistStats which are tested individually. Here we mostly check that the forwarding
// works correctly.

static_assert(std::is_nothrow_move_constructible_v<RHistEngine<int>>);
static_assert(std::is_nothrow_move_assignable_v<RHistEngine<int>>);

TEST(RHist, Constructor)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis regularAxis(Bins, {0, Bins});

   RHist<int> hist({regularAxis, regularAxis});
   EXPECT_EQ(hist.GetNDimensions(), 2);
   const auto &engine = hist.GetEngine();
   EXPECT_EQ(engine.GetNDimensions(), 2);
   const auto &stats = hist.GetStats();
   EXPECT_EQ(stats.GetNDimensions(), 2);
   EXPECT_EQ(hist.GetAxes().size(), 2);
   // Both axes include underflow and overflow bins.
   EXPECT_EQ(hist.GetTotalNBins(), (Bins + 2) * (Bins + 2));

   hist = RHist<int>(Bins, {0, Bins});
   ASSERT_EQ(hist.GetNDimensions(), 1);
   auto *regular = std::get_if<RRegularAxis>(&hist.GetAxes()[0]);
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), Bins);
   EXPECT_EQ(regular->GetLow(), 0);
   EXPECT_EQ(regular->GetHigh(), Bins);
}

TEST(RHist, Add)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> histA({axis});
   RHist<int> histB({axis});

   histA.Fill(8.5);
   histB.Fill(9.5);

   histA.Add(histB);

   EXPECT_EQ(histA.GetNEntries(), 2);
   EXPECT_EQ(histA.GetBinContent(RBinIndex(8)), 1);
   EXPECT_EQ(histA.GetBinContent(RBinIndex(9)), 1);
}

TEST(RHist, Clear)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist({axis});

   hist.Fill(8.5);
   hist.Fill(9.5);

   hist.Clear();

   EXPECT_EQ(hist.GetNEntries(), 0);
   EXPECT_EQ(hist.GetBinContent(RBinIndex(8)), 0);
   EXPECT_EQ(hist.GetBinContent(RBinIndex(9)), 0);
}

TEST(RHist, Clone)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> histA({axis});

   histA.Fill(8.5);

   RHist<int> histB = histA.Clone();
   ASSERT_EQ(histB.GetNDimensions(), 1);
   ASSERT_EQ(histB.GetTotalNBins(), Bins + 2);

   EXPECT_EQ(histB.GetNEntries(), 1);
   EXPECT_EQ(histB.GetBinContent(8), 1);

   // Check that we can continue filling the clone.
   histB.Fill(9.5);

   EXPECT_EQ(histA.GetNEntries(), 1);
   EXPECT_EQ(histB.GetNEntries(), 2);
   EXPECT_EQ(histA.GetBinContent(9), 0);
   EXPECT_EQ(histB.GetBinContent(9), 1);
}

TEST(RHist, Fill)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist({axis});

   hist.Fill(8.5);
   hist.Fill(std::make_tuple(9.5));

   EXPECT_EQ(hist.GetBinContent(RBinIndex(8)), 1);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_EQ(hist.GetBinContent(indices), 1);

   EXPECT_EQ(hist.GetStats().GetNEntries(), 2);
   EXPECT_FLOAT_EQ(hist.GetStats().ComputeNEffectiveEntries(), 2);
   EXPECT_FLOAT_EQ(hist.GetStats().ComputeMean(), 9);
   EXPECT_FLOAT_EQ(hist.GetStats().ComputeStdDev(), 0.5);
}

TEST(RHist, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<float> hist({axis});

   hist.Fill(8.5, RWeight(0.8));
   hist.Fill(std::make_tuple(9.5), RWeight(0.9));

   EXPECT_EQ(hist.GetStats().GetNEntries(), 2);
   // Cross-checked with TH1
   EXPECT_FLOAT_EQ(hist.GetStats().ComputeNEffectiveEntries(), 1.9931034);
   EXPECT_FLOAT_EQ(hist.GetStats().ComputeMean(), 9.0294118);
   EXPECT_FLOAT_EQ(hist.GetStats().ComputeStdDev(), 0.49913420);
}
