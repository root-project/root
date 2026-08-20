#include "histutil_test.hxx"

#include <TH3.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>

TEST(ConvertToTH3I, RHistEngine)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis, axis, axis);

   engine.SetBinContent(RBinIndex::Underflow(), 0, 0, 1000);
   engine.SetBinContent(0, RBinIndex::Overflow(), 0, 2000);
   engine.SetBinContent(0, 0, RBinIndex::Underflow(), 3000);
   for (std::size_t i = 0; i < Bins; i++) {
      for (std::size_t j = 0; j < Bins; j++) {
         for (std::size_t k = 0; k < Bins; k++) {
            engine.SetBinContent(i, j, k, 100 * i + 10 * j + k);
         }
      }
   }
   engine.SetBinContent(RBinIndex::Overflow(), 1, 2, 4000);

   auto th3i = ConvertToTH3I(engine);
   ASSERT_TRUE(th3i);
   EXPECT_TRUE(th3i->GetDirectory() == nullptr);
   ASSERT_EQ(th3i->GetDimension(), 3);
   ASSERT_EQ(th3i->GetNbinsX(), Bins);
   ASSERT_EQ(th3i->GetNbinsY(), Bins);
   ASSERT_EQ(th3i->GetNbinsZ(), Bins);

   EXPECT_EQ(th3i->GetBinContent(0, 1, 1), 1000);
   EXPECT_EQ(th3i->GetBinContent(1, Bins + 1, 1), 2000);
   EXPECT_EQ(th3i->GetBinContent(1, 1, 0), 3000);
   for (std::size_t i = 0; i < Bins; i++) {
      for (std::size_t j = 0; j < Bins; j++) {
         for (std::size_t k = 0; k < Bins; k++) {
            EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, k + 1), 100 * i + 10 * j + k);
         }
      }
   }
   EXPECT_EQ(th3i->GetBinContent(Bins + 1, 2, 3), 4000);

   EXPECT_EQ(th3i->GetEntries(), 0);
   Double_t stats[11];
   th3i->GetStats(stats);
   for (double stat : stats) {
      EXPECT_EQ(stat, 0);
   }
}

TEST(ConvertToTH3I, RHistEngineNoFlowBins)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins}, /*enableFlowBins=*/false);
   RHistEngine<int> engine(axis, axis, axis);

   engine.Fill(-100, 0.5, 0.5);
   engine.Fill(0.5, -100, 0.5);
   engine.Fill(0.5, 0.5, -100);
   for (std::size_t i = 0; i < Bins; i++) {
      for (std::size_t j = 0; j < Bins; j++) {
         for (std::size_t k = 0; k < Bins; k++) {
            engine.SetBinContent(i, j, k, 100 * i + 10 * j + k);
         }
      }
   }

   auto th3i = ConvertToTH3I(engine);
   ASSERT_TRUE(th3i);
   EXPECT_EQ(th3i->GetBinContent(0, 1, 1), 0);
   EXPECT_EQ(th3i->GetBinContent(1, 0, 1), 0);
   EXPECT_EQ(th3i->GetBinContent(1, 1, 0), 0);
   for (std::size_t i = 0; i < Bins; i++) {
      for (std::size_t j = 0; j < Bins; j++) {
         for (std::size_t k = 0; k < Bins; k++) {
            EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, k + 1), 100 * i + 10 * j + k);
         }
      }
   }
}

TEST(ConvertToTH3I, RHistEngineInvalid)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine(axis, axis);

   EXPECT_THROW(ConvertToTH3I(engine), std::invalid_argument);
}

TEST(ConvertToTH3I, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist(axis, axis, axis);

   for (std::size_t i = 0; i < Bins; i++) {
      hist.Fill(i, 2 * i, 3 * i);
   }

   auto th3i = ConvertToTH3I(hist);
   ASSERT_TRUE(th3i);

   ASSERT_EQ(hist.GetNEntries(), Bins);
   EXPECT_EQ(th3i->GetEntries(), Bins);
   Double_t stats[11];
   th3i->GetStats(stats);
   EXPECT_EQ(stats[0], Bins);
   EXPECT_EQ(stats[1], Bins);
   EXPECT_EQ(stats[2], 190);
   EXPECT_EQ(stats[3], 2470);
   EXPECT_EQ(stats[4], 2 * 190);
   EXPECT_EQ(stats[5], 4 * 2470);
   EXPECT_EQ(stats[6], 0);
   EXPECT_EQ(stats[7], 3 * 190);
   EXPECT_EQ(stats[8], 9 * 2470);
   EXPECT_EQ(stats[9], 0);
   EXPECT_EQ(stats[10], 0);
}

TEST(ConvertToTH3I, RHistSetBinContentTainted)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist(axis, axis, axis);
   const std::array<RBinIndex, 3> indices = {1, 2, 3};
   hist.SetBinContent(indices, 42);
   ASSERT_TRUE(hist.GetStats().IsTainted());

   auto th3i = ConvertToTH3I(hist);
   ASSERT_TRUE(th3i);

   EXPECT_EQ(th3i->GetBinContent(2, 3, 4), 42);
   EXPECT_EQ(th3i->GetEntries(), 0);
   Double_t stats[11];
   th3i->GetStats(stats);
   for (double stat : stats) {
      EXPECT_EQ(stat, 0);
   }
}

TEST(ConvertToTH3I, RHistCategoricalAxis)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   RHist<int> hist(axis, axis, axis);
   ASSERT_FALSE(hist.GetStats().IsEnabled(0));
   ASSERT_FALSE(hist.GetStats().IsEnabled(1));
   ASSERT_FALSE(hist.GetStats().IsEnabled(2));

   hist.Fill("a", "b", "c");

   auto th3i = ConvertToTH3I(hist);
   ASSERT_TRUE(th3i);
   EXPECT_EQ(th3i->GetBinContent(1, 2, 3), 1);
   EXPECT_EQ(th3i->GetEntries(), 1);
   Double_t stats[11];
   th3i->GetStats(stats);
   EXPECT_EQ(stats[0], 1);
   EXPECT_EQ(stats[1], 1);
   for (std::size_t i = 2; i < 11; i++) {
      EXPECT_EQ(stats[i], 0);
   }
}

TEST(ConvertToTH3D, RHistEngine)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engine(axis, axis, axis);

   engine.Fill(0, 1, 2, RWeight(0.5));

   auto th3d = ConvertToTH3D(engine);
   ASSERT_TRUE(th3d);
   const Double_t *sumw2 = th3d->GetSumw2()->GetArray();
   ASSERT_TRUE(sumw2 != nullptr);
   EXPECT_DOUBLE_EQ(th3d->GetBinContent(1, 2, 3), 0.5);
   EXPECT_DOUBLE_EQ(sumw2[th3d->GetBin(1, 2, 3)], 0.25);
}

TEST(ConvertToTH3C, RHistEngine)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<char> engine(axis, axis, axis);
   EXPECT_TRUE(ConvertToTH3C(engine));
}

TEST(ConvertToTH3S, RHistEngine)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<short> engine(axis, axis, axis);
   EXPECT_TRUE(ConvertToTH3S(engine));
}

TEST(ConvertToTH3L, RHistEngine)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<long> engine(axis, axis, axis);
   EXPECT_TRUE(ConvertToTH3L(engine));
}

TEST(ConvertToTH3F, RHistEngine)
{
   static constexpr std::size_t Bins = 4;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<float> engine(axis, axis, axis);
   EXPECT_TRUE(ConvertToTH3F(engine));
}