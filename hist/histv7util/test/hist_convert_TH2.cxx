#include "histutil_test.hxx"

#include <TH2.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>

TEST(ConvertToTH2I, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis, axis);

   engine.SetBinContent(RBinIndex::Underflow(), 0, 1000);
   engine.SetBinContent(RBinIndex::Underflow(), 2, 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.SetBinContent(i, RBinIndex::Underflow(), 100 * i);
      for (std::size_t j = 0; j < Bins; j++) {
         engine.SetBinContent(i, j, i * Bins + j);
      }
      engine.SetBinContent(i, RBinIndex::Overflow(), 200 * i);
   }
   engine.SetBinContent(RBinIndex::Overflow(), 3, 3000);
   engine.SetBinContent(RBinIndex::Overflow(), 6, 4000);

   auto th2i = ConvertToTH2I(engine);
   ASSERT_TRUE(th2i);
   EXPECT_TRUE(th2i->GetDirectory() == nullptr);
   ASSERT_EQ(th2i->GetDimension(), 2);
   ASSERT_EQ(th2i->GetNbinsX(), Bins);
   ASSERT_EQ(th2i->GetNbinsY(), Bins);
   ASSERT_EQ(th2i->GetNbinsZ(), 1);

   EXPECT_EQ(th2i->GetBinContent(0, 1), 1000);
   EXPECT_EQ(th2i->GetBinContent(0, 3), 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      EXPECT_EQ(th2i->GetBinContent(i + 1, 0), 100 * i);
      for (std::size_t j = 0; j < Bins; j++) {
         EXPECT_EQ(th2i->GetBinContent(i + 1, j + 1), i * Bins + j);
      }
      EXPECT_EQ(th2i->GetBinContent(i + 1, Bins + 1), 200 * i);
   }
   EXPECT_EQ(th2i->GetBinContent(Bins + 1, 4), 3000);
   EXPECT_EQ(th2i->GetBinContent(Bins + 1, 7), 4000);

   EXPECT_EQ(th2i->GetEntries(), 0);
   Double_t stats[7];
   th2i->GetStats(stats);
   EXPECT_EQ(stats[0], 0);
   EXPECT_EQ(stats[1], 0);
   EXPECT_EQ(stats[2], 0);
   EXPECT_EQ(stats[3], 0);
   EXPECT_EQ(stats[4], 0);
   EXPECT_EQ(stats[5], 0);
   EXPECT_EQ(stats[6], 0);
}

TEST(ConvertToTH2I, RHistEngineNoFlowBins)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins}, /*enableFlowBins=*/false);
   RHistEngine<int> engine(axis, axis);

   // Flow bins are disabled, so these fills will be discarded.
   engine.Fill(-100, 0.5);
   engine.Fill(-100, 2.5);
   for (std::size_t i = 0; i < Bins; i++) {
      // Flow bins are disabled, so this fill will be discarded.
      engine.Fill(i + 0.5, -100);
      for (std::size_t j = 0; j < Bins; j++) {
         engine.SetBinContent(i, j, i * Bins + j);
      }
      // Flow bins are disabled, so this fill will be discarded.
      engine.Fill(i + 0.5, 100);
   }
   // Flow bins are disabled, so these fills will be discarded.
   engine.Fill(100, 3.5);
   engine.Fill(100, 6.5);

   auto th2i = ConvertToTH2I(engine);
   ASSERT_TRUE(th2i);

   EXPECT_EQ(th2i->GetBinContent(0, 1), 0);
   EXPECT_EQ(th2i->GetBinContent(0, 3), 0);
   for (std::size_t i = 0; i < Bins; i++) {
      EXPECT_EQ(th2i->GetBinContent(i + 1, 0), 0);
      for (std::size_t j = 0; j < Bins; j++) {
         EXPECT_EQ(th2i->GetBinContent(i + 1, j + 1), i * Bins + j);
      }
      EXPECT_EQ(th2i->GetBinContent(i + 1, Bins + 1), 0);
   }
   EXPECT_EQ(th2i->GetBinContent(Bins + 1, 4), 0);
   EXPECT_EQ(th2i->GetBinContent(Bins + 1, 7), 0);
}

TEST(ConvertToTH2I, RHistEngineInvalid)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine(axis);

   EXPECT_THROW(ConvertToTH2I(engine), std::invalid_argument);
}

TEST(ConvertToTH2I, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist(axis, axis);

   for (std::size_t i = 0; i < Bins; i++) {
      hist.Fill(i, 2 * i);
   }

   auto th2i = ConvertToTH2I(hist);
   ASSERT_TRUE(th2i);

   ASSERT_EQ(hist.GetNEntries(), Bins);
   EXPECT_EQ(th2i->GetEntries(), Bins);
   Double_t stats[7];
   th2i->GetStats(stats);
   EXPECT_EQ(stats[0], Bins);
   EXPECT_EQ(stats[1], Bins);
   EXPECT_EQ(stats[2], 190);
   EXPECT_EQ(stats[3], 2470);
   EXPECT_EQ(stats[4], 2 * 190);
   EXPECT_EQ(stats[5], 4 * 2470);
   EXPECT_EQ(stats[6], 0);
}

TEST(ConvertToTH2I, RHistSetBinContentTainted)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist(axis, axis);
   const std::array<RBinIndex, 2> indices = {1, 2};
   hist.SetBinContent(indices, 42);
   ASSERT_TRUE(hist.GetStats().IsTainted());

   auto th2i = ConvertToTH2I(hist);
   ASSERT_TRUE(th2i);

   EXPECT_EQ(th2i->GetBinContent(2 + (Bins + 2) * 3), 42);

   EXPECT_EQ(th2i->GetEntries(), 0);
   Double_t stats[7];
   th2i->GetStats(stats);
   EXPECT_EQ(stats[0], 0);
   EXPECT_EQ(stats[1], 0);
   EXPECT_EQ(stats[2], 0);
   EXPECT_EQ(stats[3], 0);
   EXPECT_EQ(stats[4], 0);
   EXPECT_EQ(stats[5], 0);
   EXPECT_EQ(stats[6], 0);
}

TEST(ConvertToTH2I, RHistCategoricalAxis)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   RHist<int> hist(axis, axis);
   ASSERT_FALSE(hist.GetStats().IsEnabled(0));
   ASSERT_FALSE(hist.GetStats().IsEnabled(1));

   hist.Fill("a", "b");

   auto th2i = ConvertToTH2I(hist);
   ASSERT_TRUE(th2i);

   EXPECT_EQ(th2i->GetBinContent(1 + 5 * 2), 1);

   EXPECT_EQ(th2i->GetEntries(), 1);
   Double_t stats[7];
   th2i->GetStats(stats);
   EXPECT_EQ(stats[0], 1);
   EXPECT_EQ(stats[1], 1);
   EXPECT_EQ(stats[2], 0);
   EXPECT_EQ(stats[3], 0);
   EXPECT_EQ(stats[4], 0);
   EXPECT_EQ(stats[5], 0);
   EXPECT_EQ(stats[6], 0);
}

TEST(ConvertToTH2C, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<char> engine(axis, axis);

   auto th2c = ConvertToTH2C(engine);
   ASSERT_TRUE(th2c);
}

TEST(ConvertToTH2C, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<char> hist(axis, axis);

   auto th2c = ConvertToTH2C(hist);
   ASSERT_TRUE(th2c);
}

TEST(ConvertToTH2S, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<short> engine(axis, axis);

   auto th2s = ConvertToTH2S(engine);
   ASSERT_TRUE(th2s);
}

TEST(ConvertToTH2S, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<short> hist(axis, axis);

   auto th2s = ConvertToTH2S(hist);
   ASSERT_TRUE(th2s);
}

TEST(ConvertToTH2L, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<long> engineL(axis, axis);

   auto th2l = ConvertToTH2L(engineL);
   ASSERT_TRUE(th2l);

   RHistEngine<long long> engineLL(axis, axis);

   // Set one 64-bit long long value larger than what double can exactly represent.
   static constexpr long long Large = (1LL << 60) - 1;
   const std::array<RBinIndex, 2> indices = {1, 2};
   engineLL.SetBinContent(indices, Large);

   th2l = ConvertToTH2L(engineLL);
   ASSERT_TRUE(th2l);

   // Get the value via TArrayL::At and store into a variable to be sure about the type. During direct comparison, a
   // double return value may automatically promote Large to a double as well, introducing the truncation we want to
   // test against.
   const long long value = th2l->At(2 + (Bins + 2) * 3);
   EXPECT_EQ(value, Large);
}

TEST(ConvertToTH2L, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<long> histL(axis, axis);

   auto th2l = ConvertToTH2L(histL);
   ASSERT_TRUE(th2l);

   const RHist<long long> histLL(axis, axis);
   th2l = ConvertToTH2L(histLL);
   ASSERT_TRUE(th2l);
}

TEST(ConvertToTH2F, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine(axis, axis);

   engine.Fill(-100, 0.5, RWeight(0.25));
   for (std::size_t i = 0; i < Bins / 2; i++) {
      engine.Fill(i, 2 * i, RWeight(0.1 + i * 0.03));
   }
   engine.Fill(100, Bins - 0.5, RWeight(0.75));

   auto th2f = ConvertToTH2F(engine);
   ASSERT_TRUE(th2f);

   EXPECT_FLOAT_EQ(th2f->GetBinContent(0, 1), 0.25);
   for (std::size_t i = 0; i < Bins / 2; i++) {
      EXPECT_FLOAT_EQ(th2f->GetBinContent(i + 1, 2 * i + 1), 0.1 + i * 0.03);
   }
   EXPECT_EQ(th2f->GetBinContent(Bins + 1, Bins), 0.75);
}

TEST(ConvertToTH2F, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<float> hist(axis, axis);

   for (std::size_t i = 0; i < Bins; i++) {
      hist.Fill(i, 2 * i, RWeight(0.1 + i * 0.03));
   }

   auto th2f = ConvertToTH2F(hist);
   ASSERT_TRUE(th2f);

   ASSERT_EQ(hist.GetNEntries(), Bins);
   EXPECT_EQ(th2f->GetEntries(), Bins);
   Double_t stats[7];
   th2f->GetStats(stats);
   EXPECT_DOUBLE_EQ(stats[0], 7.7);
   EXPECT_DOUBLE_EQ(stats[1], 3.563);
   EXPECT_DOUBLE_EQ(stats[2], 93.1);
   EXPECT_DOUBLE_EQ(stats[3], 1330.0);
   EXPECT_DOUBLE_EQ(stats[4], 2 * 93.1);
   EXPECT_DOUBLE_EQ(stats[5], 4 * 1330.0);
   EXPECT_DOUBLE_EQ(stats[6], 0);
}

TEST(ConvertToTH2D, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<double> engineD(axis, axis);

   auto th2d = ConvertToTH2D(engineD);
   ASSERT_TRUE(th2d);

   RHistEngine<RBinWithError> engineE(axis, axis);
   for (std::size_t i = 0; i < Bins / 2; i++) {
      engineE.Fill(i, 2 * i, RWeight(0.1 + i * 0.03));
   }

   th2d = ConvertToTH2D(engineE);
   ASSERT_TRUE(th2d);
   const Double_t *sumw2 = th2d->GetSumw2()->GetArray();
   ASSERT_TRUE(sumw2 != nullptr);

   for (std::size_t i = 0; i < Bins / 2; i++) {
      const double weight = 0.1 + i * 0.03;
      EXPECT_EQ(th2d->GetBinContent(i + 1, 2 * i + 1), weight);
      EXPECT_EQ(sumw2[i + 1 + (Bins + 2) * (2 * i + 1)], weight * weight);
   }
}

TEST(ConvertToTH2D, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<double> histD(axis, axis);

   auto th2d = ConvertToTH2D(histD);
   ASSERT_TRUE(th2d);

   RHist<RBinWithError> histE(axis, axis);
   for (std::size_t i = 0; i < Bins; i++) {
      histE.Fill(i, 2 * i, RWeight(0.1 + i * 0.03));
   }

   th2d = ConvertToTH2D(histE);
   ASSERT_TRUE(th2d);

   ASSERT_EQ(histE.GetNEntries(), Bins);
   EXPECT_EQ(th2d->GetEntries(), Bins);
   Double_t stats[7];
   th2d->GetStats(stats);
   EXPECT_DOUBLE_EQ(stats[0], 7.7);
   EXPECT_DOUBLE_EQ(stats[1], 3.563);
   EXPECT_DOUBLE_EQ(stats[2], 93.1);
   EXPECT_DOUBLE_EQ(stats[3], 1330.0);
   EXPECT_DOUBLE_EQ(stats[4], 2 * 93.1);
   EXPECT_DOUBLE_EQ(stats[5], 4 * 1330.0);
   EXPECT_DOUBLE_EQ(stats[6], 0);
}
