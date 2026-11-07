#include "histutil_test.hxx"

#include <TH3.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>

TEST(ConvertToTH3I, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis, axis, axis);

   engine.SetBinContent(RBinIndex::Underflow(), 0, 1, 1000);
   engine.SetBinContent(RBinIndex::Underflow(), 2, 3, 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.SetBinContent(i, RBinIndex::Underflow(), 1, 100 * i);
      for (std::size_t j = 0; j < Bins; j++) {
         engine.SetBinContent(i, j, RBinIndex::Underflow(), 400 * j);
         for (std::size_t k = 0; k < Bins; k++) {
            engine.SetBinContent(i, j, k, (i * Bins + j) * Bins + k);
         }
         engine.SetBinContent(i, j, RBinIndex::Overflow(), 800 * j);
      }
      engine.SetBinContent(i, RBinIndex::Overflow(), 4, 200 * i);
   }
   engine.SetBinContent(RBinIndex::Overflow(), 4, 5, 3000);
   engine.SetBinContent(RBinIndex::Overflow(), 6, 7, 4000);

   auto th3i = ConvertToTH3I(engine);
   ASSERT_TRUE(th3i);
   EXPECT_TRUE(th3i->GetDirectory() == nullptr);
   ASSERT_EQ(th3i->GetDimension(), 3);
   ASSERT_EQ(th3i->GetNbinsX(), Bins);
   ASSERT_EQ(th3i->GetNbinsY(), Bins);
   ASSERT_EQ(th3i->GetNbinsZ(), Bins);

   EXPECT_EQ(th3i->GetBinContent(0, 1, 2), 1000);
   EXPECT_EQ(th3i->GetBinContent(0, 3, 4), 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      EXPECT_EQ(th3i->GetBinContent(i + 1, 0, 2), 100 * i);
      for (std::size_t j = 0; j < Bins; j++) {
         EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, 0), 400 * j);
         for (std::size_t k = 0; k < Bins; k++) {
            EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, k + 1), (i * Bins + j) * Bins + k);
         }
         EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, Bins + 1), 800 * j);
      }
      EXPECT_EQ(th3i->GetBinContent(i + 1, Bins + 1, 5), 200 * i);
   }
   EXPECT_EQ(th3i->GetBinContent(Bins + 1, 5, 6), 3000);
   EXPECT_EQ(th3i->GetBinContent(Bins + 1, 7, 8), 4000);

   EXPECT_EQ(th3i->GetEntries(), 0);
   Double_t stats[11];
   th3i->GetStats(stats);
   for (std::size_t i = 0; i < 11; i++) {
      EXPECT_EQ(stats[i], 0);
   }
}

TEST(ConvertToTH3I, RHistEngineNoFlowBins)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins}, /*enableFlowBins=*/false);
   RHistEngine<int> engine(axis, axis, axis);

   // Flow bins are disabled, so these fills will be discarded.
   engine.Fill(-100, 0.5, 1.5);
   engine.Fill(-100, 2.5, 3.5);
   for (std::size_t i = 0; i < Bins; i++) {
      // Flow bins are disabled, so this fill will be discarded.
      engine.Fill(i + 0.5, -100, 1.5);
      for (std::size_t j = 0; j < Bins; j++) {
         // Flow bins are disabled, so this fill will be discarded.
         engine.Fill(i + 0.5, j + 0.5, -100);
         for (std::size_t k = 0; k < Bins; k++) {
            engine.SetBinContent(i, j, k, (i * Bins + j) * Bins + k);
         }
         // Flow bins are disabled, so this fill will be discarded.
         engine.Fill(i + 0.5, j + 0.5, 100);
      }
      // Flow bins are disabled, so this fill will be discarded.
      engine.Fill(i + 0.5, 100, 4.5);
   }
   // Flow bins are disabled, so these fills will be discarded.
   engine.Fill(100, 4.5, 5.5);
   engine.Fill(100, 6.5, 7.5);

   auto th3i = ConvertToTH3I(engine);
   ASSERT_TRUE(th3i);

   EXPECT_EQ(th3i->GetBinContent(0, 1, 2), 0);
   EXPECT_EQ(th3i->GetBinContent(0, 3, 4), 0);
   for (std::size_t i = 0; i < Bins; i++) {
      EXPECT_EQ(th3i->GetBinContent(i + 1, 0, 1), 0);
      for (std::size_t j = 0; j < Bins; j++) {
         EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, 0), 0);
         for (std::size_t k = 0; k < Bins; k++) {
            EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, k + 1), (i * Bins + j) * Bins + k);
         }
         EXPECT_EQ(th3i->GetBinContent(i + 1, j + 1, Bins + 1), 0);
      }
      EXPECT_EQ(th3i->GetBinContent(i + 1, Bins + 1, 4), 0);
   }
   EXPECT_EQ(th3i->GetBinContent(Bins + 1, 5, 6), 0);
   EXPECT_EQ(th3i->GetBinContent(Bins + 1, 7, 8), 0);
}

TEST(ConvertToTH3I, RHistEngineInvalid)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine(axis);

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
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<int> hist(axis, axis, axis);
   const std::array<RBinIndex, 3> indices = {1, 2, 3};
   hist.SetBinContent(indices, 42);
   ASSERT_TRUE(hist.GetStats().IsTainted());

   auto th3i = ConvertToTH3I(hist);
   ASSERT_TRUE(th3i);

   EXPECT_EQ(th3i->GetBinContent(2 + (Bins + 2) * (3 + (Bins + 2) * 4)), 42);

   EXPECT_EQ(th3i->GetEntries(), 0);
   Double_t stats[11];
   th3i->GetStats(stats);
   for (std::size_t i = 0; i < 11; i++) {
      EXPECT_EQ(stats[i], 0);
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

   EXPECT_EQ(th3i->GetBinContent(1 + 5 * (2 + 5 * 3)), 1);

   EXPECT_EQ(th3i->GetEntries(), 1);
   Double_t stats[11];
   th3i->GetStats(stats);
   EXPECT_EQ(stats[0], 1);
   EXPECT_EQ(stats[1], 1);
   for (std::size_t i = 2; i < 11; i++) {
      EXPECT_EQ(stats[i], 0);
   }
}

TEST(ConvertToTH3C, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<char> engine(axis, axis, axis);

   auto th3c = ConvertToTH3C(engine);
   ASSERT_TRUE(th3c);
}

TEST(ConvertToTH3C, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<char> hist(axis, axis, axis);

   auto th3c = ConvertToTH3C(hist);
   ASSERT_TRUE(th3c);
}

TEST(ConvertToTH3S, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<short> engine(axis, axis, axis);

   auto th3s = ConvertToTH3S(engine);
   ASSERT_TRUE(th3s);
}

TEST(ConvertToTH3S, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<short> hist(axis, axis, axis);

   auto th3s = ConvertToTH3S(hist);
   ASSERT_TRUE(th3s);
}

TEST(ConvertToTH3L, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<long> engineL(axis, axis, axis);

   auto th3l = ConvertToTH3L(engineL);
   ASSERT_TRUE(th3l);

   RHistEngine<long long> engineLL(axis, axis, axis);

   // Set one 64-bit long long value larger than what double can exactly represent.
   static constexpr long long Large = (1LL << 60) - 1;
   const std::array<RBinIndex, 3> indices = {1, 2, 3};
   engineLL.SetBinContent(indices, Large);

   th3l = ConvertToTH3L(engineLL);
   ASSERT_TRUE(th3l);

   // Get the value via TArrayL::At and store into a variable to be sure about the type. During direct comparison, a
   // double return value may automatically promote Large to a double as well, introducing the truncation we want to
   // test against.
   const long long value = th3l->At(2 + (Bins + 2) * (3 + (Bins + 2) * 4));
   EXPECT_EQ(value, Large);
}

TEST(ConvertToTH3L, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<long> histL(axis, axis, axis);

   auto th3l = ConvertToTH3L(histL);
   ASSERT_TRUE(th3l);

   const RHist<long long> histLL(axis, axis, axis);
   th3l = ConvertToTH3L(histLL);
   ASSERT_TRUE(th3l);
}

TEST(ConvertToTH3F, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine(axis, axis, axis);

   engine.Fill(-100, 0.5, 1.5, RWeight(0.25));
   for (std::size_t i = 0; i < Bins / 3; i++) {
      engine.Fill(i, 2 * i, 3 * i, RWeight(0.1 + i * 0.03));
   }
   engine.Fill(100, Bins - 0.5, Bins - 1.5, RWeight(0.75));

   auto th3f = ConvertToTH3F(engine);
   ASSERT_TRUE(th3f);

   EXPECT_FLOAT_EQ(th3f->GetBinContent(0, 1, 2), 0.25);
   for (std::size_t i = 0; i < Bins / 3; i++) {
      EXPECT_FLOAT_EQ(th3f->GetBinContent(i + 1, 2 * i + 1, 3 * i + 1), 0.1 + i * 0.03);
   }
   EXPECT_EQ(th3f->GetBinContent(Bins + 1, Bins, Bins - 1), 0.75);
}

TEST(ConvertToTH3F, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHist<float> hist(axis, axis, axis);

   for (std::size_t i = 0; i < Bins; i++) {
      hist.Fill(i, 2 * i, 3 * i, RWeight(0.1 + i * 0.03));
   }

   auto th3f = ConvertToTH3F(hist);
   ASSERT_TRUE(th3f);

   ASSERT_EQ(hist.GetNEntries(), Bins);
   EXPECT_EQ(th3f->GetEntries(), Bins);
   Double_t stats[11];
   th3f->GetStats(stats);
   EXPECT_DOUBLE_EQ(stats[0], 7.7);
   EXPECT_DOUBLE_EQ(stats[1], 3.563);
   EXPECT_DOUBLE_EQ(stats[2], 93.1);
   EXPECT_DOUBLE_EQ(stats[3], 1330.0);
   EXPECT_DOUBLE_EQ(stats[4], 2 * 93.1);
   EXPECT_DOUBLE_EQ(stats[5], 4 * 1330.0);
   EXPECT_DOUBLE_EQ(stats[6], 0);
   EXPECT_DOUBLE_EQ(stats[7], 3 * 93.1);
   EXPECT_DOUBLE_EQ(stats[8], 9 * 1330.0);
   EXPECT_DOUBLE_EQ(stats[9], 0);
   EXPECT_DOUBLE_EQ(stats[10], 0);
}

TEST(ConvertToTH3D, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<double> engineD(axis, axis, axis);

   auto th3d = ConvertToTH3D(engineD);
   ASSERT_TRUE(th3d);

   RHistEngine<RBinWithError> engineE(axis, axis, axis);
   for (std::size_t i = 0; i < Bins / 3; i++) {
      engineE.Fill(i, 2 * i, 3 * i, RWeight(0.1 + i * 0.03));
   }

   th3d = ConvertToTH3D(engineE);
   ASSERT_TRUE(th3d);
   const Double_t *sumw2 = th3d->GetSumw2()->GetArray();
   ASSERT_TRUE(sumw2 != nullptr);

   for (std::size_t i = 0; i < Bins / 3; i++) {
      const double weight = 0.1 + i * 0.03;
      EXPECT_EQ(th3d->GetBinContent(i + 1, 2 * i + 1, 3 * i + 1), weight);
      EXPECT_EQ(sumw2[i + 1 + (Bins + 2) * (2 * i + 1 + (Bins + 2) * (3 * i + 1))], weight * weight);
   }
}

TEST(ConvertToTH3D, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHist<double> histD(axis, axis, axis);

   auto th3d = ConvertToTH3D(histD);
   ASSERT_TRUE(th3d);

   RHist<RBinWithError> histE(axis, axis, axis);
   for (std::size_t i = 0; i < Bins; i++) {
      histE.Fill(i, 2 * i, 3 * i, RWeight(0.1 + i * 0.03));
   }

   th3d = ConvertToTH3D(histE);
   ASSERT_TRUE(th3d);

   ASSERT_EQ(histE.GetNEntries(), Bins);
   EXPECT_EQ(th3d->GetEntries(), Bins);
   Double_t stats[11];
   th3d->GetStats(stats);
   EXPECT_DOUBLE_EQ(stats[0], 7.7);
   EXPECT_DOUBLE_EQ(stats[1], 3.563);
   EXPECT_DOUBLE_EQ(stats[2], 93.1);
   EXPECT_DOUBLE_EQ(stats[3], 1330.0);
   EXPECT_DOUBLE_EQ(stats[4], 2 * 93.1);
   EXPECT_DOUBLE_EQ(stats[5], 4 * 1330.0);
   EXPECT_DOUBLE_EQ(stats[6], 0);
   EXPECT_DOUBLE_EQ(stats[7], 3 * 93.1);
   EXPECT_DOUBLE_EQ(stats[8], 9 * 1330.0);
   EXPECT_DOUBLE_EQ(stats[9], 0);
   EXPECT_DOUBLE_EQ(stats[10], 0);
}
