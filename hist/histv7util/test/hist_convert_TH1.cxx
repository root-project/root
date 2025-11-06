#include "histutil_test.hxx"

#include <TAxis.h>
#include <TH1.h>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>

TEST(ConvertToTH1I, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   RHistEngine<int> engine(Bins, {0, Bins});

   engine.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
   }
   engine.Fill(100);

   // Fill bin 7 twice to test against accidental shifts.
   engine.Fill(7);

   auto th1i = ConvertToTH1I(engine);
   ASSERT_TRUE(th1i);
   EXPECT_TRUE(th1i->GetDirectory() == nullptr);
   ASSERT_EQ(th1i->GetDimension(), 1);
   ASSERT_EQ(th1i->GetNbinsX(), Bins);
   ASSERT_EQ(th1i->GetNbinsY(), 1);
   ASSERT_EQ(th1i->GetNbinsZ(), 1);

   const auto *xAxis = th1i->GetXaxis();
   EXPECT_FALSE(xAxis->IsVariableBinSize());
   EXPECT_EQ(xAxis->GetNbins(), Bins);
   EXPECT_EQ(xAxis->GetXmin(), 0.0);
   EXPECT_EQ(xAxis->GetXmax(), Bins);

   for (std::size_t i = 0; i < Bins + 2; i++) {
      // Bin 7 was filled twice.
      if (i == 7 + 1) {
         EXPECT_EQ(th1i->GetBinContent(i), 2);
      } else {
         EXPECT_EQ(th1i->GetBinContent(i), 1);
      }
   }

   EXPECT_EQ(th1i->GetEntries(), 0);
   Double_t stats[4];
   th1i->GetStats(stats);
   EXPECT_EQ(stats[0], 0);
   EXPECT_EQ(stats[1], 0);
   EXPECT_EQ(stats[2], 0);
   EXPECT_EQ(stats[3], 0);
}

TEST(ConvertToTH1I, RHistEngineNoFlowBins)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins}, /*enableFlowBins=*/false);
   RHistEngine<int> engine({axis});

   engine.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
   }
   engine.Fill(100);

   auto th1i = ConvertToTH1I(engine);
   ASSERT_TRUE(th1i);

   EXPECT_EQ(th1i->GetBinContent(0), 0);
   for (std::size_t i = 1; i <= Bins; i++) {
      EXPECT_EQ(th1i->GetBinContent(i), 1);
   }
   EXPECT_EQ(th1i->GetBinContent(Bins + 1), 0);
}

TEST(ConvertToTH1I, RHistEngineInvalid)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine({axis, axis});

   EXPECT_THROW(ConvertToTH1I(engine), std::invalid_argument);
}

TEST(ConvertToTH1I, RHist)
{
   static constexpr std::size_t Bins = 20;
   RHist<int> hist(Bins, {0, Bins});

   for (std::size_t i = 0; i < Bins; i++) {
      hist.Fill(i);
   }

   auto th1i = ConvertToTH1I(hist);
   ASSERT_TRUE(th1i);

   ASSERT_EQ(hist.GetNEntries(), Bins);
   EXPECT_EQ(th1i->GetEntries(), Bins);
   Double_t stats[4];
   th1i->GetStats(stats);
   EXPECT_EQ(stats[0], Bins);
   EXPECT_EQ(stats[1], Bins);
   EXPECT_EQ(stats[2], 190);
   EXPECT_EQ(stats[3], 2470);
}

TEST(ConvertToTH1C, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RHistEngine<char> engine(Bins, {0, Bins});

   auto th1c = ConvertToTH1C(engine);
   ASSERT_TRUE(th1c);
}

TEST(ConvertToTH1C, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RHist<char> hist(Bins, {0, Bins});

   auto th1c = ConvertToTH1C(hist);
   ASSERT_TRUE(th1c);
}

TEST(ConvertToTH1S, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RHistEngine<short> engine(Bins, {0, Bins});

   auto th1s = ConvertToTH1S(engine);
   ASSERT_TRUE(th1s);
}

TEST(ConvertToTH1S, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RHist<short> hist(Bins, {0, Bins});

   auto th1s = ConvertToTH1S(hist);
   ASSERT_TRUE(th1s);
}

TEST(ConvertToTH1L, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RHistEngine<long> engineL(Bins, {0, Bins});

   auto th1l = ConvertToTH1L(engineL);
   ASSERT_TRUE(th1l);

   RHistEngine<long long> engineLL(Bins, {0, Bins});

   // Set one 64-bit long long value larger than what double can exactly represent.
   static constexpr long long Large = (1LL << 60) - 1;
   const std::array<RBinIndex, 1> indices = {1};
   ROOT::Experimental::Internal::SetBinContent(engineLL, indices, Large);

   th1l = ConvertToTH1L(engineLL);
   ASSERT_TRUE(th1l);

   // Get the value via TArrayL::At and store into a variable to be sure about the type. During direct comparison, a
   // double return value may automatically promate Large to a double as well, introducing the truncation we want to
   // test against.
   const long long value = th1l->At(2);
   EXPECT_EQ(value, Large);
}

TEST(ConvertToTH1L, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RHist<long> histL(Bins, {0, Bins});

   auto th1l = ConvertToTH1L(histL);
   ASSERT_TRUE(th1l);

   const RHist<long long> histLL(Bins, {0, Bins});
   th1l = ConvertToTH1L(histLL);
   ASSERT_TRUE(th1l);
}

TEST(ConvertToTH1F, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   RHistEngine<float> engine(Bins, {0, Bins});

   engine.Fill(-100, RWeight(0.25));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i, RWeight(0.1 + i * 0.03));
   }
   engine.Fill(100, RWeight(0.75));

   auto th1f = ConvertToTH1F(engine);
   ASSERT_TRUE(th1f);

   EXPECT_FLOAT_EQ(th1f->GetBinContent(0), 0.25);
   for (std::size_t i = 1; i <= Bins; i++) {
      EXPECT_FLOAT_EQ(th1f->GetBinContent(i), 0.1 + (i - 1) * 0.03);
   }
   EXPECT_EQ(th1f->GetBinContent(Bins + 1), 0.75);
}

TEST(ConvertToTH1F, RHist)
{
   static constexpr std::size_t Bins = 20;
   RHist<float> hist(Bins, {0, Bins});

   for (std::size_t i = 0; i < Bins; i++) {
      hist.Fill(i, RWeight(0.1 + i * 0.03));
   }

   auto th1f = ConvertToTH1F(hist);
   ASSERT_TRUE(th1f);

   ASSERT_EQ(hist.GetNEntries(), Bins);
   EXPECT_EQ(th1f->GetEntries(), Bins);
   Double_t stats[4];
   th1f->GetStats(stats);
   EXPECT_DOUBLE_EQ(stats[0], 7.7);
   EXPECT_DOUBLE_EQ(stats[1], 3.563);
   EXPECT_DOUBLE_EQ(stats[2], 93.1);
   EXPECT_DOUBLE_EQ(stats[3], 1330.0);
}

TEST(ConvertToTH1D, RHistEngine)
{
   static constexpr std::size_t Bins = 20;
   const RHistEngine<double> engineD(Bins, {0, Bins});

   auto th1d = ConvertToTH1D(engineD);
   ASSERT_TRUE(th1d);

   RHistEngine<RBinWithError> engineE(Bins, {0, Bins});
   for (std::size_t i = 0; i < Bins; i++) {
      engineE.Fill(i, RWeight(0.1 + i * 0.03));
   }

   th1d = ConvertToTH1D(engineE);
   ASSERT_TRUE(th1d);
   const Double_t *sumw2 = th1d->GetSumw2()->GetArray();
   ASSERT_TRUE(sumw2 != nullptr);

   for (std::size_t i = 1; i <= Bins; i++) {
      const double weight = 0.1 + (i - 1) * 0.03;
      EXPECT_EQ(th1d->GetBinContent(i), weight);
      EXPECT_EQ(sumw2[i], weight * weight);
   }
}

TEST(ConvertToTH1D, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RHist<double> histD(Bins, {0, Bins});

   auto th1d = ConvertToTH1D(histD);
   ASSERT_TRUE(th1d);

   RHist<RBinWithError> histE(Bins, {0, Bins});
   for (std::size_t i = 0; i < Bins; i++) {
      histE.Fill(i, RWeight(0.1 + i * 0.03));
   }

   th1d = ConvertToTH1D(histE);
   ASSERT_TRUE(th1d);

   ASSERT_EQ(histE.GetNEntries(), Bins);
   EXPECT_EQ(th1d->GetEntries(), Bins);
   Double_t stats[4];
   th1d->GetStats(stats);
   EXPECT_DOUBLE_EQ(stats[0], 7.7);
   EXPECT_DOUBLE_EQ(stats[1], 3.563);
   EXPECT_DOUBLE_EQ(stats[2], 93.1);
   EXPECT_DOUBLE_EQ(stats[3], 1330.0);
}
