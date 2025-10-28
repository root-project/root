#include "histutil_test.hxx"

#include <TAxis.h>
#include <TH1.h>

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

   const RHistEngine<long long> engineLL(Bins, {0, Bins});
   th1l = ConvertToTH1L(engineLL);
   ASSERT_TRUE(th1l);

   // TODO: Test that 64-bit long long values are not truncated to double
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
   const RHistEngine<double> engine(Bins, {0, Bins});

   auto th1d = ConvertToTH1D(engine);
   ASSERT_TRUE(th1d);
}

TEST(ConvertToTH1D, RHist)
{
   static constexpr std::size_t Bins = 20;
   const RHist<double> hist(Bins, {0, Bins});

   auto th1d = ConvertToTH1D(hist);
   ASSERT_TRUE(th1d);
}
