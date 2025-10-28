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
