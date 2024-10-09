#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/BufferedFillWrapper.hxx"
#include "TH2D.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <thread>
#include <vector>

TEST(RDataFrameUtils, BufferedFillWrapper)
{
   using ROOT::Internal::RDF::BufferedFillWrapper;

   constexpr int nThread = 20;
   constexpr int nFill = 100;
   constexpr double fillWeight = 0.5;

   auto histo = std::make_shared<TH2D>("histo", "Histo;x;y", 10, 0., 10., 10, 0., 10.);
   BufferedFillWrapper<TH2D, double, int> bufferedFill(std::move(histo), 10);

   auto histo2 = std::make_shared<TH2D>("histo2", "Histo;x;y", 10, 0., 10., 10, 0., 10.);
   BufferedFillWrapper<TH2D, double, double, double> bufferedFillWeight(std::move(histo2), 10000);

   auto filler = [&bufferedFill, &bufferedFillWeight](std::size_t N, double valx, double valy, double weight) {
      for (unsigned int i = 0; i < N; ++i) {
         bufferedFill.Fill(valx, valy /*, weight*/);
         bufferedFillWeight.Fill(valx, valy, weight);
      }
   };

   {
      std::vector<std::thread> threads;
      threads.reserve(nThread);
      for (unsigned int i = 0; i < nThread; ++i) {
         threads.emplace_back(filler, nFill, 1, i, fillWeight);
      }
      for (auto &thread : threads)
         thread.join();
   }

   auto histoHandle = bufferedFill.Get();
   EXPECT_FLOAT_EQ(histoHandle->GetEntries(), nThread * nFill);
   EXPECT_FLOAT_EQ(histoHandle->GetBinContent(histoHandle->GetBin(0, 0)), 0.);
   EXPECT_FLOAT_EQ(histoHandle->GetBinContent(histoHandle->GetBin(1, 1)), 0.);
   EXPECT_FLOAT_EQ(histoHandle->GetBinContent(histoHandle->GetBin(2, 1)), nFill);
   EXPECT_FLOAT_EQ(histoHandle->GetBinContent(histoHandle->GetBin(2, 12)), (nThread - 10.) * nFill);

   histo2 = bufferedFillWeight.Release();
   EXPECT_FLOAT_EQ(histo2->GetEntries(), nThread * nFill);
   EXPECT_FLOAT_EQ(histo2->GetBinContent(histo2->GetBin(0, 0)), 0.);
   EXPECT_FLOAT_EQ(histo2->GetBinContent(histo2->GetBin(1, 1)), 0.);
   EXPECT_FLOAT_EQ(histo2->GetBinContent(histo2->GetBin(2, 1)), nFill * fillWeight);
   EXPECT_FLOAT_EQ(histo2->GetBinContent(histo2->GetBin(2, 12)), (nThread - 10.) * nFill * fillWeight);
}