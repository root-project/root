#include "gtest/gtest.h"

#include "TH3D.h"

#include <random>
#include <thread>

TEST(TH3L, SetBinContent)
{
   TH3L h("", "", 1, 0, 1, 1, 0, 1, 1, 0, 1);
   // Something that does not fit into Int_t, but is exactly representable in Double_t
   static constexpr long long Large = 1LL << 42;
   h.SetBinContent(1, 1, 1, Large);
   EXPECT_EQ(h.GetBinContent(1, 1, 1), Large);
}

#ifdef TH3D_FILL_THREADSAFE

TEST(TH3D, FillThreadSafe)
{
   TH3D h0("h0", "h0", 50, 0, 50, 50, 0, 50, 50, 0, 50);
   h0.Sumw2();
   TH3D h1(h0);

   std::vector<std::tuple<double, double, double, double>> data(100000);

   std::mt19937 gen(123456);
   std::uniform_real_distribution<> dis(-1.0, 51.);

   std::generate(data.begin(), data.end(), [&]() { return std::tuple{dis(gen), dis(gen), dis(gen), dis(gen)}; });

   constexpr unsigned int nThread = 50;
   std::vector<std::thread> threads;
   threads.reserve(nThread);
   for (unsigned int i = 0; i < nThread; ++i) {
      threads.emplace_back(
         [&](unsigned int threadId) {
            for (unsigned int j = threadId; j < data.size(); j += nThread) {
               std::apply([&](auto... args) { ROOT::Internal::FillThreadSafe(h0, args...); }, data[j]);
            }
         },
         i);
   }

   for (const auto &tuple : data) {
      std::apply([&](auto... args) { h1.Fill(args...); }, tuple);
   }

   for (auto &thread : threads) {
      thread.join();
   }

   // Test this accurately, because here we are literally just counting +1., +1., etc:
   EXPECT_EQ(h0.GetEntries(), h1.GetEntries());
   EXPECT_NEAR(h0.GetEffectiveEntries(), h1.GetEffectiveEntries(), h0.GetEffectiveEntries() * 1.E-12);

   for (auto axis : {1, 2, 3, 11, 12, 13}) {
      // Since we fill with weights between -1. to 51., different execution order can significantly change the squared
      // sums
      EXPECT_NEAR(h0.GetMean(axis), h1.GetMean(axis), h0.GetMean() * 1.E-9);
   }

   const auto nbin = (h0.GetNbinsX() + 2) * (h0.GetNbinsY() + 2) + (h0.GetNbinsZ() + 2);
   for (int i = 0; i < nbin; ++i) {
      EXPECT_NEAR(h0.GetBinContent(i), h1.GetBinContent(i), fabs(h0.GetBinContent(i)) * 1.E-12);
   }
}

TEST(TH3D, FillAtomicNoSumW2)
{
   TH3D h0("h0", "h0", 50, 0, 50, 50, 0, 50, 50, 0, 50);
   // Forgetting to call SumW2, we cannot fill with weights:
   EXPECT_THROW(ROOT::Internal::FillThreadSafe(h0, 1., 2., 3., 4.), std::logic_error);
}

#endif

// ROOT-10678
TEST(TH3D, InterpolateCloseToEdge)
{
   TH3F hist("3D", "boring histo", 20, 0, 20, 20, 0, 20, 20, 0, 20);

   for (int i = 0; i < hist.GetNbinsX(); i++)
      for (int j = 0; j < hist.GetNbinsY(); j++)
         for (int k = 0; k < hist.GetNbinsZ(); k++)
            hist.SetBinContent(i + 1, j + 1, k + 1, i + 100. * j + 10000. * k);

   EXPECT_DOUBLE_EQ(hist.Interpolate(hist.GetXaxis()->GetBinCenter(2), hist.GetYaxis()->GetBinCenter(3),
                                     hist.GetZaxis()->GetBinCenter(4)),
                    1. + 100. * 2. + 10000. * 3);
   EXPECT_DOUBLE_EQ(hist.Interpolate(hist.GetXaxis()->GetBinCenter(2) + 0.5, hist.GetYaxis()->GetBinCenter(3) + 0.4,
                                     hist.GetZaxis()->GetBinCenter(4) + 0.3),
                    1. + 0.5 + 100. * 2.4 + 10000. * 3.3);

   EXPECT_DOUBLE_EQ(hist.Interpolate(0., 0., 5.), 10000. * 4.5);
   EXPECT_DOUBLE_EQ(hist.Interpolate(0.3, 19.9, 5.), 100. * 19 + 10000. * 4.5);
   EXPECT_DOUBLE_EQ(hist.Interpolate(0.8, 19.9, 5.), 0.3 + 100. * 19 + 10000. * 4.5);
}
