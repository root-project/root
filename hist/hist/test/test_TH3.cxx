#include "gtest/gtest.h"

#include "TH3D.h"

#include <random>
#include <thread>

#ifdef __cpp_lib_atomic_ref

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