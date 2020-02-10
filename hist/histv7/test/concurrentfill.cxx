#include "gtest/gtest.h"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistConcurrentFill.hxx"

#include <iostream>
#include <future>

using namespace ROOT;

using Filler_t = Experimental::RHistConcurrentFiller<Experimental::RH2D, 1024>;

void theTask(Filler_t filler)
{
   for (int i = 0; i < 3000; ++i) {
      filler.Fill({(double)i/100, (double)i/10});
   }
}

void concurrentHistFill(Experimental::RH2D & hist)
{
   Experimental::RHistConcurrentFillManager<Experimental::RH2D> fillMgr(hist);

   std::array<std::thread, 2> threads;

   for (auto &thr : threads) {
      thr = std::thread(theTask, fillMgr.MakeFiller());
   }

   for (auto &thr : threads)
      thr.join();
}

// Test consistancy of the hist after concurrentfill
TEST(ConcurrentFillTest, HistConsistancy)
{
   Experimental::RH2D hist{{100, 0., 1.}, {{0., 1., 2., 3., 10.}}};

   concurrentHistFill(hist);

   EXPECT_EQ(2 * 3000, hist.GetEntries());

   EXPECT_FLOAT_EQ(2.f, hist.GetBinContent({(double)42/100, (double)42/10}));
}
