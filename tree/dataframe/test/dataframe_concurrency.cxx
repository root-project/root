#include <ROOT/RDataFrame.hxx>
#include <ROOT/TThreadExecutor.hxx>
#include <atomic>
#include <chrono>
#include <thread>

#include "gtest/gtest.h"

#ifdef R__USE_IMT
TEST(RDFConcurrency, NestedParallelismBetweenDefineCalls)
{
   ROOT::EnableImplicitMT();

   // this Define will return unique values from 0 to nEntries - 1 (over all threads)
   const auto nEntries = 100000u;
   std::atomic_int i(0);
   auto df = ROOT::RDataFrame(nEntries).Define("i", [&] { return i++; });

   // this lambda will be used to introduce nested parallelism via a dummy Filter
   auto manysleeps = [&] {
      ROOT::TThreadExecutor().Foreach(
         [] { std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() / RAND_MAX * 200)); }, 8);
      return true;
   };

   // first Take triggers the computation of i, then the Filter executes, then Take accesses the cached value of i
   // hopefully it won't have changed in the meanwhile!
   auto res1 = df.Take<int>("i");
   auto res2 = df.Filter(manysleeps).Take<int>("i");

   // check both Takes return vectors with all numbers from 0 to nEntries - 1
   auto checkAllNumbersAreThere = [](std::vector<int> &vec) {
      std::sort(vec.begin(), vec.end());
      const int s = vec.size();
      for (auto j = 0; j < s; ++j)
         EXPECT_EQ(vec[j], j);
   };

   checkAllNumbersAreThere(*res1);
   checkAllNumbersAreThere(*res2);

   ROOT::DisableImplicitMT();
}
#endif
