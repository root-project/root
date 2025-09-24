#include <ROOT/RConfig.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/TThreadExecutor.hxx>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#ifndef R__B64
static const auto NUM_THREADS = 8u;
#else
static const auto NUM_THREADS = 0u;
#endif

template <typename T>
void expect_vec_eq(const std::vector<T> &v1, const std::vector<T> &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (decltype(v1.size()) i{}; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

#ifdef R__USE_IMT
TEST(RDFConcurrency, NestedParallelismBetweenDefineCalls)
{
   ROOT::EnableImplicitMT(NUM_THREADS);

   // this Define will return unique values from 0 to nEntries - 1 (over all threads)
   const auto nEntries = 100000u;
   std::atomic_int i(0);
   auto df = ROOT::RDataFrame(nEntries).Define("i", [&] { return i++; });

   // this lambda will be used to introduce nested parallelism via a dummy Filter
   auto manysleeps = [&] {
      ROOT::TThreadExecutor(NUM_THREADS).Foreach(
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

// ROOT-10346:
// [DF] Warn on mismatch between slot pool size and effective number of slots
TEST(RDFSimpleTests, ThrowOnPoolSizeMismatch)
{
   const unsigned int nslots = std::min(3u, std::thread::hardware_concurrency());

   // pool created after RDF
   if (nslots > 1) {
      ROOT::RDataFrame df(1);
      ROOT::EnableImplicitMT(nslots);
      try {
         df.Count().GetValue();
      } catch (const std::runtime_error &e) {
         const auto expected_msg = "RLoopManager::Run: when the RDataFrame was constructed the number of slots required "
                                   "was 1, but when starting the event loop it was " + std::to_string(nslots) + ". Maybe EnableImplicitMT() was "
                                   "called after the RDataFrame was constructed?";
         EXPECT_STREQ(e.what(), expected_msg.c_str());
      }
      ROOT::DisableImplicitMT();
   }

   // pool deleted after RDF creation
   if (nslots > 1) {
      ROOT::EnableImplicitMT(nslots);
      ROOT::RDataFrame df(1);
      ROOT::DisableImplicitMT();
      try {
         df.Count().GetValue();
      } catch (const std::runtime_error &e) {
         const auto expected_msg = "RLoopManager::Run: when the RDataFrame was constructed the number of slots required "
                                   "was " + std::to_string(nslots) + ", but when starting the event loop it was 1. Maybe DisableImplicitMT() was "
                                   "called after the RDataFrame was constructed?";
         EXPECT_STREQ(e.what(), expected_msg.c_str());
      }
   }
}


void SimpleParallelRDFs()
{
   // Run the RDF construction and the event loop in parallel
   auto func = [] {
      std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() / RAND_MAX * 200));
      ROOT::RDataFrame df(10);
      return df.Define("x", [](ULong64_t ievt) { return ievt; }, {"rdfentry_"}).Mean<ULong64_t>("x").GetValue();
   };

   ROOT::TThreadExecutor pool(NUM_THREADS);
   auto res = pool.Map(func, NUM_THREADS);

   const auto ref = func();

   for(auto& r : res) {
      EXPECT_EQ(r, ref);
   }
}

TEST(RDFConcurrency, SimpleParallelRDFsEnableThreadSafety)
{
   ROOT::EnableThreadSafety();
   SimpleParallelRDFs();
}

TEST(RDFConcurrency, SimpleParallelRDFsEnableImplicitMT)
{
   ROOT::EnableImplicitMT(NUM_THREADS);
   SimpleParallelRDFs();
   ROOT::DisableImplicitMT();
}

void SimpleParallelRDFLoops()
{
   // Run only the event loops in parallel
   auto create_df = [] {
         std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() / RAND_MAX * 200));
         ROOT::RDataFrame df(10);
         return df.Define("x", [](ULong64_t ievt) {return ievt;}, {"rdfentry_"}).Mean<ULong64_t>("x");
      };

   std::vector<ROOT::RDF::RResultPtr<double>> vals(64);
   for (auto i = 0u; i < vals.size(); i++)
       vals[i] = create_df();

   ROOT::TThreadExecutor pool(NUM_THREADS);
   auto func = [](ROOT::RDF::RResultPtr<double> rptr){ return rptr.GetValue(); };
   auto res = pool.Map(func, vals);

   auto ref = func(create_df());

   for(auto& r : res) {
      EXPECT_EQ(r, ref);
   }
}

TEST(RDFConcurrency, SimpleParallelRDFLoopsEnableThreadSafety)
{
   ROOT::EnableThreadSafety();
   SimpleParallelRDFLoops();
}

TEST(RDFConcurrency, SimpleParallelRDFLoopsEnableImplicitMT)
{
   ROOT::EnableImplicitMT(NUM_THREADS);
   SimpleParallelRDFLoops();
   ROOT::DisableImplicitMT();
}

void ParallelRDFSnapshots()
{
   // Run the RDF construction and the event loop in parallel
   const auto nevts = 100u;
   auto func = [&](int i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() / RAND_MAX * 200));
      ROOT::RDataFrame df(nevts);
      df.Define("x", [](ULong64_t ievt) { return int(ievt); }, {"rdfentry_"})
         .Snapshot("tree", "dataframe_parallel_snapshots_" + std::to_string(i) + ".root", {"x"});
   };

   ROOT::TThreadExecutor pool(NUM_THREADS);
   std::vector<int> vals = {0, 1, 2, 3, 4, 5, 6, 7};
   pool.Foreach(func, vals);

   // Because a parallel snapshot does not preserve ordering of the events, we can just test
   // an aggregate such as the sum of all values.
   auto sum_ref = 0u;
   for (auto i = 0u; i < nevts; i++) sum_ref += i;
   for(auto& v : vals) {
      ROOT::RDataFrame df("tree", "dataframe_parallel_snapshots_" + std::to_string(v) + ".root");
      auto col = df.Take<int>("x");
      int sum_col = 0;
      for (auto i = 0u; i < nevts; i++)
         sum_col += col->at(i);
      EXPECT_EQ(sum_col, sum_ref);
   }
}

TEST(RDFConcurrency, ParallelRDFSnapshotsEnableThreadSafety)
{
   ROOT::EnableThreadSafety();
   ParallelRDFSnapshots();
}

TEST(RDFConcurrency, ParallelRDFSnapshotsEnableImplicitMT)
{
   ROOT::EnableImplicitMT(NUM_THREADS);
   ParallelRDFSnapshots();
   ROOT::DisableImplicitMT();
}

void ParallelRDFCaches()
{
   // Run the RDF construction and the event loop in parallel
   // and use an intermediate cache in the computation graph
   const auto nevts = 100u;
   auto func = [&] {
         std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() / RAND_MAX * 200));
         ROOT::RDataFrame df(nevts);
         auto cache = df.Define("x", [](ULong64_t ievt) {return (int)ievt;}, {"rdfentry_"}).Cache<int>({"x"});
         return cache.Sum<int>("x").GetValue();
      };

   ROOT::TThreadExecutor pool(NUM_THREADS);
   auto res = pool.Map(func, 8);

   auto sum_ref = 0u;
   for (auto i = 0u; i < nevts; i++) sum_ref += i;
   for (auto i = 0u; i < res.size(); i++) {
      EXPECT_EQ(res[i], sum_ref);
   }
}

TEST(RDFConcurrency, ParallelRDFCachesEnableThreadSafety)
{
   ROOT::EnableThreadSafety();
   ParallelRDFCaches();
}

TEST(RDFConcurrency, ParallelRDFCachesEnableImplicitMT)
{
   ROOT::EnableImplicitMT(NUM_THREADS);
   ParallelRDFCaches();
   ROOT::DisableImplicitMT();
}
#endif

// Check that multiple RDF can JIT at the same time without interfering
// with each other
TEST(RDFConcurrency, JITWithManyThreads)
{
   ROOT::EnableThreadSafety();

   std::vector<int> expected(25);
   std::iota(expected.begin(), expected.end(), 0);

   std::vector<int> results(25);
   auto do_work = [&results](int slot) {
      const int begin{slot * 5};
      const int end{begin + 5};
      for (int i = begin; i < end; i++) {
         results[i] = ROOT::RDataFrame{1}.Define("x", std::to_string(i)).Sum<int>("x").GetValue();
      }
   };

   std::vector<std::thread> threads;
   threads.reserve(5);
   for (int slot = 0; slot < 5; slot++)
      threads.emplace_back(do_work, slot);

   for (auto &&t : threads)
      t.join();

   expect_vec_eq(results, expected);
}

TEST(RDFConcurrency, JITManyThreadsAndExceptions)
{
   ROOT::EnableThreadSafety();

   std::condition_variable cv_1;
   std::condition_variable cv_2;
   std::condition_variable cv_3;
   std::mutex m_1;
   std::mutex m_2;
   std::mutex m_3;
   bool ready_1{false};
   bool ready_2{false};
   bool ready_3{false};

   int res_work_2{};
   int res_work_3{};

   auto work_1 = [&]() {
      std::lock_guard lk_1{m_1};
      try {
         ROOT::RDataFrame df{1};
         auto df1 = df.Define("work_1_x", "throw std::runtime_error(\"Error in RDF!\"); return 42;");
         df1.Sum<int>("work_1_x").GetValue();
      } catch (...) {
      }
      ready_1 = true;
      cv_1.notify_all();
   };

   auto work_2 = [&]() {
      std::unique_lock lk_1(m_1);
      std::lock_guard lk_2{m_2};
      std::unique_lock lk_3{m_3};
      ROOT::RDataFrame df{1};
      auto df1 = df.Define("work_2_x", "42");
      cv_1.wait(lk_1, [&ready_1] { return ready_1; });
      auto df2 = df1.Define("work_2_y", "58");
      cv_3.wait(lk_3, [&ready_3] { return ready_3; });
      auto df3 = df2.Define("work_2_z", "work_2_x + work_2_y");
      res_work_2 = df3.Sum<int>("work_2_z").GetValue();
      ready_2 = true;
      cv_2.notify_one();
   };

   auto work_3 = [&]() {
      std::unique_lock lk_1(m_1);
      std::unique_lock lk_2(m_2);
      std::unique_lock lk_3(m_3);
      ROOT::RDataFrame df{1};
      auto df1 = df.Define("work_3_x", "11");
      auto df2 = df1.Define("work_3_y", "work_3_x * 2");
      cv_1.wait(lk_1, [&ready_1] { return ready_1; });
      cv_2.wait(lk_2, [&ready_2] { return ready_2; });
      auto df3 = df2.Define("work_3_z", "work_3_y + work_3_x");
      cv_3.wait(lk_3, [&ready_3] { return ready_3; });
      auto df4 = df3.Define("work_3_fin", "work_3_x + work_3_y + work_3_z");
      res_work_3 = df4.Sum<int>("work_3_fin").GetValue();
   };

   auto work_4 = [&]() {
      std::lock_guard lk_3{m_3};
      try {
         ROOT::RDataFrame df{1};
         auto df1 = df.Define("x", "rndm");
      } catch (...) {
      }
      ready_3 = true;
      cv_3.notify_all();
   };

   std::array<std::thread, 4> threads{std::thread{work_1}, std::thread{work_2}, std::thread{work_3},
                                      std::thread{work_4}};
   for (auto &&t : threads)
      t.join();

   EXPECT_EQ(res_work_2, 100);
   EXPECT_EQ(res_work_3, 66);
}
