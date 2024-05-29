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

TEST(RDFConcurrency, HistoNSparseDInMT)
{
   const int ncores = NUM_THREADS;
   const int nbins_per_axis = 10;
   const int nevents = 100000;

   ROOT::EnableImplicitMT(ncores);

   ROOT::RDataFrame df{nevents};

   auto col1 = df.Define("x0", [=](ULong64_t e) { return double(e % nbins_per_axis); }, {"rdfentry_"});
   auto col2 = col1.Define("x1", [=](ULong64_t e) { return double(e % nbins_per_axis); }, {"rdfentry_"});
   auto col3 = col2.Define("x2", [=](ULong64_t e) { return double(e % nbins_per_axis); }, {"rdfentry_"});
   auto col4 = col3.Define("x3", [=](ULong64_t e) { return double(e % nbins_per_axis); }, {"rdfentry_"});

   int nbins[4] = {nbins_per_axis, nbins_per_axis, nbins_per_axis, nbins_per_axis};
   double xmin[4] = {0., 0., 0., 0.};
   double xmax[4] = {nbins_per_axis, nbins_per_axis, nbins_per_axis, nbins_per_axis};
   auto hist = col4.HistoNSparseD<double, double, double, double>({"name", "title", 4, nbins, xmin, xmax},
                                                                  {"x0", "x1", "x2", "x3"});

   EXPECT_EQ(hist->GetEntries(), nevents);

   for (int i = 1; i <= nbins_per_axis; ++i) {
      std::vector<int> idx = {i, i, i, i};
      EXPECT_EQ(hist->GetBinContent(idx.data()), nevents / nbins_per_axis);
   }

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
   // Tests multiple concurrrent threads running RDataFrame computations with
   // JITting involved. The threads are synchronized to ensure the JIT compilations
   // are interleaved multiple times. Some of the threads will also throw exceptions
   // to test that they do not interfere with other valid RDataFrame graphs.
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
      try {
         ROOT::RDataFrame df{1};
         auto df1 = df.Define("work_1_x", "return 42;")
                       .Redefine("work_1_x",
                                 [](int x) {
                                    throw std::runtime_error("Error in RDF!");
                                    return x;
                                 },
                                 {"work_1_x"});
         df1.Sum<int>("work_1_x").GetValue();
      } catch (const std::runtime_error &e) {
         EXPECT_STREQ(e.what(), "Error in RDF!");
      }
      {
         std::lock_guard lk_1{m_1};
         ready_1 = true;
      }
      cv_1.notify_all();
   };

   auto work_2 = [&]() {
      ROOT::RDataFrame df{1};
      // work_2 starts only when work_1 is done
      {
         std::unique_lock lk_1(m_1);
         cv_1.wait(lk_1, [&ready_1] { return ready_1; });
      }
      auto df1 = df.Define("work_2_x", "42");
      // work_2 proceeds only when work_3 is done
      {
         std::unique_lock lk_3{m_3};
         cv_3.wait(lk_3, [&ready_3] { return ready_3; });
      }
      auto df2 = df1.Define("work_2_y", "58");
      auto df3 = df2.Define("work_2_z", "work_2_x + work_2_y");
      res_work_2 = df3.Sum<int>("work_2_z").GetValue();
      {
         std::lock_guard lk_2{m_2};
         ready_2 = true;
      }
      cv_2.notify_all();
   };

   auto work_3 = [&]() {
      try {
         ROOT::RDataFrame df{1};
         auto df1 = df.Define("x", "rndm");
      } catch (...) {
      }
      {
         std::lock_guard lk_3{m_3};
         ready_3 = true;
      }

      cv_3.notify_all();
   };

   auto work_sync = [&]() {
      // work_sync starts immediately
      ROOT::RDataFrame df{1};
      auto df1 = df.Define("work_sync_x", "11");

      // work_sync proceeds only when work_1 is done
      {
         std::unique_lock lk_1(m_1);
         cv_1.wait(lk_1, [&ready_1] { return ready_1; });
      }
      auto df2 = df1.Define("work_sync_y", "work_sync_x * 2");

      // work_sync proceeds only when work_2 is done
      {
         std::unique_lock lk_2(m_2);
         cv_2.wait(lk_2, [&ready_2] { return ready_2; });
      }
      auto df3 = df2.Define("work_sync_z", "work_sync_y + work_sync_x");

      // work_sync finishes only when work_3 is done
      {
         std::unique_lock lk_3(m_3);
         cv_3.wait(lk_3, [&ready_3] { return ready_3; });
      }
      auto df4 = df3.Define("work_sync_fin", "work_sync_x + work_sync_y + work_sync_z");
      res_work_3 = df4.Sum<int>("work_sync_fin").GetValue();
   };

   std::array<std::thread, 4> threads{std::thread{work_1}, std::thread{work_2}, std::thread{work_3},
                                      std::thread{work_sync}};
   for (auto &&t : threads)
      t.join();

   EXPECT_EQ(res_work_2, 100);
   EXPECT_EQ(res_work_3, 66);
}
