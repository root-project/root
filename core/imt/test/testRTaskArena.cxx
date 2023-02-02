#include "TROOT.h"
#include "ROOT/RTaskArena.hxx"
#include "ROOT/TThreadExecutor.hxx"
#include "../src/ROpaqueTaskArena.hxx"

#include "ROOT/TestSupport.hxx"

#include <fstream>
#include <random>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "gtest/gtest.h"

#ifdef R__USE_IMT

const unsigned maxConcurrency = ROOT::Internal::LogicalCPUBandwithControl();
std::mt19937 randGenerator(0);                                      // seed the generator
std::uniform_int_distribution<> plausibleNCores(1, maxConcurrency); // define the range

/// Suppress the task arena diagnostics for tests where we try to create the task arena multiple times.
#define SUPPRESS_DIAG \
   ROOT::TestSupport::CheckDiagsRAII raii; \
   raii.optionalDiag(kWarning, "RTaskArenaWrapper", "There's already an active task arena", false);

TEST(RTaskArena, Size0WhenNoInstance)
{
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), 0u);
}

TEST(RTaskArena, Construction)
{
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
}

TEST(RTaskArena, DefaultConstruction)
{
   auto gTAInstance = ROOT::Internal::GetGlobalTaskArena();
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);
}

TEST(RTaskArena, Reconstruction)
{
   unsigned nCores;
   {
      nCores = plausibleNCores(randGenerator);
      auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);
      EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
   }

   EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), 0u);

   nCores = plausibleNCores(randGenerator);
   auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
}

TEST(RTaskArena, SingleInstance)
{
   SUPPRESS_DIAG;
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance1 = ROOT::Internal::GetGlobalTaskArena(nCores);
   auto gTAInstance2 = ROOT::Internal::GetGlobalTaskArena(plausibleNCores(randGenerator));
   ASSERT_EQ(&(*gTAInstance1), &(*gTAInstance2));
}

TEST(RTaskArena, AccessWorkingTBBtaskArena)
{
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);
   auto tbbTACores = gTAInstance->Access().max_concurrency();
   ASSERT_EQ(nCores, tbbTACores);
}

TEST(RTaskArena, KeepSize)
{
   SUPPRESS_DIAG
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance1 = ROOT::Internal::GetGlobalTaskArena(nCores);
   auto gTAInstance2 = ROOT::Internal::GetGlobalTaskArena(plausibleNCores(randGenerator));
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
}

////////////////////////////////////////////////////////////////////////
// Integration Tests

TEST(RTaskArena, CorrectSizeIMT)
{
   SUPPRESS_DIAG
   auto gTAInstance1 = ROOT::Internal::GetGlobalTaskArena();
   ROOT::EnableImplicitMT(plausibleNCores(randGenerator));
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);
   ROOT::DisableImplicitMT();
}

TEST(RTaskArena, KeepSizeTThreadExecutor)
{
   SUPPRESS_DIAG
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);
   ROOT::TThreadExecutor threadExecutor(plausibleNCores(randGenerator));
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
}

TEST(RTaskArena, InterleaveAndNest)
{
   SUPPRESS_DIAG
   unsigned nCores;

   // IMT + GTA
   {
      ROOT::EnableImplicitMT();
      nCores = plausibleNCores(randGenerator);
      auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);

      EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);

      ROOT::DisableImplicitMT();
   }
   EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), 0u);

   // IMT + TThreadExecutor
   {
      ROOT::EnableImplicitMT();
      ROOT::TThreadExecutor threadExecutor(plausibleNCores(randGenerator));

      EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);

      ROOT::DisableImplicitMT();
   }
   EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), 0u);

   // TThreadExecutor + IMT
   {
      ROOT::TThreadExecutor threadExecutor{};
      ROOT::EnableImplicitMT(plausibleNCores(randGenerator));

      EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);

      ROOT::DisableImplicitMT();
   }
   EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), 0u);

   // Nested TThreadExecutor
   {
      ROOT::TThreadExecutor threadExecutor{};
      auto fcn = []() {
         ROOT::TThreadExecutor te(plausibleNCores(randGenerator));
         EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);
      };
      threadExecutor.Foreach(fcn, 2);
      EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);
   }
   EXPECT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), 0u);
}


// Acquire pointers to ROOT's task arena from many threads in parallel.
// To create more chaos, half of the threads will immediately try to get the pointer,
// while the other half waits for a condition variable.
// To test destroying and recreating a task arena, the first half of the shared
// pointers will be destroyed before the other threads are released.
// Then, using a notify_all(), the other half of threads will wake up
// and race to create a task arena again.
TEST(RTaskArena, ThreadSafety) {
   constexpr unsigned int nThreads = 200;
   using TaskArenaPtr_t = std::shared_ptr<ROOT::Internal::RTaskArenaWrapper>;
   bool firstHalfDone = false;
   std::condition_variable cv;
   std::mutex m;

   std::vector<TaskArenaPtr_t> arenas1(nThreads/2);
   std::vector<TaskArenaPtr_t> arenas2(nThreads/2);
   std::vector<std::thread> threads;
   for (unsigned int i=0; i < nThreads; ++i) {
      threads.emplace_back([&,i]{
         if (i < nThreads/2) {
            arenas1.at(i) = ROOT::Internal::GetGlobalTaskArena();
         } else {
            bool canContinue = false;
            do {
               std::unique_lock<std::mutex> lock(m);
               canContinue = cv.wait_for(lock, std::chrono::microseconds(10), [=](){return firstHalfDone;});
            } while (!canContinue);
            arenas2.at(i % arenas2.size()) = ROOT::Internal::GetGlobalTaskArena();
         }
      });
   }

   std::for_each(threads.begin(), threads.end()-nThreads/2, [](std::thread& thr){thr.join();});
   const ROOT::Internal::RTaskArenaWrapper* const ptrFirstHalf = arenas1.front().get();
   EXPECT_TRUE(std::all_of(arenas1.begin(), arenas1.end(), [ptrFirstHalf](const TaskArenaPtr_t& ptr){
      return ptr.get() == ptrFirstHalf;
   }));

   // Destroys ROOT's thread pool
   arenas1.clear();
   {
      std::unique_lock<std::mutex> lock(m);
      firstHalfDone = true;
      cv.notify_all();
   }

   std::for_each(threads.begin()+nThreads/2, threads.end(), [](std::thread& thr){thr.join();});

   const ROOT::Internal::RTaskArenaWrapper* const ptrSecondHalf = arenas2.front().get();

   EXPECT_TRUE(std::all_of(arenas2.begin(), arenas2.end(), [=](const TaskArenaPtr_t& ptr){
      return ptr.get() == ptrSecondHalf;
   }));
}


// Have many threads create TThreadExecutor instances in parallel, which
// each increment atomic counters in parallel.
// Extra chaos is created by having all threads yield() for a while.
// When all threads are done, check that the atomic counters
// have been incremented the desired number of times.
TEST(TThreadExecutor, ThreadSafety) {
   constexpr unsigned int nThreads = 200;
   std::vector<std::thread> threads;
   std::vector<std::atomic<int>> counters(nThreads);

   for (unsigned int i=0; i < nThreads; ++i) {
      threads.emplace_back([i,&counters](){
         ROOT::TThreadExecutor ttex;
         ttex.Foreach([i,&counters](){
            const auto start = std::chrono::high_resolution_clock::now();
            const auto end = start + std::chrono::microseconds(100);
            do {
               std::this_thread::yield();
            } while (std::chrono::high_resolution_clock::now() < end);
            counters[i] += 1;
         }, i);
      });
   }

   std::vector<int> target(nThreads);
   std::iota(target.begin(), target.end(), 0);

   std::for_each(threads.begin(), threads.end(), [](std::thread& thr){thr.join();});

   EXPECT_TRUE(std::equal(counters.begin(), counters.end(), target.begin()));
}

// Checking if we correctly handle uneven chunks
TEST(TThreadExecutor, StdVectorChunks)
{
   ROOT::TThreadExecutor ttex;
   auto func = [](int x) -> int { return x; };
   // redfunc must be such that does not have 0 as identity (i.e. not addition but multiplication)
   auto redfunc = [](const std::vector<int> &v) {
      return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
   };

   // will be calculating 7 factorial = 5040, const and non-const vectors to invoke different overloads
   std::vector<int> vec{1, 2, 3, 4, 5, 6, 7};
   const std::vector<int> cvec{1, 2, 3, 4, 5, 6, 7};

   EXPECT_EQ(ttex.MapReduce(func, vec, redfunc, 3), 5040); // with 3 chunks, last chunk is smaller
   EXPECT_EQ(ttex.MapReduce(func, cvec, redfunc, 3), 5040);

   EXPECT_EQ(ttex.MapReduce(func, vec, redfunc, 9), 5040); // with 9 chunks, 2 empty chunks
   EXPECT_EQ(ttex.MapReduce(func, cvec, redfunc, 9), 5040);
}

TEST(TThreadExecutor, TSeqActions)
{
   ROOT::TThreadExecutor ttex;
   auto func = [](int x) -> int { return x; };
   auto redfunc = [](const std::vector<int> &v) { return std::accumulate(v.begin(), v.end(), 0); };

   // MapReduce on TSeq with end specified only
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(5), redfunc, 3), 10); // with 3 chunks
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(5), redfunc), 10);    // with 0 chunks

   // MapReduce on TSeq with begin and end specified only
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(2, 5), redfunc, 3), 9);
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(2, 5), redfunc), 9);

   // MapReduce on increasing and decreasing TSeq with begin, end and step specified
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(2, 5, 2), redfunc, 3), 6);
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(2, 5, 2), redfunc), 6);
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(5, 2, -2), redfunc, 3), 8);
   EXPECT_EQ(ttex.MapReduce(func, ROOT::TSeqI(5, 2, -2), redfunc), 8);

   // Map on TSeq with end specified only
   EXPECT_EQ(redfunc(ttex.Map(func, ROOT::TSeqI(5))), 10);

   // Map on TSeq with begin and end specified only
   EXPECT_EQ(redfunc(ttex.Map(func, ROOT::TSeqI(2, 5))), 9);

   // Map on increasing and decreasing TSeq with begin, end and step specified
   EXPECT_EQ(redfunc(ttex.Map(func, ROOT::TSeqI(2, 5, 2))), 6);
   EXPECT_EQ(redfunc(ttex.Map(func, ROOT::TSeqI(5, 2, -2))), 8);
}

#endif
