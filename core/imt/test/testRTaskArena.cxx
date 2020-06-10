#include "TROOT.h"
#include "ROOT/RTaskArena.hxx"
#include "ROOT/TThreadExecutor.hxx"
#include <fstream>
#include <random>
#include <thread>
#include "gtest/gtest.h"
#include "tbb/task_arena.h"

#ifdef R__USE_IMT

const unsigned maxConcurrency = ROOT::Internal::LogicalCPUBandwithControl();
std::mt19937 randGenerator(0);                                      // seed the generator
std::uniform_int_distribution<> plausibleNCores(1, maxConcurrency); // define the range

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
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance1 = ROOT::Internal::GetGlobalTaskArena(nCores);
   auto gTAInstance2 = ROOT::Internal::GetGlobalTaskArena(plausibleNCores(randGenerator));
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
}

////////////////////////////////////////////////////////////////////////
// Integration Tests

TEST(RTaskArena, CorrectSizeIMT)
{
   auto gTAInstance1 = ROOT::Internal::GetGlobalTaskArena();
   ROOT::EnableImplicitMT(plausibleNCores(randGenerator));
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), maxConcurrency);
   ROOT::DisableImplicitMT();
}

TEST(RTaskArena, KeepSizeTThreadExecutor)
{
   const unsigned nCores = plausibleNCores(randGenerator);
   auto gTAInstance = ROOT::Internal::GetGlobalTaskArena(nCores);
   ROOT::TThreadExecutor threadExecutor(plausibleNCores(randGenerator));
   ASSERT_EQ(ROOT::Internal::RTaskArenaWrapper::TaskArenaSize(), nCores);
}

TEST(RTaskArena, InterleaveAndNest)
{
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

#endif
