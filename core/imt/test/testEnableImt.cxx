#include "TROOT.h"
#include "ROOT/RTaskArena.hxx"

#include "tbb/task_arena.h"

#include "ROOT/TestSupport.hxx"
#include "gtest/gtest.h"

#ifdef R__USE_IMT

static const unsigned gMaxConcurrency = ROOT::Internal::LogicalCPUBandwidthControl();

TEST(EnableImt, TBBAttach)
{
   tbb::task_arena main_arena{2};

   main_arena.execute([&]() { ROOT::EnableImplicitMT(ROOT::EIMTConfig::kExistingTBBArena); });

   auto psize = ROOT::GetThreadPoolSize();

   EXPECT_TRUE(psize > 1);
   EXPECT_EQ(main_arena.max_concurrency(), 2);
   EXPECT_EQ(psize, 2);
}

#endif
