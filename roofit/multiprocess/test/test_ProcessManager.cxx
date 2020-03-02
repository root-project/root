/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2019, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <RooFit/MultiProcess/ProcessManager.h>  // ... JobManager::process_manager()

#include "gtest/gtest.h"

TEST(TestMPProcessManager, birthAndDeath)
{
   // Note that this test does not use terminate() or check for sigterm_received,
   // it just destroys and exits itself on every process. This is not how it is
   // used in real life. The checkState test below more closely follows real life
   // usage.
   auto pm = RooFit::MultiProcess::ProcessManager(2);
   sleep(5);
}

TEST(TestMPProcessManager, checkState)
{
   std::size_t N_workers = 2;
   auto master_pid = getpid();
   auto pm = RooFit::MultiProcess::ProcessManager(N_workers);
   ASSERT_TRUE(pm.is_initialized());
   ASSERT_EQ(pm.N_workers(), N_workers);

   if (pm.is_master()) {
      EXPECT_EQ(getpid(), master_pid);
      EXPECT_FALSE(pm.is_queue());
      EXPECT_FALSE(pm.is_worker());
      pm.terminate();
   } else if (pm.is_queue()) {
      EXPECT_FALSE(pm.is_master());
      EXPECT_FALSE(pm.is_worker());
      while(!pm.sigterm_received()) {}
      std::_Exit(0);
   } else if (pm.is_worker()) {
      EXPECT_FALSE(pm.is_master());
      EXPECT_FALSE(pm.is_queue());
      EXPECT_TRUE(pm.worker_id() == 0 || pm.worker_id() == 1);
      while(!pm.sigterm_received()) {}
      std::_Exit(0);
   }
}