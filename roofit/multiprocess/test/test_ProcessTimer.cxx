/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, Nikhef, zefwolffs@gmail.com
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/ProcessTimer.h"
#include "RooFit/MultiProcess/ProcessManager.h" // ... JobManager::process_manager()

#include "gtest/gtest.h"

#include <chrono>
#include <thread>

using std::list;
namespace chrono = std::chrono; // alias

/// It's not easy to guarantee that this test always passes. It is not
/// deterministic because we time something across multiple processes and
/// sometimes some processes might be doing something else during the running
/// of these tests. That's why the test is disabled for now.
TEST(TestMPProcessTimer, DISABLED_Timings)
{
   RooFit::MultiProcess::ProcessManager pm(1);

   // Setup processtimers on processes
   if (pm.is_master())
      RooFit::MultiProcess::ProcessTimer::setup(999);
   if (!pm.is_master())
      RooFit::MultiProcess::ProcessTimer::setup(0);

   // Do some timed sleeping to emulate timing work being done
   RooFit::MultiProcess::ProcessTimer::start_timer("all:cumulative");
   std::this_thread::sleep_for(std::chrono::milliseconds(200));
   if (pm.is_master()) {
      // on master
      RooFit::MultiProcess::ProcessTimer::start_timer("master:wait_500");
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      RooFit::MultiProcess::ProcessTimer::end_timer("master:wait_500");
   }
   if (pm.is_worker()) {
      // on worker
      RooFit::MultiProcess::ProcessTimer::start_timer("worker:wait_1000");
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      RooFit::MultiProcess::ProcessTimer::end_timer("worker:wait_1000");
   }
   RooFit::MultiProcess::ProcessTimer::end_timer("all:cumulative");

   // Check if results make sense
   if (pm.is_master()) {
      list<chrono::time_point<chrono::steady_clock>> durations =
         RooFit::MultiProcess::ProcessTimer::get_durations("all:cumulative");
      auto it = durations.begin();
      long duration = chrono::duration_cast<chrono::milliseconds>(*std::next(it) - *it).count();
      EXPECT_NEAR(duration, 700, 100);

      durations = RooFit::MultiProcess::ProcessTimer::get_durations("master:wait_500");
      it = durations.begin();
      duration = chrono::duration_cast<chrono::milliseconds>(*std::next(it) - *it).count();
      EXPECT_NEAR(duration, 500, 100);
   }
   if (pm.is_worker()) {
      list<chrono::time_point<chrono::steady_clock>> durations =
         RooFit::MultiProcess::ProcessTimer::get_durations("all:cumulative");
      auto it = durations.begin();
      long duration = chrono::duration_cast<chrono::milliseconds>(*std::next(it) - *it).count();
      EXPECT_NEAR(duration, 1200, 100);

      durations = RooFit::MultiProcess::ProcessTimer::get_durations("worker:wait_1000");
      it = durations.begin();
      duration = chrono::duration_cast<chrono::milliseconds>(*std::next(it) - *it).count();
      EXPECT_NEAR(duration, 1000, 100);
   }
}
