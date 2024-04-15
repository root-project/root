/*
* Project: RooFit
* Authors:
*   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
*
* Copyright (c) 2022, CERN
*
* Redistribution and use in source and binary forms,
* with or without modification, are permitted according to the terms
* listed in LICENSE (http://roofit.sourceforge.net/license.txt)
*/

#include "RooFit/MultiProcess/Config.h"

#include <algorithm>  // std::equal, std::rotate

#include "OrderTrackingJob.h"
#include "gtest/gtest.h"

TEST(FIFOQueue, TaskOrder)
{
   // one worker so we can easily check order, because of deterministic serial task execution on one worker
   RooFit::MultiProcess::Config::setDefaultNWorkers(1);

   std::size_t n_tasks = 10;
   OrderTrackingJob job(n_tasks, 0);

   std::vector<std::size_t> expected_order{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

   job.do_the_job();

   auto job_id = job.get_job_id();
   EXPECT_TRUE(std::equal(job.received_task_order[job_id].cbegin(),
                          job.received_task_order[job_id].cend(),
                          expected_order.cbegin()));
}

// Calculate the expected task execution order for a priority queue
//
// We would expect the first executed task to be the one with the highest
// priority, but it is sometimes (even often) rather one of the first
// submitted tasks instead. This is because the worker will immediately steal
// a task once one is on the queue. Often, the worker's first request comes
// before the highest priority one was added. So when comparing to the
// suggested order, we have to move this first executed task to the front.
//
// Things can go even worse in slow systems; there may be several tasks that
// get executed before the queue managed to receive all tasks. We loop
// through all of these, until we hit the last submitted task. From that point
// on, we demand adherence to the expected order.
//
// In the worst case, this could mean that the tasks are received as 0, 1,
// 2, 3, 4, 5, 6, 7, 8, 9. This is still expected behavior, given the
// asynchronous nature of the system, although, of course, quite
// undesirable and would indicate a clear need to improve performance.
std::vector<RooFit::MultiProcess::Task> expected_priority_queue_order(const std::vector<RooFit::MultiProcess::Task>& suggested_order,
                                                                      const std::vector<RooFit::MultiProcess::Task>& received_order,
                                                                      RooFit::MultiProcess::Task last_submitted_task)
{
   auto expected_order(suggested_order);
   auto expected_order_begin = expected_order.begin();
   for (auto& received_value : received_order) {
      auto first_executed_task_it = std::find(expected_order_begin, expected_order.end(), received_value);
      std::rotate(expected_order_begin, first_executed_task_it, first_executed_task_it + 1);
      if (received_value == last_submitted_task) {
         // we have found the last submitted task, from here on we expect the
         // order to be as suggested
         break;
      }
      ++expected_order_begin;
   }
   return expected_order;
}

TEST(PriorityQueue, TaskOrder)
{
   // one worker so we can easily check order, because of deterministic serial task execution on one worker
   RooFit::MultiProcess::Config::setDefaultNWorkers(1);

   EXPECT_TRUE(RooFit::MultiProcess::Config::Queue::setQueueType(RooFit::MultiProcess::Config::Queue::QueueType::Priority));

   std::size_t n_tasks = 10;
   OrderTrackingJob job(n_tasks, 100);

   std::vector<RooFit::MultiProcess::Task> suggested_order{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

   RooFit::MultiProcess::Config::Queue::suggestTaskOrder(job.get_job_id(), suggested_order);

   job.do_the_job();

   auto const& received = job.received_task_order.at(job.get_job_id());
   auto expected_order = expected_priority_queue_order(suggested_order, received, n_tasks - 1);
   EXPECT_TRUE(std::equal(received.cbegin(), received.cend(), expected_order.cbegin()));
   printf("%zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu\n", received[0], received[1], received[2], received[3], received[4], received[5], received[6], received[7], received[8], received[9]);
}

TEST(PriorityQueue, TaskPriority)
{
   // one worker so we can easily check order, because of deterministic serial task execution on one worker
   RooFit::MultiProcess::Config::setDefaultNWorkers(1);

   EXPECT_TRUE(RooFit::MultiProcess::Config::Queue::setQueueType(RooFit::MultiProcess::Config::Queue::QueueType::Priority));

   std::size_t n_tasks = 10;
   OrderTrackingJob job(n_tasks, 10000);

   std::vector<RooFit::MultiProcess::Task> suggested_order{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

   std::vector<std::size_t> priorities{0, 30, 480, 1290, 82103, 222222, 333333, 444444, 555555, 999999};

   RooFit::MultiProcess::Config::Queue::setTaskPriorities(job.get_job_id(), priorities);

   job.do_the_job();

   auto const& received = job.received_task_order.at(job.get_job_id());
   auto expected_order = expected_priority_queue_order(suggested_order, received, n_tasks - 1);
   EXPECT_TRUE(std::equal(received.cbegin(), received.cend(), expected_order.cbegin()));
   printf("%zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu, %zu\n", received[0], received[1], received[2], received[3], received[4], received[5], received[6], received[7], received[8], received[9]);
}

/// This test makes sure the program doesn't break when the user
/// forgets to set the priority for a Job.
TEST(PriorityQueue, ForgotToSetPriority)
{
   // one worker to keep it light, it's really irrelevant in this test
   RooFit::MultiProcess::Config::setDefaultNWorkers(1);

   EXPECT_TRUE(RooFit::MultiProcess::Config::Queue::setQueueType(RooFit::MultiProcess::Config::Queue::QueueType::Priority));

   std::size_t n_tasks = 3;
   NoopJob job(n_tasks);

   job.do_the_job();

   // It should just have run successfully at this point, no specific results to check.
   // This test exists because an empty task_priority_ for a job would cause a segfault
   // if we didn't handle it (as we didn't in some previous version).
}

/// Test changing queueType after a Queue has already been built; this should not be allowed.
TEST(QueueConfig, ChangeTypeAfterConstruction)
{
   // one worker to keep it light, it's really irrelevant in this test
   RooFit::MultiProcess::Config::setDefaultNWorkers(1);

   EXPECT_TRUE(RooFit::MultiProcess::Config::Queue::setQueueType(RooFit::MultiProcess::Config::Queue::QueueType::FIFO));

   std::size_t n_tasks = 1;
   NoopJob job(n_tasks);

   job.do_the_job();

   // set type illegally after JobManager construction
   printf("The following warning is expected:\n");
   EXPECT_FALSE(RooFit::MultiProcess::Config::Queue::setQueueType(RooFit::MultiProcess::Config::Queue::QueueType::Priority));
}
