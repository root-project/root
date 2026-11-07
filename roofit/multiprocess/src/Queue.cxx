/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/Queue.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/ProcessManager.h"
#include "RooFit/MultiProcess/util.h"

#include <cassert>

namespace RooFit {
namespace MultiProcess {

/** \class Queue
 * \brief Keeps a queue of tasks for workers and manages the queue process through its event loop
 *
 * The Queue maintains a set of tasks on the queue process by receiving them
 * from the master process. Worker processes can request to pop them off the
 * queue. The communication between these processes is handled inside
 * 'Queue::loop()', the queue process's event loop that polls the Messenger's
 * sockets for incoming messages and handles them when they come.
 *
 * The reason for this class is to get automatic load balancing between
 * workers. By allowing workers to request tasks whenever they are ready to
 * do work, we don't need to manually distribute work over workers and they
 * will always have something to do until all tasks have been completed.
 * The alternative simple strategy of just distributing all tasks evenly over
 * workers will be suboptimal when tasks have different or even varying
 * runtimes (this simple strategy could be implemented with a PUSH-PULL
 * ZeroMQ socket from master to workers, which would distribute tasks in a
 * round-robin fashion, which, indeed, does not do load balancing).
 */

/// Helper function for 'Queue::loop()'
void Queue::process_master_message(M2Q message)
{
   switch (message) {
   case M2Q::enqueue: {
      // enqueue task
      auto job_object_id = JobManager::instance()->messenger().receive_from_master_on_queue<std::size_t>();
      auto state_id = JobManager::instance()->messenger().receive_from_master_on_queue<State>();
      auto task_id = JobManager::instance()->messenger().receive_from_master_on_queue<Task>();
      JobTask job_task{job_object_id, state_id, task_id};
      add(job_task);
      N_tasks_++;
      break;
   }
   }
}

/// Helper function for 'Queue::loop()'
void Queue::process_worker_message(std::size_t this_worker_id, W2Q message)
{
   switch (message) {
   case W2Q::dequeue: {
      // dequeue task
      JobTask job_task;
      bool popped = pop(job_task);
      if (popped) {
         // Note: below two commands should be run atomically for thread safety (if that ever becomes an issue)
         JobManager::instance()->messenger().send_from_queue_to_worker(
            this_worker_id, Q2W::dequeue_accepted, job_task.job_id, job_task.state_id, job_task.task_id);
         ++N_tasks_at_workers_;
      } else {
         JobManager::instance()->messenger().send_from_queue_to_worker(this_worker_id, Q2W::dequeue_rejected);
      }
      break;
   }
   }
}

/// \brief The queue process's event loop
///
/// Polls for incoming messages from other processes and handles them.
void Queue::loop()
{
   assert(JobManager::instance()->process_manager().is_queue());
   Poller poller;
   std::size_t mq_index;
   std::tie(poller, mq_index) = JobManager::instance()->messenger().create_queue_poller();

   // The SIGTERM handler was set in the ProcessManager after forking to the queue and worker
   // processes; it wakes up any poll through the self-pipe, so no signal blocking is needed here.
   while (!ProcessManager::sigterm_received()) {
      try { // watch for poll interruption caused by SIGTERM from master
         // poll: wait until status change (-1: infinite timeout)
         auto poll_result = poller.poll(-1);
         // then process incoming messages from the channels
         for (auto readable_index : poll_result) {
            // message comes from the master/queue channel (first element):
            if (readable_index == mq_index) {
               auto message = JobManager::instance()->messenger().receive_from_master_on_queue<M2Q>();
               process_master_message(message);
            } else { // from a worker channel
               // by construction of the queue poller: the master-queue channel is
               // registered first (index 0), followed by the worker channels in
               // worker-ID order
               auto this_worker_id = readable_index - 1;
               auto message = JobManager::instance()->messenger().receive_from_worker_on_queue<W2Q>(this_worker_id);
               process_worker_message(this_worker_id, message);
            }
         }
      } catch (ppoll_error_t &) {
         // SIGTERM received (benign signal interruptions are retried inside
         // Channel::wait), so exit the loop
         break;
      }
   }
}

} // namespace MultiProcess
} // namespace RooFit
