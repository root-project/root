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

#include "RooFit/MultiProcess/worker.h"

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/types.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Job.h"
#include "RooFit/MultiProcess/util.h"
#include "RooFit/MultiProcess/ProcessTimer.h"
#include "RooFit/MultiProcess/Config.h"

#include <string>
#include <unistd.h> // getpid, pid_t
#include <cassert>
#include <cerrno>   // EINTR
#include <csignal>  // sigprocmask etc

namespace RooFit {
namespace MultiProcess {

static bool worker_loop_running = false;

bool is_worker_loop_running()
{
   return worker_loop_running;
}

/// \brief The worker processes' event loop
///
/// Asks the queue process for tasks, polls for incoming messages from other
/// processes and handles them.
void worker_loop()
{
   assert(JobManager::instance()->process_manager().is_worker());
   worker_loop_running = true;
   Q2W message_q2w;

   // use a flag to not ask twice
   bool dequeue_acknowledged = true;

   Poller poller;
   std::size_t mw_sub_index;

   std::tie(poller, mw_sub_index) = JobManager::instance()->messenger().create_worker_poller();

   // The SIGTERM handler was set in the ProcessManager after forking to the queue and worker
   // processes; it wakes up any poll through the self-pipe, so no signal blocking is needed here.
   while (!ProcessManager::sigterm_received()) {
      try { // watch for error from poll (which is called inside receive functions) caused by SIGTERM from master

         // try to dequeue a task
         if (dequeue_acknowledged) { // don't ask twice
            JobManager::instance()->messenger().send_from_worker_to_queue(W2Q::dequeue);
            dequeue_acknowledged = false;
         }

         // wait for handshake from queue or update from the master-worker channel
         auto poll_result = poller.poll(-1);
         // because the poller may now have a waiting update from master over the master-worker
         // channel, but the queue channel could be first in the poll_result vector, and during
         // handling of a new task it is possible we need to already receive the updated state,
         // we have to then flip this boolean so that in the for loop when we reach the
         // master-worker channel's result, we can skip it (otherwise we will hang there,
         // because no more updated state will be coming):
         bool skip_sub = false;
         // then process incoming messages from the channels
         for (auto readable_index : poll_result) {
            // message comes from the master-worker channel (first element):
            if (readable_index == mw_sub_index) {
               if (!skip_sub) {
                  auto job_id = JobManager::instance()->messenger().receive_from_master_on_worker<std::size_t>();
                  JobManager::get_job_object(job_id)->update_state();
               }
            } else { // from queue channel
               message_q2w = JobManager::instance()->messenger().receive_from_queue_on_worker<Q2W>();
               switch (message_q2w) {
               case Q2W::dequeue_rejected: {
                  dequeue_acknowledged = true;
                  break;
               }
               case Q2W::dequeue_accepted: {
                  dequeue_acknowledged = true;
                  auto job_id = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
                  auto state_id = JobManager::instance()->messenger().receive_from_queue_on_worker<State>();
                  auto task_id = JobManager::instance()->messenger().receive_from_queue_on_worker<Task>();

                  // while loop, because multiple jobs may have updated state coming
                  while (state_id != JobManager::get_job_object(job_id)->get_state_id()) {
                     skip_sub = true;
                     auto job_id_for_state =
                        JobManager::instance()->messenger().receive_from_master_on_worker<std::size_t>();
                     JobManager::get_job_object(job_id_for_state)->update_state();
                  }
                  if (RooFit::MultiProcess::Config::getTimingAnalysis()) ProcessTimer::start_timer("worker:eval_task:" + std::to_string(task_id));
                  JobManager::get_job_object(job_id)->evaluate_task(task_id);
                  if (RooFit::MultiProcess::Config::getTimingAnalysis()) ProcessTimer::end_timer("worker:eval_task:" + std::to_string(task_id));
                  JobManager::get_job_object(job_id)->send_back_task_result_from_worker(task_id);

                  break;
               }
               }
            }
         }

      } catch (ppoll_error_t &) {
         // SIGTERM received (benign signal interruptions are retried inside
         // Channel::wait), so exit the loop
         break;
      }
   }

   if (RooFit::MultiProcess::Config::getTimingAnalysis())
      ProcessTimer::write_file();

   worker_loop_running = false;
}

} // namespace MultiProcess
} // namespace RooFit
