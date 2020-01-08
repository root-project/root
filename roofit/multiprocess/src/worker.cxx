/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2016-2019, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <unistd.h> // getpid, pid_t
#include <chrono>

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/types.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Job.h"

#include "RooFit/MultiProcess/worker.h"

namespace RooFit {
namespace MultiProcess {

static bool worker_loop_running = false;

bool is_worker_loop_running()
{
   return worker_loop_running;
}

void worker_loop()
{
   assert(JobManager::instance()->process_manager().is_worker());
   std::cout << "started worker_loop" << std::endl;
   worker_loop_running = true;
   bool carry_on = true;
   Task task;
   std::size_t job_id;
   Q2W message_q2w;

   // use a flag to not ask twice
   bool dequeue_acknowledged = true;

   auto get_time = []() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
         .count();
   };

   while (carry_on) {
      decltype(get_time()) t1 = 0, t2 = 0, t3 = 0;
      std::cout << "worker_loop next iteration" << std::endl;

      // try to dequeue a task
      if (dequeue_acknowledged) { // don't ask twice
         t1 = get_time();
         JobManager::instance()->messenger().send_from_worker_to_queue(W2Q::dequeue);
         std::cout << "worker_loop sent dequeue" << std::endl;
         dequeue_acknowledged = false;
      }

      // receive handshake
      message_q2w = JobManager::instance()->messenger().receive_from_queue_on_worker<Q2W>();
      std::cout << "worker_loop received handshake" << std::endl;

      switch (message_q2w) {
      case Q2W::terminate: {
         carry_on = false;
         break;
      }

      case Q2W::dequeue_rejected: {
         t2 = get_time();
         std::cout
            << "no work: worker " << JobManager::instance()->process_manager().worker_id() << " asked at " << t1
            << " and got rejected at " << t2 << std::endl;

         dequeue_acknowledged = true;
         break;
      }
      case Q2W::dequeue_accepted: {
         dequeue_acknowledged = true;
         job_id = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
         task = JobManager::instance()->messenger().receive_from_queue_on_worker<Task>();

         t2 = get_time();
         JobManager::get_job_object(job_id)->evaluate_task(task);

         t3 = get_time();
         std::cout
            << "task done: worker " << JobManager::instance()->process_manager().worker_id() << " asked at " << t1 << ", started at "
            << t2 << " and finished at " << t3 << std::endl;

         JobManager::instance()->messenger().send_from_worker_to_queue(W2Q::send_result, job_id, task);
         JobManager::get_job_object(job_id)->send_back_task_result_from_worker(task);

         message_q2w = JobManager::instance()->messenger().receive_from_queue_on_worker<Q2W>();
         if (message_q2w != Q2W::result_received) {
            std::cerr << "worker " << getpid()
                      << " sent result, but did not receive Q2W::result_received handshake! Got " << message_q2w
                      << " instead." << std::endl;
            throw std::runtime_error("");
         }
         break;
      }

         // previously this was non-work mode

      case Q2W::update_real: {
         auto t1 = get_time();

         job_id = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
         std::size_t ix = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
         double val = JobManager::instance()->messenger().receive_from_queue_on_worker<double>();
         bool is_constant = JobManager::instance()->messenger().receive_from_queue_on_worker<bool>();
         JobManager::get_job_object(job_id)->update_real(ix, val, is_constant);

         auto t2 = get_time();
         std::cout
            << "update_real on worker " << JobManager::instance()->process_manager().worker_id() << ": " << (t2 - t1) / 1.e9
            << "s (from " << t1 << " to " << t2 << "ns)" << std::endl;

         break;
      }

      case Q2W::result_received: {
         std::cerr << "In worker_loop: " << message_q2w
                   << " message received, but should only be received as handshake!" << std::endl;
         break;
      }
      }
   }
   worker_loop_running = false;
}

} // namespace MultiProcess
} // namespace RooFit
