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
#include <cerrno> // EINTR
#include <chrono>
#include <csignal>  // sigprocmask etc

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/types.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Job.h"
#include "RooFit/MultiProcess/util.h"

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

   auto poller = JobManager::instance()->messenger().create_worker_poller();

   // Before blocking SIGTERM, set the signal handler, so we can also check after blocking whether a signal occurred
   // In our case, we already set it in the ProcessManager after forking to the queue and worker processes.

   sigset_t sigmask;
   sigemptyset(&sigmask);
   sigaddset(&sigmask, SIGTERM);
   sigprocmask(SIG_BLOCK, &sigmask, &JobManager::instance()->messenger().ppoll_sigmask);

   // Before doing anything, check whether we have received a terminate signal while blocking signals!
   // In this case, we also do that in the while condition.
   while (!ProcessManager::sigterm_received() && carry_on) {
      try { // watch for error from ppoll (which is called inside receive functions) caused by SIGTERM from master
         decltype(get_time()) t1 = 0, t2 = 0, t3 = 0;

         // try to dequeue a task
         if (dequeue_acknowledged) { // don't ask twice
            t1 = get_time();
            JobManager::instance()->messenger().send_from_worker_to_queue(W2Q::dequeue);
            dequeue_acknowledged = false;
         }

         // receive handshake
         message_q2w = JobManager::instance()->messenger().receive_from_queue_on_worker<Q2W>();

         switch (message_q2w) {
         case Q2W::terminate: {
            carry_on = false;
            break;
         }

         case Q2W::dequeue_rejected: {
            t2 = get_time();
//            std::cout
//               << "no work: worker " << JobManager::instance()->process_manager().worker_id() << " asked at " << t1
//               << " and got rejected at " << t2 << std::endl;

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
//            std::cout
//               << "task done: worker " << JobManager::instance()->process_manager().worker_id() << " asked at " << t1 << ", started at "
//               << t2 << " and finished at " << t3 << std::endl;

            JobManager::instance()->messenger().send_from_worker_to_queue(W2Q::send_result, job_id, task);
            JobManager::get_job_object(job_id)->send_back_task_result_from_worker(task);

//            std::cout << "task done: worker " << JobManager::instance()->process_manager().worker_id() << " sent back results" << std::endl;

            message_q2w = JobManager::instance()->messenger().receive_from_queue_on_worker<Q2W>();
            if (message_q2w != Q2W::result_received) {
               std::cerr << "worker " << getpid()
                         << " sent result, but did not receive Q2W::result_received handshake! Got " << message_q2w
                         << " instead." << std::endl;
               throw std::runtime_error("");
            }
//            std::cout << "task done: worker " << JobManager::instance()->process_manager().worker_id() << " finished dequeue_accepted" << std::endl;
            break;
         }

            // previously this was non-work mode

         case Q2W::update_real: {
            t1 = get_time();

            job_id = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
            std::size_t ix = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
            double val = JobManager::instance()->messenger().receive_from_queue_on_worker<double>();
            bool is_constant = JobManager::instance()->messenger().receive_from_queue_on_worker<bool>();
            JobManager::get_job_object(job_id)->update_real(ix, val, is_constant);

            t2 = get_time();
//            std::cout
//               << "update_real on worker " << JobManager::instance()->process_manager().worker_id() << ": " << (t2 - t1) / 1.e9
//               << "s (from " << t1 << " to " << t2 << "ns)" << std::endl;

            break;
         }

         case Q2W::update_bool: {
            job_id = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
            auto ix = JobManager::instance()->messenger().receive_from_queue_on_worker<std::size_t>();
            auto value = JobManager::instance()->messenger().receive_from_queue_on_worker<bool>();
            JobManager::get_job_object(job_id)->update_bool(ix, value);

            break;
         }

         case Q2W::result_received: {
            std::cerr << "In worker_loop: " << message_q2w
                      << " message received, but should only be received as handshake!" << std::endl;
            break;
         }
         }
      } catch (ZMQ::ppoll_error_t& e) {
         zmq_ppoll_error_response response;
         try {
            response = handle_zmq_ppoll_error(e);
         } catch (std::logic_error& e) {
            printf("worker loop at PID %d got unhandleable ZMQ::ppoll_error_t\n", getpid());
            throw;
         }
         if (response == zmq_ppoll_error_response::abort) {
            break;
         } else if (response == zmq_ppoll_error_response::unknown_eintr) {
            printf("EINTR in worker loop at PID %d but no SIGTERM received, continuing\n", getpid());
            continue;
         } else if (response == zmq_ppoll_error_response::retry) {
            printf("EAGAIN from ppoll in worker loop at PID %d, continuing\n", getpid());
            continue;
         }
      } catch (zmq::error_t& e) {
         printf("unhandled zmq::error_t (not a ppoll_error_t) in worker loop at PID %d with errno %d: %s\n", getpid(), e.num(), e.what());
         throw;
      }
   }
   // clean up signal management modifications
   sigprocmask(SIG_SETMASK, &JobManager::instance()->messenger().ppoll_sigmask, nullptr);

   worker_loop_running = false;
}

} // namespace MultiProcess
} // namespace RooFit
