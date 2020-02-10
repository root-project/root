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

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/ProcessManager.h"
#include "RooFit/MultiProcess/Job.h"  // complete Job object for JobManager::get_job_object()
#include "RooFit/MultiProcess/Queue.h"

namespace RooFit {
namespace MultiProcess {

// Have a worker ask for a task-message from the queue
bool Queue::pop(JobTask &job_task)
{
   if (_queue.empty()) {
      return false;
   } else {
      job_task = _queue.front();
      _queue.pop();
      return true;
   }
}

// Enqueue a task
void Queue::add(JobTask job_task)
{
   if (JobManager::instance()->process_manager().is_master()) {
// not necessary, the job adds tasks, so it is already activated there; TODO: remove commented out lines
//      if (!JobManager::instance()->is_activated()) {
//         JobManager::instance()->activate();
//      }
      JobManager::instance()->messenger().send_from_master_to_queue(M2Q::enqueue, job_task.first, job_task.second);
   } else if (JobManager::instance()->process_manager().is_queue()) {
      _queue.push(job_task);
   } else {
      throw std::logic_error("calling Communicator::to_master_queue from slave process");
   }
}


bool Queue::process_master_message(M2Q message)
{
   bool carry_on = true;

   switch (message) {
   case M2Q::terminate: {
      carry_on = false;
      break;
   }

   case M2Q::enqueue: {
      // enqueue task
//      mq_poller.ppoll(-1, ppoll_sigmask);
      auto job_object_id = JobManager::instance()->messenger().receive_from_master_on_queue<std::size_t>();
//      mq_poller.ppoll(-1, ppoll_sigmask);
      auto task = JobManager::instance()->messenger().receive_from_master_on_queue<Task>();
      JobTask job_task(job_object_id, task);
      add(job_task);
      N_tasks++;
      break;
   }

   case M2Q::retrieve: {
      // retrieve task results after queue is empty and all
      // tasks have been completed
      std::cout << "queue empty: " << _queue.empty() << std::endl;
      std::cout << "tasks completed: " << N_tasks_completed << std::endl;
      std::cout << "tasks: " << N_tasks << std::endl;
      if (_queue.empty() && N_tasks_completed == 0) {
         JobManager::instance()->messenger().send_from_queue_to_master(Q2M::retrieve_rejected); // handshake message: no tasks enqueued, premature retrieve!
      } else if (_queue.empty() && N_tasks_completed == N_tasks) {
         JobManager::instance()->results_from_queue_to_master();
         // reset number of received and completed tasks
         N_tasks = 0;
         N_tasks_completed = 0;
      } else {
         JobManager::instance()->messenger().send_from_queue_to_master(Q2M::retrieve_later); // handshake message: tasks not done yet, try again
      }
      break;
   }

   case M2Q::update_real: {
      auto get_time = []() {
         return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
      };
      auto t1 = get_time();
      auto job_id = JobManager::instance()->messenger().receive_from_master_on_queue<std::size_t>();
      auto ix = JobManager::instance()->messenger().receive_from_master_on_queue<std::size_t>();
      auto val = JobManager::instance()->messenger().receive_from_master_on_queue<double>();
      auto is_constant = JobManager::instance()->messenger().receive_from_master_on_queue<bool>();
      for (std::size_t worker_ix = 0; worker_ix < JobManager::instance()->process_manager().N_workers(); ++worker_ix) {
         JobManager::instance()->messenger().send_from_queue_to_worker(worker_ix, Q2W::update_real, job_id, ix, val, is_constant);
      }
      auto t2 = get_time();
      std::cout << "update_real on queue: " << (t2 - t1) / 1.e9 << "s (from " << t1 << " to " << t2 << "ns)" << std::endl;
      break;
   }
   }

   return carry_on;
}


void Queue::process_worker_message(std::size_t this_worker_id, W2Q message)
{
   switch (message) {
   case W2Q::dequeue: {
      // dequeue task
      JobTask job_task;
      auto queue_size_before_pop = _queue.size();
      bool popped = pop(job_task);
      std::cout << "in Queue::process_worker_message, W2Q::dequeue: popped = " << popped << ", queue size before pop = " << queue_size_before_pop << " and after pop = " << _queue.size() << std::endl;
      if (popped) {
         JobManager::instance()->messenger().send_from_queue_to_worker(this_worker_id, Q2W::dequeue_accepted, job_task.first, job_task.second);
      } else {
         JobManager::instance()->messenger().send_from_queue_to_worker(this_worker_id, Q2W::dequeue_rejected);
      }
      break;
   }

   case W2Q::send_result: {
      // receive back task result
      auto job_object_id = JobManager::instance()->messenger().receive_from_worker_on_queue<std::size_t>(this_worker_id);
      auto task = JobManager::instance()->messenger().receive_from_worker_on_queue<Task>(this_worker_id);
      JobManager::get_job_object(job_object_id)->receive_task_result_on_queue(task, this_worker_id);
      JobManager::instance()->messenger().send_from_queue_to_worker(this_worker_id, Q2W::result_received);
      N_tasks_completed++;
      break;
   }
   }
}


void Queue::loop()
{
   assert(JobManager::instance()->process_manager().is_queue());
   bool carry_on = true;
   ZeroMQPoller poller;
   std::size_t mq_index;
   std::tie(poller, mq_index) = JobManager::instance()->messenger().create_queue_poller();

   // Before blocking SIGTERM, set the signal handler, so we can also check after blocking whether a signal occurred
   // In our case, we already set it in the ProcessManager after forking to the queue and worker processes.

   sigset_t sigmask;
   sigemptyset(&sigmask);
   sigaddset(&sigmask, SIGTERM);
   sigprocmask(SIG_BLOCK, &sigmask, &JobManager::instance()->messenger().ppoll_sigmask);

   // Before doing anything, check whether we have received a terminate signal while blocking signals!
   // In this case, we also do that in the while condition.
   while (!ProcessManager::sigterm_received() && carry_on) {
      try { // watch for zmq_error from ppoll caused by SIGTERM from master
         // poll: wait until status change (-1: infinite timeout)
         auto poll_result = poller.ppoll(-1, &JobManager::instance()->messenger().ppoll_sigmask);
//         JobManager::instance()->messenger().N_available_polled_results = poll_result.size();
         // then process incoming messages from sockets
         for (auto readable_socket : poll_result) {
            // message comes from the master/queue socket (first element):
            if (readable_socket.first == mq_index) {
               auto message = JobManager::instance()->messenger().receive_from_master_on_queue<M2Q>();
               carry_on = process_master_message(message);
               // on terminate, also stop for-loop, no need to check other
               // sockets anymore:
               if (!carry_on) {
                  break;
               }
            } else { // from a worker socket
               // TODO: dangerous assumption for this_worker_id, may become invalid if we allow multiple queue_loops on the same process!
               auto this_worker_id = readable_socket.first - 1;  // TODO: replace with a more reliable lookup
               auto message = JobManager::instance()->messenger().receive_from_worker_on_queue<W2Q>(this_worker_id);
               process_worker_message(this_worker_id, message);
            }
         }
      } catch (zmq::error_t& e) {
         std::cout << "catching something on the queue" << std::endl;
         if ((e.num() == EINTR) && (ProcessManager::sigterm_received())) {
            std::cout << "queue out [MIC DROP]" << std::endl;
            break;
         } else if (e.num() == EAGAIN) {
            // This can happen from recv if ppoll initially gets a read-ready signal for a socket,
            // but the received data does not pass the checksum test, so the socket becomes unreadable
            // again or from non-blocking send if the socket becomes unwritable either due to the HWM
            // being reached or the socket not being connected (anymore). The latter case usually means
            // the connection has been severed from the other side, meaning it has probably been killed
            // and in that case the next ppoll call will probably also receive a SIGTERM, ending the
            // loop. In case something else is wrong, this message will print multiple times, which
            // should be taken as a cue for writing a bug report :)
            // TODO: handle this more rigorously
            std::cout << "EAGAIN in Queue::loop() (from either send or receive), continuing" << std::endl;
            continue;
         } else {
            throw;
         }
      }
   }

   // clean up signal management modifications
   sigprocmask(SIG_SETMASK, &JobManager::instance()->messenger().ppoll_sigmask, nullptr);
}

} // namespace MultiProcess
} // namespace RooFit
