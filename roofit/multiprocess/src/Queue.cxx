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
#include "RooFit/MultiProcess/Job.h" // complete Job object for JobManager::get_job_object()
#include "RooFit/MultiProcess/util.h"

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

/// Have a worker ask for a task-message from the queue
///
/// \param[out] job_task JobTask reference to put the Job ID and the task index into.
/// \return true if a task was popped from the queue successfully, false if the queue was empty.
bool Queue::pop(JobTask &job_task)
{
   if (queue_.empty()) {
      return false;
   } else {
      job_task = queue_.front();
      queue_.pop();
      return true;
   }
}

/// Enqueue a task
///
/// \param[in] job_task JobTask object that contains the Job ID and the task index.
void Queue::add(JobTask job_task)
{
   if (JobManager::instance()->process_manager().is_master()) {
      JobManager::instance()->messenger().send_from_master_to_queue(M2Q::enqueue, job_task.job_id, job_task.state_id,
                                                                    job_task.task_id);
   } else if (JobManager::instance()->process_manager().is_queue()) {
      queue_.push(job_task);
   } else {
      throw std::logic_error("calling Communicator::to_master_queue from slave process");
   }
}

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
   while (!ProcessManager::sigterm_received()) {
      try { // watch for zmq_error from ppoll caused by SIGTERM from master
         // poll: wait until status change (-1: infinite timeout)
         auto poll_result = poller.ppoll(-1, &JobManager::instance()->messenger().ppoll_sigmask);
         // then process incoming messages from sockets
         for (auto readable_socket : poll_result) {
            // message comes from the master/queue socket (first element):
            if (readable_socket.first == mq_index) {
               auto message = JobManager::instance()->messenger().receive_from_master_on_queue<M2Q>();
               process_master_message(message);
            } else { // from a worker socket
               auto this_worker_id = readable_socket.first - 1;
               auto message = JobManager::instance()->messenger().receive_from_worker_on_queue<W2Q>(this_worker_id);
               process_worker_message(this_worker_id, message);
            }
         }
      } catch (ZMQ::ppoll_error_t &e) {
         zmq_ppoll_error_response response;
         try {
            response = handle_zmq_ppoll_error(e);
         } catch (std::logic_error &) {
            printf("queue loop got unhandleable ZMQ::ppoll_error_t\n");
            throw;
         }
         if (response == zmq_ppoll_error_response::abort) {
            break;
         } else if (response == zmq_ppoll_error_response::unknown_eintr) {
            printf("EINTR in queue loop but no SIGTERM received, continuing\n");
            continue;
         } else if (response == zmq_ppoll_error_response::retry) {
            printf("EAGAIN from ppoll in queue loop, continuing\n");
            continue;
         }
      } catch (zmq::error_t &e) {
         printf("unhandled zmq::error_t (not a ppoll_error_t) in queue loop with errno %d: %s\n", e.num(), e.what());
         throw;
      }
   }

   // clean up signal management modifications
   sigprocmask(SIG_SETMASK, &JobManager::instance()->messenger().ppoll_sigmask, nullptr);
}

} // namespace MultiProcess
} // namespace RooFit
