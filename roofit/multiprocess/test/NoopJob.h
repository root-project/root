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

#ifndef ROOT_NOOPJOB_H
#define ROOT_NOOPJOB_H

#include "RooFit/MultiProcess/Job.h"
// needed to complete type returned from...
#include "RooFit/MultiProcess/JobManager.h"     // ... Job::get_manager()
#include "RooFit/MultiProcess/ProcessManager.h" // ... JobManager::process_manager()
#include "RooFit/MultiProcess/Queue.h"          // ... JobManager::queue()


class NoopJob : public RooFit::MultiProcess::Job {
public:
   explicit NoopJob(std::size_t n_tasks) : n_tasks_(n_tasks) {}

   void do_the_job()
   {
      if (!get_manager()->process_manager().is_master()) return;

      // master fills queue with tasks
      for (std::size_t task_id = 0; task_id < n_tasks_; ++task_id) {
         RooFit::MultiProcess::JobTask job_task{id_, state_id_, task_id};
         get_manager()->queue()->add(job_task);
         ++N_tasks_at_workers_;
      }

      // wait for task results back from workers to master
      gather_worker_results();
   }

   // -- BEGIN plumbing --
   struct task_result_t {
      std::size_t job_id; // job ID must always be the first part of any result message/type
      std::size_t task_id;
   };

   void send_back_task_result_from_worker(std::size_t task) override
   {
      task_result_t task_result{id_, task};
      zmq::message_t message(sizeof(task_result_t));
      memcpy(message.data(), &task_result, sizeof(task_result_t));
      get_manager()->messenger().send_from_worker_to_master(std::move(message));
   }

   bool receive_task_result_on_master(const zmq::message_t &message) override
   {
      /*auto result =*/ message.data<task_result_t>();
      --N_tasks_at_workers_;
      bool job_completed = (N_tasks_at_workers_ == 0);
      return job_completed;
   }
   // -- END plumbing --

private:
   void evaluate_task(std::size_t /*task*/) override
   {
      assert(get_manager()->process_manager().is_worker());
   }

protected:
   std::size_t n_tasks_ = 1;
   std::size_t N_tasks_at_workers_ = 0;
};

#endif // ROOT_NOOPJOB_H
