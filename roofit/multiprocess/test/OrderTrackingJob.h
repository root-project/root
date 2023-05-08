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

#ifndef ROOT_ORDERTRACKINGJOB_H
#define ROOT_ORDERTRACKINGJOB_H

#include "RooFit/MultiProcess/types.h"
#include "NoopJob.h"
#include <vector>

class OrderTrackingJob : public NoopJob {
public:
   OrderTrackingJob(std::size_t n_tasks, unsigned sleep_first_task_us)
      : NoopJob(n_tasks), sleep_first_task_us_(sleep_first_task_us)
   {
   }

   bool receive_task_result_on_master(const zmq::message_t &message) override
   {
      auto result = message.data<task_result_t>();
      received_task_order[id_].push_back(result->task_id);
      --N_tasks_at_workers_;
      bool job_completed = (N_tasks_at_workers_ == 0);
      return job_completed;
   }

   void evaluate_task(std::size_t /*task*/) override
   {
      assert(get_manager()->process_manager().is_worker());
      if (first_task) {
         usleep(sleep_first_task_us_);
         first_task = false;
      }
   }

   std::size_t get_job_id() { return id_; }

   // key: job ID, value: order in which tasks were received
   std::unordered_map<std::size_t, std::vector<RooFit::MultiProcess::Task>> received_task_order;
   unsigned sleep_first_task_us_;
   bool first_task = true;
};
#endif // ROOT_ORDERTRACKINGJOB_H
