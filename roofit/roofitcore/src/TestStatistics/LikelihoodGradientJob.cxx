/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include <TestStatistics/LikelihoodGradientJob.h>

namespace RooFit {
namespace TestStatistics {

/*
void LikelihoodGradientJob::CalculateAll(const double *x) {
//   auto get_time = [](){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();};
//   decltype(get_time()) t1, t2;

   if (get_manager()->is_master()) {
      // do Grad, G2 and Gstep here and then just return results from the
      // separate functions below
      bool was_not_synced = sync_parameters(x);
      if (was_not_synced) {
         // update parameters and object states that changed since last calculation (or creation if first time)
//         t1 = get_time();
         update_state();
//         t2 = get_time();

//         RooWallTimer timer;

         // master fills queue with tasks
         for (std::size_t ix = 0; ix < N_tasks; ++ix) {
            JobTask job_task(id, ix);
            get_manager()->to_queue(job_task);
         }
         waiting_for_queued_tasks = true;

         // wait for task results back from workers to master (put into _grad)
         gather_worker_results();

//         timer.stop();

//         oocxcoutD((TObject*)nullptr,Benchmarking1) << "update_state: " << (t2 - t1)/1.e9 << "s (from " << t1 << " to " << t2 << "ns), gradient work: " << timer.timing_s() << "s" << std::endl;
      }
   }
}
*/

void LikelihoodGradientJob::fill_gradient(const double *x, double *grad) {

}

void LikelihoodGradientJob::fill_second_derivative(const double *x, double *g2) {}

void LikelihoodGradientJob::fill_step_size(const double *x, double *gstep) {}

void LikelihoodGradientJob::synchronize_parameter_settings(const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) {
   _gradf.SetInitialGradient(parameter_settings);
}

} // namespace TestStatistics
} // namespace RooFit
