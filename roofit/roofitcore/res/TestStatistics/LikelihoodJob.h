/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_TESTSTATISTICS_LikelihoodJob
#define ROOT_ROOFIT_TESTSTATISTICS_LikelihoodJob

#include "RooFit/MultiProcess/Job.h"
#include "RooFit/MultiProcess/types.h"
#include "RooFit/TestStatistics/LikelihoodWrapper.h"
#include "RooArgList.h"

#include "Math/MinimizerOptions.h"

#include <vector>

namespace RooFit {
namespace TestStatistics {

class LikelihoodJob : public MultiProcess::Job, public LikelihoodWrapper {
public:
   LikelihoodJob(std::shared_ptr<RooAbsL> _likelihood,
                 std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean /*, RooMinimizer *minimizer*/);
   LikelihoodJob *clone() const override;

   void init_vars();

   void evaluate() override;
   inline ROOT::Math::KahanSum<double> getResult() const override { return result_; }

   void updateWorkersParameters(); // helper for evaluate
   void updateWorkersOffsetting(); // helper for enableOffsetting

   // Job overrides:
   void evaluate_task(std::size_t task) override;
   void update_state() override;

   struct update_state_t {
      std::size_t var_index;
      double value;
      bool is_constant;
   };
   enum class update_state_mode : int { parameters, offsetting };

   // --- RESULT LOGISTICS ---
   struct task_result_t {
      std::size_t job_id; // job ID must always be the first part of any result message/type
      double value;
      double carry;
   };

   void send_back_task_result_from_worker(std::size_t task) override;
   bool receive_task_result_on_master(const zmq::message_t &message) override;

   void enableOffsetting(bool flag) override;

private:
   ROOT::Math::KahanSum<double> result_;
   std::vector<ROOT::Math::KahanSum<double>> results_;

   RooArgList vars_;      // Variables
   RooArgList save_vars_; // Copy of variables

   LikelihoodType likelihood_type_;
   std::size_t N_tasks_at_workers_ = 0;
};

std::ostream &operator<<(std::ostream &out, const LikelihoodJob::update_state_mode value);

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_LikelihoodJob
