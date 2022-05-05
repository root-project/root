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

#include "LikelihoodJob.h"

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/ProcessManager.h"
#include "RooFit/MultiProcess/Queue.h"
#include "RooFit/MultiProcess/Job.h"
#include "RooFit/MultiProcess/types.h"
#include "RooFit/TestStatistics/RooAbsL.h"
#include "RooFit/TestStatistics/RooUnbinnedL.h"
#include "RooFit/TestStatistics/RooBinnedL.h"
#include "RooFit/TestStatistics/RooSubsidiaryL.h"
#include "RooFit/TestStatistics/RooSumL.h"
#include "RooRealVar.h"

namespace RooFit {
namespace TestStatistics {

LikelihoodJob::LikelihoodJob(
   std::shared_ptr<RooAbsL> likelihood,
   std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean /*, RooMinimizer *minimizer*/)
   : LikelihoodWrapper(std::move(likelihood), std::move(calculation_is_clean) /*, minimizer*/)
{
   init_vars();
   // determine likelihood type
   if (dynamic_cast<RooUnbinnedL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::unbinned;
   } else if (dynamic_cast<RooBinnedL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::binned;
   } else if (dynamic_cast<RooSumL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::sum;
   } else if (dynamic_cast<RooSubsidiaryL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::subsidiary;
   } else {
      throw std::logic_error("in LikelihoodJob constructor: likelihood is not of a valid subclass!");
   }
   // Note to future maintainers: take care when storing the minimizer_fcn pointer. The
   // RooAbsMinimizerFcn subclasses may get cloned inside MINUIT, which means the pointer
   // should also somehow be updated in this class.
}

LikelihoodJob *LikelihoodJob::clone() const
{
   return new LikelihoodJob(*this);
}

// This is a separate function (instead of just in ctor) for historical reasons.
// Its predecessor RooRealMPFE::initVars() was used from multiple ctors, but also
// from RooRealMPFE::constOptimizeTestStatistic at the end, which makes sense,
// because it might change the set of variables. We may at some point want to do
// this here as well.
void LikelihoodJob::init_vars()
{
   // Empty current lists
   vars_.removeAll();
   save_vars_.removeAll();

   // Retrieve non-constant parameters
   auto vars = std::make_unique<RooArgSet>(
      *likelihood_->getParameters()); // TODO: make sure this is the right list of parameters, compare to original
                                      // implementation in RooRealMPFE.cxx
   RooArgList varList(*vars);

   // Save in lists
   vars_.add(varList);
   save_vars_.addClone(varList);
}

void LikelihoodJob::update_state()
{
   if (get_manager()->process_manager().is_worker()) {
      auto mode = get_manager()->messenger().receive_from_master_on_worker<update_state_mode>();
      switch (mode) {
      case update_state_mode::parameters: {
         state_id_ = get_manager()->messenger().receive_from_master_on_worker<RooFit::MultiProcess::State>();
         auto message = get_manager()->messenger().receive_from_master_on_worker<zmq::message_t>();
         auto message_begin = message.data<update_state_t>();
         auto message_end = message_begin + message.size() / sizeof(update_state_t);
         std::vector<update_state_t> to_update(message_begin, message_end);
         for (auto const &item : to_update) {
            RooRealVar *rvar = (RooRealVar *)vars_.at(item.var_index);
            rvar->setVal(static_cast<Double_t>(item.value));
            if (rvar->isConstant() != item.is_constant) {
               rvar->setConstant(static_cast<bool>(item.is_constant));
            }
         }
         break;
      }
      case update_state_mode::offsetting: {
         LikelihoodWrapper::enableOffsetting(get_manager()->messenger().receive_from_master_on_worker<bool>());
         break;
      }
      }
   }
}

void LikelihoodJob::updateWorkersParameters()
{
   if (get_manager()->process_manager().is_master()) {
      bool valChanged = false;
      bool constChanged = false;
      std::vector<update_state_t> to_update;
      for (std::size_t ix = 0u; ix < static_cast<std::size_t>(vars_.getSize()); ++ix) {
         valChanged = !vars_[ix].isIdentical(save_vars_[ix], true);
         constChanged = (vars_[ix].isConstant() != save_vars_[ix].isConstant());

         if (valChanged || constChanged) {
            if (constChanged) {
               ((RooRealVar *)&save_vars_[ix])->setConstant(vars_[ix].isConstant());
            }
            // TODO: Check with Wouter why he uses copyCache in MPFE; makes it very difficult to extend, because
            // copyCache is protected (so must be friend). Moved setting value to if-block below.
            //          _saveVars[ix].copyCache(&_vars[ix]);

            // send message to queue (which will relay to workers)
            RooAbsReal *rar_val = dynamic_cast<RooAbsReal *>(&vars_[ix]);
            if (rar_val) {
               Double_t val = rar_val->getVal();
               dynamic_cast<RooRealVar *>(&save_vars_[ix])->setVal(val);
               bool isC = vars_[ix].isConstant();
               to_update.push_back(update_state_t{ix, val, isC});
            }
         }
      }
      if (!to_update.empty()) {
         ++state_id_;
         zmq::message_t message(to_update.begin(), to_update.end());
         // always send Job id first! This is used in worker_loop to route the
         // update_state call to the correct Job.
         get_manager()->messenger().publish_from_master_to_workers(id_, update_state_mode::parameters, state_id_);
         // have to pass message separately to avoid copies from parameter pack to first parameter
         get_manager()->messenger().publish_from_master_to_workers(std::move(message));
      }
   }
}

void LikelihoodJob::updateWorkersOffsetting()
{
   get_manager()->messenger().publish_from_master_to_workers(id_, update_state_mode::offsetting, isOffsetting());
}

void LikelihoodJob::evaluate()
{
   if (get_manager()->process_manager().is_master()) {
      // update parameters that changed since last calculation (or creation if first time)
      updateWorkersParameters();

      // master fills queue with tasks
      for (std::size_t ix = 0; ix < get_manager()->process_manager().N_workers(); ++ix) {
         get_manager()->queue().add({id_, state_id_, ix});
      }
      N_tasks_at_workers_ = get_manager()->process_manager().N_workers();

      // wait for task results back from workers to master
      gather_worker_results();

      for (auto const &item : results_) {
         result_ += item;
      }
      result_ = applyOffsetting(result_);
      results_.clear();
   }
}

// --- RESULT LOGISTICS ---

void LikelihoodJob::send_back_task_result_from_worker(std::size_t /*task*/)
{
   task_result_t task_result{id_, result_.Result(), result_.Carry()};
   zmq::message_t message(sizeof(task_result_t));
   memcpy(message.data(), &task_result, sizeof(task_result_t));
   get_manager()->messenger().send_from_worker_to_master(std::move(message));
}

bool LikelihoodJob::receive_task_result_on_master(const zmq::message_t &message)
{
   auto task_result = message.data<task_result_t>();
   results_.emplace_back(task_result->value, task_result->carry);
   printf("result received: %f\n", task_result->value);
   --N_tasks_at_workers_;
   bool job_completed = (N_tasks_at_workers_ == 0);
   return job_completed;
}

// --- END OF RESULT LOGISTICS ---

void LikelihoodJob::evaluate_task(std::size_t task)
{
   assert(get_manager()->process_manager().is_worker());

   std::size_t N_events = likelihood_->numDataEntries();

   // used to have multiple modes here, but only kept "bulk" mode; dropped interleaved, single_event and all_events from
   // old MultiProcess::NLLVar
   std::size_t first = N_events * task / get_manager()->process_manager().N_workers();
   std::size_t last = N_events * (task + 1) / get_manager()->process_manager().N_workers();

   switch (likelihood_type_) {
   case LikelihoodType::unbinned:
   case LikelihoodType::binned: {
      result_ = likelihood_->evaluatePartition(
         {static_cast<double>(first) / N_events, static_cast<double>(last) / N_events}, 0, 0);
      break;
   }
   default: {
      throw std::logic_error(
         "in LikelihoodJob::evaluate_task: likelihood types other than binned and unbinned not yet implemented!");
      break;
   }
   }
}

void LikelihoodJob::enableOffsetting(bool flag)
{
   LikelihoodWrapper::enableOffsetting(flag);
   updateWorkersOffsetting();
}

#define PROCESS_VAL(p) \
   case (p): s = #p; break;

std::ostream &operator<<(std::ostream &out, const LikelihoodJob::update_state_mode value)
{
   std::string s;
   switch (value) {
      PROCESS_VAL(LikelihoodJob::update_state_mode::offsetting);
      PROCESS_VAL(LikelihoodJob::update_state_mode::parameters);
   default: s = std::to_string(static_cast<int>(value));
   }
   return out << s;
}

#undef PROCESS_VAL

} // namespace TestStatistics
} // namespace RooFit
