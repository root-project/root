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
#include "RooFit/MultiProcess/Config.h"
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
   std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean)
   : LikelihoodWrapper(std::move(likelihood), std::move(calculation_is_clean)),
     n_event_tasks_(MultiProcess::Config::LikelihoodJob::defaultNEventTasks),
     n_component_tasks_(MultiProcess::Config::LikelihoodJob::defaultNComponentTasks)
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
            rvar->setVal(static_cast<double>(item.value));
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

/// \warning In automatic mode, this function can start MultiProcess (forks, starts workers, etc)!
std::size_t LikelihoodJob::getNEventTasks()
{
   std::size_t val = n_event_tasks_;
   if (val == MultiProcess::Config::LikelihoodJob::automaticNEventTasks) {
      val = get_manager()->process_manager().N_workers();
   }
   if (val > likelihood_->getNEvents()) {
      val = likelihood_->getNEvents();
   }
   return val;
}


std::size_t LikelihoodJob::getNComponentTasks()
{
   std::size_t val = n_component_tasks_;
   if (val == MultiProcess::Config::LikelihoodJob::automaticNComponentTasks) {
      val = 1;
   }
   if (val > likelihood_->getNComponents()) {
      val = likelihood_->getNComponents();
   }
   return val;
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
               double val = rar_val->getVal();
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
         get_manager()->messenger().publish_from_master_to_workers(id_, update_state_mode::parameters, state_id_, std::move(message));
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
      auto N_tasks = getNEventTasks() * getNComponentTasks();
      for (std::size_t ix = 0; ix < N_tasks; ++ix) {
         get_manager()->queue().add({id_, state_id_, ix});
      }
      n_tasks_at_workers_ = N_tasks;

      // wait for task results back from workers to master
      gather_worker_results();

      result_ = 0;
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
   --n_tasks_at_workers_;
   bool job_completed = (n_tasks_at_workers_ == 0);
   return job_completed;
}

// --- END OF RESULT LOGISTICS ---

void LikelihoodJob::evaluate_task(std::size_t task)
{
   assert(get_manager()->process_manager().is_worker());

   double section_first = 0;
   double section_last = 1;
   if (getNEventTasks() > 1) {
      std::size_t event_task = task % getNEventTasks();
      std::size_t N_events = likelihood_->numDataEntries();
      if (event_task > 0) {
         std::size_t first = N_events * event_task / getNEventTasks();
         section_first = static_cast<double>(first) / N_events;
      }
      if (event_task < getNEventTasks() - 1) {
         std::size_t last = N_events * (event_task + 1) / getNEventTasks();
         section_last = static_cast<double>(last) / N_events;
      }
   }

   switch (likelihood_type_) {
   case LikelihoodType::unbinned:
   case LikelihoodType::binned: {
      result_ = likelihood_->evaluatePartition({section_first, section_last}, 0, 0);
      break;
   }
   case LikelihoodType::sum: {
      std::size_t components_first = 0;
      std::size_t components_last = likelihood_->getNComponents();
      if (getNComponentTasks() > 1) {
         std::size_t component_task = task / getNEventTasks();
         components_first = likelihood_->getNComponents() * component_task / getNComponentTasks();
         if (component_task == getNComponentTasks() - 1) {
            components_last = likelihood_->getNComponents();
         } else {
            components_last = likelihood_->getNComponents() * (component_task + 1) / getNComponentTasks();
         }
      }
      result_ = likelihood_->evaluatePartition({section_first, section_last}, components_first, components_last);
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
