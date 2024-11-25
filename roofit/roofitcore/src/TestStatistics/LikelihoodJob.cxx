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

#include "LikelihoodSerial.h"
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
#include "RooNaNPacker.h"

#include "TMath.h" // IsNaN

namespace RooFit {
namespace TestStatistics {

LikelihoodJob::LikelihoodJob(std::shared_ptr<RooAbsL> likelihood,
                             std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean, SharedOffset offset)
   : LikelihoodWrapper(std::move(likelihood), std::move(calculation_is_clean), std::move(offset)),
     n_event_tasks_(MultiProcess::Config::LikelihoodJob::defaultNEventTasks),
     n_component_tasks_(MultiProcess::Config::LikelihoodJob::defaultNComponentTasks),
     likelihood_serial_(likelihood_, calculation_is_clean_, shared_offset_)
{
   init_vars();
   offsets_previous_ = shared_offset_.offsets();
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
   std::unique_ptr<RooArgSet> vars{likelihood_->getParameters()};
   // TODO: make sure this is the right list of parameters, compare to original
   // implementation in RooRealMPFE.cxx

   RooArgList varList(*vars);

   // Save in lists
   vars_.add(varList);
   save_vars_.addClone(varList);
}

void LikelihoodJob::update_state()
{
   if (get_manager()->process_manager().is_worker()) {
      bool more;

      auto mode = get_manager()->messenger().receive_from_master_on_worker<update_state_mode>(&more);
      assert(more);

      switch (mode) {
      case update_state_mode::parameters: {
         state_id_ = get_manager()->messenger().receive_from_master_on_worker<RooFit::MultiProcess::State>(&more);
         assert(more);
         auto message = get_manager()->messenger().receive_from_master_on_worker<zmq::message_t>(&more);
         auto message_begin = message.data<update_state_t>();
         auto message_end = message_begin + message.size() / sizeof(update_state_t);
         std::vector<update_state_t> to_update(message_begin, message_end);
         for (auto const &item : to_update) {
            RooRealVar *rvar = static_cast<RooRealVar *>(vars_.at(item.var_index));
            rvar->setVal(static_cast<double>(item.value));
            if (rvar->isConstant() != item.is_constant) {
               rvar->setConstant(static_cast<bool>(item.is_constant));
            }
         }

         if (more) {
            // offsets also incoming
            auto offsets_message = get_manager()->messenger().receive_from_master_on_worker<zmq::message_t>(&more);
            assert(!more);
            auto offsets_message_begin = offsets_message.data<ROOT::Math::KahanSum<double>>();
            std::size_t N_offsets = offsets_message.size() / sizeof(ROOT::Math::KahanSum<double>);
            shared_offset_.offsets().resize(N_offsets);
            auto offsets_message_end = offsets_message_begin + N_offsets;
            std::copy(offsets_message_begin, offsets_message_end, shared_offset_.offsets().begin());
         }

         break;
      }
      case update_state_mode::offsetting: {
         LikelihoodWrapper::enableOffsetting(get_manager()->messenger().receive_from_master_on_worker<bool>(&more));
         assert(!more);
         break;
      }
      }
   }
}

std::size_t LikelihoodJob::getNEventTasks()
{
   std::size_t val = n_event_tasks_;
   if (val == MultiProcess::Config::LikelihoodJob::automaticNEventTasks) {
      val = 1;
   }
   if (val > likelihood_->getNEvents()) {
      val = likelihood_->getNEvents();
   }
   return val;
}

/// \warning In automatic mode, this function can start MultiProcess (forks, starts workers, etc)!
std::size_t LikelihoodJob::getNComponentTasks()
{
   std::size_t val = n_component_tasks_;
   if (val == MultiProcess::Config::LikelihoodJob::automaticNComponentTasks) {
      val = get_manager()
               ->process_manager()
               .N_workers(); // get_manager() is the call that can start MultiProcess, mentioned above
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
      for (std::size_t ix = 0u; ix < static_cast<std::size_t>(vars_.size()); ++ix) {
         valChanged = !vars_[ix].isIdentical(save_vars_[ix], true);
         constChanged = (vars_[ix].isConstant() != save_vars_[ix].isConstant());

         if (valChanged || constChanged) {
            if (constChanged) {
               (static_cast<RooRealVar *>(&save_vars_[ix]))->setConstant(vars_[ix].isConstant());
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
      bool update_offsets = isOffsetting() && shared_offset_.offsets() != offsets_previous_;
      if (!to_update.empty() || update_offsets) {
         ++state_id_;
         zmq::message_t message(to_update.begin(), to_update.end());
         // always send Job id first! This is used in worker_loop to route the
         // update_state call to the correct Job.
         if (update_offsets) {
            zmq::message_t offsets_message(shared_offset_.offsets().begin(), shared_offset_.offsets().end());
            get_manager()->messenger().publish_from_master_to_workers(id_, update_state_mode::parameters, state_id_,
                                                                      std::move(message), std::move(offsets_message));
            offsets_previous_ = shared_offset_.offsets();
         } else {
            get_manager()->messenger().publish_from_master_to_workers(id_, update_state_mode::parameters, state_id_,
                                                                      std::move(message));
         }
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
      // evaluate the serial likelihood to set the offsets
      if (do_offset_ && shared_offset_.offsets().empty()) {
         likelihood_serial_.evaluate();
         // note: we don't need to get the offsets from the serial likelihood, because they are already coupled through
         // the shared_ptr
      }

      // update parameters that changed since last calculation (or creation if first time)
      updateWorkersParameters();

      // master fills queue with tasks
      auto N_tasks = getNEventTasks() * getNComponentTasks();
      for (std::size_t ix = 0; ix < N_tasks; ++ix) {
         get_manager()->queue()->add({id_, state_id_, ix});
      }
      n_tasks_at_workers_ = N_tasks;

      // wait for task results back from workers to master
      gather_worker_results();

      RooNaNPacker packedNaN;

      // Note: initializing result_ to results_[0] instead of zero-initializing it makes
      // a difference due to Kahan sum precision. This way, a single-worker run gives
      // the same result as a run with serial likelihood. Adding the terms to a zero
      // initial sum can cancel the carry in some cases, causing divergent values.
      result_ = results_[0];
      packedNaN.accumulate(results_[0].Sum());
      for (auto item_it = results_.cbegin() + 1; item_it != results_.cend(); ++item_it) {
         result_ += *item_it;
         packedNaN.accumulate(item_it->Sum());
      }
      results_.clear();

      if (packedNaN.getPayload() != 0) {
         result_ = ROOT::Math::KahanSum<double>(packedNaN.getNaNWithPayload());
      }

      if (TMath::IsNaN(result_.Sum())) {
         RooAbsReal::logEvalError(nullptr, GetName().c_str(), "function value is NAN");
      }
   }
}

// --- RESULT LOGISTICS ---

void LikelihoodJob::send_back_task_result_from_worker(std::size_t /*task*/)
{
   int numErrors = RooAbsReal::numEvalErrors();

   if (numErrors) {
      // Clear error list on local side
      RooAbsReal::clearEvalErrorLog();
   }

   task_result_t task_result{id_, result_.Result(), result_.Carry(), numErrors > 0};
   zmq::message_t message(sizeof(task_result_t));
   memcpy(message.data(), &task_result, sizeof(task_result_t));
   get_manager()->messenger().send_from_worker_to_master(std::move(message));
}

bool LikelihoodJob::receive_task_result_on_master(const zmq::message_t &message)
{
   auto task_result = message.data<task_result_t>();
   results_.emplace_back(task_result->value, task_result->carry);
   if (task_result->has_errors) {
      RooAbsReal::logEvalError(nullptr, "LikelihoodJob", "evaluation errors at the worker processes", "no servervalue");
   }
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
      if (do_offset_ && section_last == 1) {
         // we only subtract at the end of event sections, otherwise the offset is subtracted for each event split
         result_ -= shared_offset_.offsets()[0];
      }
      break;
   }
   case LikelihoodType::subsidiary: {
      result_ = likelihood_->evaluatePartition({0, 1}, 0, 0);
      if (do_offset_ && offsetting_mode_ == OffsettingMode::full) {
         result_ -= shared_offset_.offsets()[0];
      }
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

      result_ = ROOT::Math::KahanSum<double>();
      RooNaNPacker packedNaN;
      for (std::size_t comp_ix = components_first; comp_ix < components_last; ++comp_ix) {
         auto component_result = likelihood_->evaluatePartition({section_first, section_last}, comp_ix, comp_ix + 1);
         packedNaN.accumulate(component_result.Sum());
         if (do_offset_ && section_last == 1 &&
             shared_offset_.offsets()[comp_ix] != ROOT::Math::KahanSum<double>(0, 0)) {
            // we only subtract at the end of event sections, otherwise the offset is subtracted for each event split
            result_ += (component_result - shared_offset_.offsets()[comp_ix]);
         } else {
            result_ += component_result;
         }
      }
      if (packedNaN.getPayload() != 0) {
         result_ = ROOT::Math::KahanSum<double>(packedNaN.getNaNWithPayload());
      }

      break;
   }
   }
}

void LikelihoodJob::enableOffsetting(bool flag)
{
   likelihood_serial_.enableOffsetting(flag);
   LikelihoodWrapper::enableOffsetting(flag);
   if (RooFit::MultiProcess::JobManager::is_instantiated()) {
      printf("WARNING: when calling MinuitFcnGrad::setOffsetting after the run has already been started the "
             "MinuitFcnGrad::likelihood_in_gradient object (a LikelihoodSerial) on the workers can no longer be "
             "updated! This function (LikelihoodJob::enableOffsetting) can in principle be used outside of "
             "MinuitFcnGrad, but be aware of this limitation. To do a minimization with a different offsetting "
             "setting, please delete all RooFit::MultiProcess based objects so that the forked processes are killed "
             "and then set up a new RooMinimizer.\n");
      updateWorkersOffsetting();
   }
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
