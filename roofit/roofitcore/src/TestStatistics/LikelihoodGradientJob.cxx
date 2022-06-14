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

#include "LikelihoodGradientJob.h"

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Queue.h"
#include "RooMsgService.h"
#include "RooMinimizer.h"

#include "Minuit2/MnStrategy.h"

namespace RooFit {
namespace TestStatistics {

LikelihoodGradientJob::LikelihoodGradientJob(std::shared_ptr<RooAbsL> likelihood,
                                             std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean,
                                             std::size_t N_dim, RooMinimizer *minimizer)
   : LikelihoodGradientWrapper(std::move(likelihood), std::move(calculation_is_clean), N_dim, minimizer), grad_(N_dim)
{
   // Note to future maintainers: take care when storing the minimizer_fcn pointer. The
   // RooAbsMinimizerFcn subclasses may get cloned inside MINUIT, which means the pointer
   // should also somehow be updated in this class.
   N_tasks_ = N_dim;
   minuit_internal_x_.reserve(N_dim);
}

LikelihoodGradientJob::LikelihoodGradientJob(const LikelihoodGradientJob &other)
   : MultiProcess::Job(other), LikelihoodGradientWrapper(other), grad_(other.grad_), gradf_(other.gradf_),
     N_tasks_(other.N_tasks_), minuit_internal_x_(other.minuit_internal_x_)
{
}

LikelihoodGradientJob *LikelihoodGradientJob::clone() const
{
   return new LikelihoodGradientJob(*this);
}

void LikelihoodGradientJob::synchronizeParameterSettings(
   const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings)
{
   LikelihoodGradientWrapper::synchronizeParameterSettings(parameter_settings);
}

void LikelihoodGradientJob::synchronizeParameterSettings(
   ROOT::Math::IMultiGenFunction *function, const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings)
{
   gradf_.SetInitialGradient(function, parameter_settings, grad_);
}

void LikelihoodGradientJob::synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions &options)
{
   setStrategy(options.Strategy());
   setErrorLevel(options.ErrorDef());
}

void LikelihoodGradientJob::setStrategy(int istrat)
{
   assert(istrat >= 0);
   ROOT::Minuit2::MnStrategy strategy(static_cast<unsigned int>(istrat));

   setStepTolerance(strategy.GradientStepTolerance());
   setGradTolerance(strategy.GradientTolerance());
   setNCycles(strategy.GradientNCycles());
}

void LikelihoodGradientJob::setStepTolerance(double step_tolerance) const
{
   gradf_.SetStepTolerance(step_tolerance);
}

void LikelihoodGradientJob::setGradTolerance(double grad_tolerance) const
{
   gradf_.SetGradTolerance(grad_tolerance);
}

void LikelihoodGradientJob::setNCycles(unsigned int ncycles) const
{
   gradf_.SetNCycles(ncycles);
}

void LikelihoodGradientJob::setErrorLevel(double error_level) const
{
   gradf_.SetErrorLevel(error_level);
}

///////////////////////////////////////////////////////////////////////////////
/// Job overrides:

void LikelihoodGradientJob::evaluate_task(std::size_t task)
{
   run_derivator(task);
}

// SYNCHRONIZATION FROM WORKERS TO MASTER

void LikelihoodGradientJob::send_back_task_result_from_worker(std::size_t task)
{
   task_result_t task_result{id_, task, grad_[task]};
   zmq::message_t message(sizeof(task_result_t));
   memcpy(message.data(), &task_result, sizeof(task_result_t));
   get_manager()->messenger().send_from_worker_to_master(std::move(message));
}

bool LikelihoodGradientJob::receive_task_result_on_master(const zmq::message_t &message)
{
   auto result = message.data<task_result_t>();
   grad_[result->task_id] = result->grad;
   --N_tasks_at_workers_;
   bool job_completed = (N_tasks_at_workers_ == 0);
   return job_completed;
}

// END SYNCHRONIZATION FROM WORKERS TO MASTER

// SYNCHRONIZATION FROM MASTER TO WORKERS (STATE)

void LikelihoodGradientJob::update_workers_state()
{
   // TODO optimization: only send changed parameters (now sending all)
   zmq::message_t gradient_message(grad_.begin(), grad_.end());
   zmq::message_t minuit_internal_x_message(minuit_internal_x_.begin(), minuit_internal_x_.end());
   ++state_id_;
   get_manager()->messenger().publish_from_master_to_workers(id_, state_id_, std::move(gradient_message),
                                                             std::move(minuit_internal_x_message));
}

void LikelihoodGradientJob::update_state()
{
   bool more;

   state_id_ = get_manager()->messenger().receive_from_master_on_worker<MultiProcess::State>(&more);

   auto gradient_message = get_manager()->messenger().receive_from_master_on_worker<zmq::message_t>(&more);
   assert(more);
   auto gradient_message_begin = gradient_message.data<ROOT::Minuit2::DerivatorElement>();
   auto gradient_message_end =
      gradient_message_begin + gradient_message.size() / sizeof(ROOT::Minuit2::DerivatorElement);
   std::copy(gradient_message_begin, gradient_message_end, grad_.begin());

   auto minuit_internal_x_message = get_manager()->messenger().receive_from_master_on_worker<zmq::message_t>(&more);
   assert(!more);
   auto minuit_internal_x_message_begin = minuit_internal_x_message.data<double>();
   auto minuit_internal_x_message_end =
      minuit_internal_x_message_begin + minuit_internal_x_message.size() / sizeof(double);
   std::copy(minuit_internal_x_message_begin, minuit_internal_x_message_end, minuit_internal_x_.begin());

   gradf_.SetupDifferentiate(minimizer_->getMultiGenFcn(), minuit_internal_x_.data(),
                             minimizer_->fitter()->Config().ParamsSettings());
}

// END SYNCHRONIZATION FROM MASTER TO WORKERS (STATE)

///////////////////////////////////////////////////////////////////////////////
/// Calculation stuff (mostly duplicates of RooGradMinimizerFcn code):

void LikelihoodGradientJob::run_derivator(unsigned int i_component) const
{
   // Calculate the derivative etc for these parameters
   grad_[i_component] = gradf_.FastPartialDerivative(
      minimizer_->getMultiGenFcn(), minimizer_->fitter()->Config().ParamsSettings(), i_component, grad_[i_component]);
}

void LikelihoodGradientJob::calculate_all()
{
   if (get_manager()->process_manager().is_master()) {
      update_workers_state();

      // master fills queue with tasks
      for (std::size_t ix = 0; ix < N_tasks_; ++ix) {
         MultiProcess::JobTask job_task{id_, state_id_, ix};
         get_manager()->queue().add(job_task);
      }
      N_tasks_at_workers_ = N_tasks_;
      // wait for task results back from workers to master (put into _grad)
      gather_worker_results();

      calculation_is_clean_->gradient = true;
   }
}

void LikelihoodGradientJob::fillGradient(double *grad)
{
   if (get_manager()->process_manager().is_master()) {
      if (!calculation_is_clean_->gradient) {
         calculate_all();
      }

      // put the results from _grad into *grad
      for (Int_t ix = 0; ix < minimizer_->getNPar(); ++ix) {
         grad[ix] = grad_[ix].derivative;
      }
   }
}

void LikelihoodGradientJob::fillGradientWithPrevResult(double *grad, double *previous_grad, double *previous_g2,
                                                       double *previous_gstep)
{
   if (get_manager()->process_manager().is_master()) {
      for (std::size_t i_component = 0; i_component < N_tasks_; ++i_component) {
         grad_[i_component] = {previous_grad[i_component], previous_g2[i_component], previous_gstep[i_component]};
      }

      if (!calculation_is_clean_->gradient) {
         calculate_all();
      }

      // put the results from _grad into *grad
      for (Int_t ix = 0; ix < minimizer_->getNPar(); ++ix) {
         grad[ix] = grad_[ix].derivative;
         previous_g2[ix] = grad_[ix].second_derivative;
         previous_gstep[ix] = grad_[ix].step_size;
      }
   }
}

void LikelihoodGradientJob::updateMinuitInternalParameterValues(const std::vector<double> &minuit_internal_x)
{
   minuit_internal_x_ = minuit_internal_x;
}

bool LikelihoodGradientJob::usesMinuitInternalValues()
{
   return true;
}

} // namespace TestStatistics
} // namespace RooFit
