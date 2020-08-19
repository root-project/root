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
#include <Minuit2/MnStrategy.h>

#include <RooTimer.h>
#include "RooMsgService.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Queue.h"
#include "RooMinimizer.h"

namespace RooFit {
namespace TestStatistics {

LikelihoodGradientJob::LikelihoodGradientJob(std::shared_ptr<RooAbsL> likelihood, RooMinimizer *minimizer)
   : LikelihoodGradientWrapper(std::move(likelihood), minimizer), _grad(minimizer->getNPar()), _gradf(_grad)
{
}

LikelihoodGradientJob::LikelihoodGradientJob(const LikelihoodGradientJob &other)
: LikelihoodGradientWrapper(other), _grad(other._grad), _gradf(other._gradf, _grad)
{
}

LikelihoodGradientJob *LikelihoodGradientJob::clone() const
{
   return new LikelihoodGradientJob(*this);
}

void LikelihoodGradientJob::synchronize_parameter_settings(
   const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings)
{
   _gradf.SetInitialGradient(_minimizer->getMultiGenFcn(), parameter_settings);
}

void LikelihoodGradientJob::synchronize_with_minimizer(const ROOT::Math::MinimizerOptions &options)
{
   set_strategy(options.Strategy());
   set_error_level(options.ErrorDef());
}

void LikelihoodGradientJob::set_strategy(int istrat)
{
   assert(istrat >= 0);
   ROOT::Minuit2::MnStrategy strategy(static_cast<unsigned int>(istrat));

   set_step_tolerance(strategy.GradientStepTolerance());
   set_grad_tolerance(strategy.GradientTolerance());
   set_ncycles(strategy.GradientNCycles());
}

void LikelihoodGradientJob::set_step_tolerance(double step_tolerance) const
{
   _gradf.set_step_tolerance(step_tolerance);
}

void LikelihoodGradientJob::set_grad_tolerance(double grad_tolerance) const
{
   _gradf.set_grad_tolerance(grad_tolerance);
}

void LikelihoodGradientJob::set_ncycles(unsigned int ncycles) const
{
   _gradf.set_ncycles(ncycles);
}

void LikelihoodGradientJob::set_error_level(double error_level) const
{
   _gradf.set_error_level(error_level);
}


///////////////////////////////////////////////////////////////////////////////
/// Job overrides:

void LikelihoodGradientJob::evaluate_task(std::size_t task) {
   RooWallTimer timer;
   RooCPUTimer ctimer;
   run_derivator(task);
   ctimer.stop();
   timer.stop();
   oocxcoutD((TObject*)nullptr,Benchmarking1) << "worker_id: " << get_manager()->process_manager().worker_id() << ", task: " << task << ", partial derivative time: " << timer.timing_s() << "s -- cputime: " << ctimer.timing_s() << "s" << std::endl;
}

void LikelihoodGradientJob::update_real(std::size_t ix, double val, bool /*is_constant*/) {
   if (get_manager()->process_manager().is_worker()) {
      // ix is defined in "flat" FunctionGradient space ix_dim * size + ix_component
      switch (ix / _minimizer->getNPar()) {
      case 0: {
         mutable_grad()(ix % _minimizer->getNPar()) = val;
         break;
      }
      case 1: {
         mutable_g2()(ix % _minimizer->getNPar()) = val;
         break;
      }
      case 2: {
         mutable_gstep()(ix % _minimizer->getNPar()) = val;
         break;
      }
      case 3: {
         _minimizer->set_function_parameter_value(val, ix % _minimizer->getNPar());
         break;
      }
      default:
         throw std::runtime_error("ix out of range in LikelihoodGradientJob::update_real!");
      }
   }
}

void LikelihoodGradientJob::update_bool(std::size_t /*ix*/, bool /*value*/) {

}

// SYNCHRONIZATION FROM WORKERS TO MASTER

void LikelihoodGradientJob::send_back_task_result_from_worker(std::size_t task) {
   get_manager()->messenger().send_from_worker_to_queue(id, task, _grad.Grad()(task), _grad.G2()(task), _grad.Gstep()(task));
}

void LikelihoodGradientJob::receive_task_result_on_queue(std::size_t task, std::size_t worker_id) {
   completed_task_ids.push_back(task);
   mutable_grad()(task)  = get_manager()->messenger().receive_from_worker_on_queue<double>(worker_id);
   mutable_g2()(task)    = get_manager()->messenger().receive_from_worker_on_queue<double>(worker_id);
   mutable_gstep()(task) = get_manager()->messenger().receive_from_worker_on_queue<double>(worker_id);
}

void LikelihoodGradientJob::send_back_results_from_queue_to_master() {
   get_manager()->messenger().send_from_queue_to_master(completed_task_ids.size());
   for (auto task : completed_task_ids) {
      get_manager()->messenger().send_from_queue_to_master(task, _grad.Grad()(task), _grad.G2()(task), _grad.Gstep()(task));
   }
}

void LikelihoodGradientJob::clear_results() {
   completed_task_ids.clear();
}

void LikelihoodGradientJob::receive_results_on_master() {
   std::size_t N_completed_tasks = get_manager()->messenger().receive_from_queue_on_master<std::size_t>();
   for (unsigned int sync_ix = 0u; sync_ix < N_completed_tasks; ++sync_ix) {
      std::size_t task = get_manager()->messenger().receive_from_queue_on_master<std::size_t>();
      mutable_grad()(task) = get_manager()->messenger().receive_from_queue_on_master<double>();
      mutable_g2()(task) = get_manager()->messenger().receive_from_queue_on_master<double>();
      mutable_gstep()(task) = get_manager()->messenger().receive_from_queue_on_master<double>();
   }
}

// END SYNCHRONIZATION FROM WORKERS TO MASTER


///////////////////////////////////////////////////////////////////////////////
/// Calculation stuff (mostly duplicates of RooGradMinimizerFcn code):

void LikelihoodGradientJob::run_derivator(unsigned int i_component) const
{
   // Calculate the derivative etc for these parameters
   auto parameter_values = _minimizer->get_function_parameter_values();
   std::tie(mutable_grad()(i_component), mutable_g2()(i_component), mutable_gstep()(i_component)) =
      _gradf.partial_derivative(_minimizer->getMultiGenFcn(), parameter_values.data(), _minimizer->fitter()->Config().ParamsSettings(), i_component);
}


///////////////////////////////////////////////////////////////////////////////
/// copy pasted and adapted from old MP::GradMinimizerFcn:

void LikelihoodGradientJob::update_workers_state() {
   // TODO optimization: only send changed parameters (now sending all)
   RooFit::MultiProcess::M2Q msg = RooFit::MultiProcess::M2Q::update_real;
   for (Int_t ix = 0; ix < _minimizer->getNPar(); ++ix) {
      get_manager()->messenger().send_from_master_to_queue(msg, id, ix, _grad.Grad()(ix), false);
   }
   for (Int_t ix = 0; ix < _minimizer->getNPar(); ++ix) {
      get_manager()->messenger().send_from_master_to_queue(msg, id, ix + 1 * _minimizer->getNPar(), _grad.G2()(ix), false);
   }
   for (Int_t ix = 0; ix < _minimizer->getNPar(); ++ix) {
      get_manager()->messenger().send_from_master_to_queue(msg, id, ix + 2 * _minimizer->getNPar(), _grad.Gstep()(ix), false);
   }

   std::size_t ix = 0;
   auto parameter_values = _minimizer->get_function_parameter_values();
   for (auto& parameter : parameter_values) {
      get_manager()->messenger().send_from_master_to_queue(msg, id, ix + 3 * _minimizer->getNPar(), parameter, false);
      ++ix;
   }
}

void LikelihoodGradientJob::calculate_all() {
   auto get_time = [](){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();};
   decltype(get_time()) t1, t2;

   if (get_manager()->process_manager().is_master()) {
      // do Grad, G2 and Gstep here and then just return results from the
      // separate functions below

      // update parameters and object states that changed since last calculation (or creation if first time)
      t1 = get_time();
      update_workers_state();
      t2 = get_time();

      RooWallTimer timer;

      // master fills queue with tasks
      for (std::size_t ix = 0; ix < N_tasks; ++ix) {
         MultiProcess::JobTask job_task(id, ix);
         get_manager()->queue().add(job_task);
      }

      // wait for task results back from workers to master (put into _grad)
      gather_worker_results();

      timer.stop();

      oocxcoutD((TObject*)nullptr,Benchmarking1) << "update_state: " << (t2 - t1)/1.e9 << "s (from " << t1 << " to " << t2 << "ns), gradient work: " << timer.timing_s() << "s" << std::endl;
   }
}

void LikelihoodGradientJob::fill_gradient(double *grad) {
   if (get_manager()->process_manager().is_master()) {
      calculate_all();

      // TODO: maybe make a flag to avoid this copy operation, but maybe not worth the effort
      // put the results from _grad into *grad
      for (Int_t ix = 0; ix < _minimizer->getNPar(); ++ix) {
         grad[ix] = _grad.Grad()(ix);
      }
   }
}


void LikelihoodGradientJob::fill_second_derivative(double *g2) {
   if (get_manager()->process_manager().is_master()) {
      calculate_all();

      // TODO: maybe make a flag to avoid this copy operation, but maybe not worth the effort
      // put the results from _grad into *grad
      for (Int_t ix = 0; ix < _minimizer->getNPar(); ++ix) {
         g2[ix] = _grad.G2()(ix);
      }
   }
}

void LikelihoodGradientJob::fill_step_size(double *gstep) {
   if (get_manager()->process_manager().is_master()) {
      calculate_all();

      // TODO: maybe make a flag to avoid this copy operation, but maybe not worth the effort
      // put the results from _grad into *grad
      for (Int_t ix = 0; ix < _minimizer->getNPar(); ++ix) {
         gstep[ix] = _grad.Gstep()(ix);
      }
   }
}

ROOT::Minuit2::MnAlgebraicVector &LikelihoodGradientJob::mutable_grad() const
{
   return const_cast<ROOT::Minuit2::MnAlgebraicVector &>(_grad.Grad());
}
ROOT::Minuit2::MnAlgebraicVector &LikelihoodGradientJob::mutable_g2() const
{
   return const_cast<ROOT::Minuit2::MnAlgebraicVector &>(_grad.G2());
}
ROOT::Minuit2::MnAlgebraicVector &LikelihoodGradientJob::mutable_gstep() const
{
   return const_cast<ROOT::Minuit2::MnAlgebraicVector &>(_grad.Gstep());
}

} // namespace TestStatistics
} // namespace RooFit
