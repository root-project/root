/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <MultiProcess/GradMinimizer.h>
#include <MultiProcess/messages.h>
#include <stdexcept>

#include <RooTimer.h>

namespace RooFit {
  namespace MultiProcess {
    GradMinimizerFcn::GradMinimizerFcn(RooAbsReal *funct, RooMinimizerGenericPtr context, std::size_t _N_workers,
                                       bool verbose) :
        RooFit::MultiProcess::Vector<RooGradMinimizerFcn>(_N_workers, funct, context, verbose) {
      N_tasks = NDim();
      completed_task_ids.reserve(N_tasks);
      // TODO: make sure that the full gradients are sent back so that the
      // derivator will depart from correct state next step everywhere!
    }

    // copy ctor (necessary for Clone)
    GradMinimizerFcn::GradMinimizerFcn(const GradMinimizerFcn& other) :
        RooFit::MultiProcess::Vector<RooGradMinimizerFcn>(other),
        N_tasks(other.N_tasks),
        completed_task_ids(other.completed_task_ids) {}

    ROOT::Math::IMultiGradFunction* GradMinimizerFcn::Clone() const {
      return new GradMinimizerFcn(*this) ;
    }

    // SYNCHRONIZATION FROM MASTER TO WORKERS

    void GradMinimizerFcn::update_state() {
      // TODO optimization: only send changed parameters (now sending all)
      RooFit::MultiProcess::M2Q msg = RooFit::MultiProcess::M2Q::update_real;
      for (std::size_t ix = 0; ix < NDim(); ++ix) {
        get_manager()->send_from_master_to_queue(msg, id, ix, _grad.Grad()(ix), false);
        get_manager()->send_from_master_to_queue(msg, id, ix + 1 * NDim(), _grad.G2()(ix), false);
        get_manager()->send_from_master_to_queue(msg, id, ix + 2 * NDim(), _grad.Gstep()(ix), false);
      }

      std::size_t ix = 0;
      for (auto parameter : _grad_params) {
        get_manager()->send_from_master_to_queue(msg, id, ix + 3 * NDim(), parameter, false);
        ++ix;
      }
    }

    void GradMinimizerFcn::update_real(std::size_t ix, double val, bool /*is_constant*/)  {
      if (get_manager()->is_worker()) {
        // ix is defined in "flat" FunctionGradient space ix_dim * size + ix_component
        switch (ix / NDim()) {
          case 0: {
            mutable_grad()(ix % NDim()) = val;
            break;
          }
          case 1: {
            mutable_g2()(ix % NDim()) = val;
            break;
          }
          case 2: {
            mutable_gstep()(ix % NDim()) = val;
            break;
          }
          case 3: {
            sync_parameter(val, ix % NDim());
            break;
          }
          default:
            throw std::runtime_error("ix out of range in GradMinimizerFcn::update_real!");
        }
      }
    }

    // END SYNCHRONIZATION FROM MASTER TO WORKERS


    // SYNCHRONIZATION FROM WORKERS TO MASTER

    void GradMinimizerFcn::send_back_task_result_from_worker(std::size_t task) {
      get_manager()->send_from_worker_to_queue(id, task, _grad.Grad()(task), _grad.G2()(task), _grad.Gstep()(task));
    }

    void GradMinimizerFcn::receive_task_result_on_queue(std::size_t task, std::size_t worker_id) {
      completed_task_ids.push_back(task);
      mutable_grad()(task)  = get_manager()->receive_from_worker_on_queue<double>(worker_id);
      mutable_g2()(task)    = get_manager()->receive_from_worker_on_queue<double>(worker_id);
      mutable_gstep()(task) = get_manager()->receive_from_worker_on_queue<double>(worker_id);
    }

    void GradMinimizerFcn::send_back_results_from_queue_to_master() {
      get_manager()->send_from_queue_to_master(completed_task_ids.size());
      for (auto task : completed_task_ids) {
        get_manager()->send_from_queue_to_master(task, _grad.Grad()(task), _grad.G2()(task), _grad.Gstep()(task));
      }
    }

    void GradMinimizerFcn::clear_results() {
      completed_task_ids.clear();
    }

    void GradMinimizerFcn::receive_results_on_master() {
      std::size_t N_completed_tasks = get_manager()->receive_from_queue_on_master<std::size_t>();
      for (unsigned int sync_ix = 0u; sync_ix < N_completed_tasks; ++sync_ix) {
        std::size_t task = get_manager()->receive_from_queue_on_master<std::size_t>();
        mutable_grad()(task) = get_manager()->receive_from_queue_on_master<double>();
        mutable_g2()(task) = get_manager()->receive_from_queue_on_master<double>();
        mutable_gstep()(task) = get_manager()->receive_from_queue_on_master<double>();
      }
    }

    // END SYNCHRONIZATION FROM WORKERS TO MASTER


    // ACTUAL WORK

    void GradMinimizerFcn::evaluate_task(std::size_t task) {
      run_derivator(task);
    }

    double GradMinimizerFcn::get_task_result(std::size_t task) {
      // this is useless here
      return _grad.Grad()(task);
    }


    void GradMinimizerFcn::CalculateAll(const double *x) {
      if (get_manager()->is_master()) {
        // do Grad, G2 and Gstep here and then just return results from the
        // separate functions below
        bool was_not_synced = sync_parameters(x);
        if (was_not_synced) {
          // update parameters and object states that changed since last calculation (or creation if first time)
          RooWallTimer timer;
          update_state();
          timer.stop();
          auto time_update_state = timer.timing_s();

          timer.start();
          // activate work mode
          get_manager()->set_work_mode(true);

          // master fills queue with tasks
          for (std::size_t ix = 0; ix < N_tasks; ++ix) {
            JobTask job_task(id, ix);
            get_manager()->to_queue(job_task);
          }
          waiting_for_queued_tasks = true;

          // wait for task results back from workers to master (put into _grad)
          gather_worker_results();

          // end work mode
          get_manager()->set_work_mode(false);
          timer.stop();

          std::cout << "update_state: " << time_update_state << "s, gradient work: " << timer.timing_s() << "s" << std::endl;
        }
      }
    }

    void GradMinimizerFcn::Gradient(const double *x, double *grad) const {
      const_cast<GradMinimizerFcn *>(this)->mutable_Gradient(x, grad);
    }

    void GradMinimizerFcn::mutable_Gradient(const double *x, double *grad) {
      if (get_manager()->is_master()) {
        CalculateAll(x);

        // TODO: maybe make a flag to avoid this copy operation, but maybe not worth the effort
        // put the results from _grad into *grad
        for (std::size_t ix = 0; ix < NDim(); ++ix) {
          grad[ix] = _grad.Grad()(ix);
        }
      }
    }

    void GradMinimizerFcn::G2ndDerivative(const double *x, double *g2) const {
      const_cast<GradMinimizerFcn *>(this)->mutable_G2ndDerivative(x, g2);
    }

    void GradMinimizerFcn::mutable_G2ndDerivative(const double *x, double *g2) {
      if (get_manager()->is_master()) {
        CalculateAll(x);

        // TODO: maybe make a flag to avoid this copy operation, but maybe not worth the effort
        // put the results from _grad into *grad
        for (std::size_t ix = 0; ix < NDim(); ++ix) {
          g2[ix] = _grad.G2()(ix);
        }
      }
    }

    void GradMinimizerFcn::GStepSize(const double *x, double *gstep) const {
      const_cast<GradMinimizerFcn *>(this)->mutable_GStepSize(x, gstep);
    }

    void GradMinimizerFcn::mutable_GStepSize(const double *x, double *gstep) {
      if (get_manager()->is_master()) {
        CalculateAll(x);

        // TODO: maybe make a flag to avoid this copy operation, but maybe not worth the effort
        // put the results from _grad into *grad
        for (std::size_t ix = 0; ix < NDim(); ++ix) {
          gstep[ix] = _grad.Gstep()(ix);
        }
      }
    }

    // END ACTUAL WORK

//      RooGradMinimizerFcn(funct, context, verbose), N_workers() {}
  } // namespace MultiProcess
} // namespace RooFit
