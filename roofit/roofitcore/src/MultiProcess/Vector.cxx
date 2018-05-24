/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   IP, Inti Pelupessy,  NL eScience Center, i.pelupessy@esciencecenter.nl  *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <MultiProcess/Vector.h>

namespace RooFit {
  namespace MultiProcess {
    ~Vector::Vector() {
      TaskManager::remove_job_object(id);
    }

    void Vector::update_real(std::size_t ix, double val, bool is_constant) {
      if (get_manager()->is_worker()) {
        RooRealVar *rvar = (RooRealVar *) _vars.at(ix);
        rvar->setVal(val);
        if (rvar->isConstant() != is_constant) {
          rvar->setConstant(is_constant);
        }
      }
    }

    void Vector::gather_worker_results() {
      if (!retrieved) {
        get_manager()->retrieve();
      }
    }

    void Vector::receive_task_result_on_queue(std::size_t task, std::size_t worker_id) {
      result_t result = get_manager()->receive_from_worker_on_queue<result_t>(worker_id);
      results[task] = result;
    }

    void Vector::send_back_results_from_queue_to_master() {
      get_manager()->send_from_queue_to_master(results.size());
      for (auto const &item : results) {
        get_manager()->send_from_queue_to_master(item.first, item.second);
      }
    }

    void Vector::clear_results() {
      // empty results cache
      results.clear();
    }

    void Vector::receive_results_on_master() {
      std::size_t N_job_tasks = get_manager()->receive_from_queue_on_master<std::size_t>();
      for (std::size_t task_ix = 0ul; task_ix < N_job_tasks; ++task_ix) {
        std::size_t task_id = get_manager()->receive_from_queue_on_master<std::size_t>();
        results[task_id] = get_manager()->receive_from_queue_on_master<result_t>();
      }
    }

    double Vector::call_double_const_method(std::string key) {
      return (this->*double_const_methods[key])();
    }

  }  // namespace MultiProcess
}  // namespace RooFit
