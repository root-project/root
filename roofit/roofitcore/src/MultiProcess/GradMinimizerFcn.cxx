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

namespace RooFit {
  namespace MultiProcess {
    GradMinimizerFcn::GradMinimizerFcn(RooAbsReal *funct, RooMinimizerGenericPtr context, std::size_t _N_workers,
                                       bool verbose) :
        RooFit::MultiProcess::Vector<RooGradMinimizerFcn>(_N_workers, funct, context, verbose) {}

//    void GradMinimizerFcn::send_back_task_result_from_worker(std::size_t task) {
//    }
//
//    void GradMinimizerFcn::receive_task_result_on_queue(std::size_t task, std::size_t worker_id) {
//    }
//
//    void GradMinimizerFcn::send_back_results_from_queue_to_master() {
//    }
//
//    void GradMinimizerFcn::clear_results() {
//    }
//
//    void GradMinimizerFcn::receive_results_on_master() {
//    }

    void GradMinimizerFcn::evaluate_task(std::size_t task) {

    }

    double GradMinimizerFcn::get_task_result(std::size_t) {
      return 0;
    }
//      RooGradMinimizerFcn(funct, context, verbose), N_workers() {}
  } // namespace MultiProcess
} // namespace RooFit
