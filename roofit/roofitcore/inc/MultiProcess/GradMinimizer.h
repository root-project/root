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

#ifndef ROOFIT_MULTIPROCESS_GRADMINIMIZER_H
#define ROOFIT_MULTIPROCESS_GRADMINIMIZER_H

#include <map>
#include <MultiProcess/Vector.h>
#include <RooGradMinimizerFcn.h>
#include <RooMinimizer.h>

namespace RooFit {
  namespace MultiProcess {

    class GradMinimizerFcn : public RooFit::MultiProcess::Vector<RooGradMinimizerFcn> {
     public:
      GradMinimizerFcn(std::size_t n_workers, const RooGradMinimizerFcn& gmfcn);
      void init_vars() override;
      void update_parameters();

      // the const is inherited from ...::evaluate. We are not
      // actually const though, so we use a horrible hack.
//      Double_t evaluate() const override;
//      Double_t evaluate_non_const();

      // --- RESULT LOGISTICS ---
      void send_back_task_result_from_worker(std::size_t task) override;
      void receive_task_result_on_queue(std::size_t task, std::size_t worker_id) override;
      void send_back_results_from_queue_to_master() override;
      void clear_results() override;
      void receive_results_on_master() override;

     private:
      void evaluate_task(std::size_t task) override;
      double get_task_result(std::size_t /*task*/) override;

      // members
      std::map<std::size_t, double> carrys;
      double result = 0;
      double carry = 0;
      std::size_t N_tasks = 0;
    };

    using GradMinimizer = RooMinimizerTemplate<GradMinimizerFcn, RooFit::MinimizerType::Minuit2>;

  } // namespace MultiProcess
} // namespace RooFit

#endif //ROOFIT_MULTIPROCESS_GRADMINIMIZER_H
