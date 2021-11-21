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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientJob
#define ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientJob

#include "RooFit/MultiProcess/Job.h"
#include "RooFit/TestStatistics/LikelihoodGradientWrapper.h"

#include "Math/MinimizerOptions.h"
#include "Minuit2/NumericalDerivator.h"
#include "Minuit2/MnMatrix.h"

#include <vector>

namespace RooFit {
namespace TestStatistics {

class LikelihoodGradientJob : public MultiProcess::Job, public LikelihoodGradientWrapper {
public:
   LikelihoodGradientJob(std::shared_ptr<RooAbsL> likelihood,
                         std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean, std::size_t N_dim,
                         RooMinimizer *minimizer);
   LikelihoodGradientJob *clone() const override;
   LikelihoodGradientJob(const LikelihoodGradientJob &other);

   void fillGradient(double *grad) override;
   void fillGradientWithPrevResult(double *grad, double *previous_grad, double *previous_g2,
                                   double *previous_gstep) override;

   void update_state() override;

   enum class GradientCalculatorMode { ExactlyMinuit2, AlmostMinuit2 };

private:
   void run_derivator(unsigned int i_component) const;

   void synchronizeParameterSettings(ROOT::Math::IMultiGenFunction *function,
                                     const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) override;
   // this overload must also be overridden here so that the one above doesn't trigger a overloaded-virtual warning:
   void synchronizeParameterSettings(const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) override;

   void synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions &options) override;
   void setStrategy(int istrat);
   void setStepTolerance(double step_tolerance) const;
   void setGradTolerance(double grad_tolerance) const;
   void setNCycles(unsigned int ncycles) const;
   void setErrorLevel(double error_level) const;

   void updateMinuitInternalParameterValues(const std::vector<double> &minuit_internal_x) override;

   bool usesMinuitInternalValues() override;

   // Job overrides:
   void evaluate_task(std::size_t task) override;

   struct task_result_t {
      std::size_t job_id;
      std::size_t task_id;
      ROOT::Minuit2::DerivatorElement grad;
   };
   void send_back_task_result_from_worker(std::size_t task) override;
   bool receive_task_result_on_master(const zmq::message_t &message) override;

   void update_workers_state();
   void calculate_all();

   // members

   // mutables below are because ROOT::Math::IMultiGradFunction::DoDerivative is const
   mutable std::vector<ROOT::Minuit2::DerivatorElement> grad_;
   mutable ROOT::Minuit2::NumericalDerivator gradf_;

   std::size_t N_tasks_ = 0;
   std::size_t N_tasks_at_workers_ = 0;
   std::vector<double> minuit_internal_x_;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_LikelihoodGradientJob
