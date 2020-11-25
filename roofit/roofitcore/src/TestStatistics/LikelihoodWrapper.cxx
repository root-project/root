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
#include <TestStatistics/LikelihoodWrapper.h>
#include <TestStatistics/RooAbsL.h> // need complete type for likelihood->...
#include <TestStatistics/MinuitFcnGrad.h>

namespace RooFit {
namespace TestStatistics {

LikelihoodWrapper::LikelihoodWrapper(std::shared_ptr<RooAbsL> likelihood,
                                     std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean,
                                     RooMinimizer *minimizer)
   : likelihood_(std::move(likelihood)), _minimizer(minimizer),
     calculation_is_clean_(std::move(calculation_is_clean)) /*, _minimizer_fcn(minimizer_fcn)*/
{
   // Note to future maintainers: take care when storing the minimizer_fcn pointer. The
   // RooAbsMinimizerFcn subclasses may get cloned inside MINUIT, which means the pointer
   // should also somehow be updated in this class.
}

void LikelihoodWrapper::synchronize_with_minimizer(const ROOT::Math::MinimizerOptions & /*options*/) {}

void LikelihoodWrapper::constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt)
{
   likelihood_->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
}

void LikelihoodWrapper::synchronize_parameter_settings(
   const std::vector<ROOT::Fit::ParameterSettings> & /*parameter_settings*/)
{
}

std::string LikelihoodWrapper::GetName() const
{
   return likelihood_->GetName();
}
std::string LikelihoodWrapper::GetTitle() const
{
   return likelihood_->GetTitle();
}
double LikelihoodWrapper::defaultErrorLevel() const
{
   return likelihood_->defaultErrorLevel();
}
bool LikelihoodWrapper::is_offsetting() const
{
   return likelihood_->is_offsetting();
}
void LikelihoodWrapper::enable_offsetting(bool flag)
{
   likelihood_->enable_offsetting(flag);
}

void LikelihoodWrapper::update_minuit_internal_parameter_values(const std::vector<double>& /*minuit_internal_x*/) {}
void LikelihoodWrapper::update_minuit_external_parameter_values(const std::vector<double>& /*minuit_external_x*/) {}

} // namespace TestStatistics
} // namespace RooFit
