// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "TestStatistics/LikelihoodGradientWrapper.h"
#include "RooMinimizer.h"

namespace RooFit {
namespace TestStatistics {

LikelihoodGradientWrapper::LikelihoodGradientWrapper(std::shared_ptr<RooAbsL> likelihood,
                                                     std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean,
                                                     std::size_t /*N_dim*/, RooMinimizer *minimizer)
   : likelihood_(std::move(likelihood)), minimizer_(minimizer), calculation_is_clean_(std::move(calculation_is_clean))
{
   // Note to future maintainers: take care when storing the minimizer_fcn pointer. The
   // RooAbsMinimizerFcn subclasses may get cloned inside MINUIT, which means the pointer
   // should also somehow be updated in this class.
}

void LikelihoodGradientWrapper::synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions & /*options*/) {}

void LikelihoodGradientWrapper::synchronizeParameterSettings(
   const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings)
{
   synchronizeParameterSettings(minimizer_->getMultiGenFcn(), parameter_settings);
}

void LikelihoodGradientWrapper::updateMinuitInternalParameterValues(const std::vector<double>& /*minuit_internal_x*/) {}
void LikelihoodGradientWrapper::updateMinuitExternalParameterValues(const std::vector<double>& /*minuit_external_x*/) {}

} // namespace TestStatistics
} // namespace RooFit
