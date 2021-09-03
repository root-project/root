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

/** \class LikelihoodGradientWrapper
 * \brief Virtual base class for implementation of likelihood gradient calculation strategies
 *
 * This class provides the interface necessary for RooMinimizer (through MinuitFcnGrad) to get the likelihood gradient
 * values it needs for fitting the pdf to the data. The strategy by which these values are obtained is up to the
 * implementer of this class. Its intended purpose was mainly to allow for parallel calculation strategies.
 *
 * \note The class is not intended for use by end-users. We recommend to either use RooMinimizer with a RooAbsL derived
 * likelihood object, or to use a higher level entry point like RooAbsPdf::fitTo() or RooAbsPdf::createNLL().
 */

/*
 * \param[in] likelihood Shared pointer to the likelihood that must be evaluated
 * \param[in] calculation_is_clean Shared pointer to the object that keeps track of what has been evaluated for the current parameter set provided by Minuit. This information can be used by different calculators, so must be shared between them.
 * \param[in] minimizer Raw pointer to the minimizer that owns the MinuitFcnGrad object that owns this wrapper object.
 */
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
