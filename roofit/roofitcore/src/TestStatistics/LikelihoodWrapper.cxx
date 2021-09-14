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

#include <TestStatistics/LikelihoodWrapper.h>

#include <TestStatistics/RooAbsL.h> // need complete type for likelihood->...
#include <TestStatistics/MinuitFcnGrad.h>
#include <TestStatistics/RooUnbinnedL.h>
#include <TestStatistics/RooSumL.h> // need complete type for dynamic cast

#include "RooMsgService.h"

namespace RooFit {
namespace TestStatistics {

/** \class LikelihoodWrapper
 * \brief Virtual base class for implementation of likelihood calculation strategies
 *
 * This class provides the interface necessary for RooMinimizer (through MinuitFcnGrad) to get the likelihood values it
 * needs for fitting the pdf to the data. The strategy by which these values are obtained is up to the implementer of
 * this class. Its intended purpose was mainly to allow for parallel calculation strategies, but serial strategies are
 * possible too, as illustrated in LikelihoodSerial.
 *
 * \note The class is not intended for use by end-users. We recommend to either use RooMinimizer with a RooAbsL derived
 * likelihood object, or to use a higher level entry point like RooAbsPdf::fitTo() or RooAbsPdf::createNLL().
 */

/*
 * \param[in] likelihood Shared pointer to the likelihood that must be evaluated
 * \param[in] calculation_is_clean Shared pointer to the object that keeps track of what has been evaluated for the current parameter set provided by Minuit. This information can be used by different calculators, so must be shared between them.
 */
LikelihoodWrapper::LikelihoodWrapper(std::shared_ptr<RooAbsL> likelihood,
                                     std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean)
   : likelihood_(std::move(likelihood)),
     calculation_is_clean_(std::move(calculation_is_clean))
{
   // Note to future maintainers: take care when storing the minimizer_fcn pointer. The
   // RooAbsMinimizerFcn subclasses may get cloned inside MINUIT, which means the pointer
   // should also somehow be updated in this class.
}

void LikelihoodWrapper::synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions & /*options*/) {}

void LikelihoodWrapper::synchronizeParameterSettings(
   const std::vector<ROOT::Fit::ParameterSettings> & /*parameter_settings*/)
{
}

void LikelihoodWrapper::constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt)
{
   likelihood_->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
}

double LikelihoodWrapper::defaultErrorLevel() const
{
   return likelihood_->defaultErrorLevel();
}
std::string LikelihoodWrapper::GetName() const
{
   return likelihood_->GetName();
}
std::string LikelihoodWrapper::GetTitle() const
{
   return likelihood_->GetTitle();
}

void LikelihoodWrapper::enableOffsetting(bool flag)
{
   do_offset_ = flag;
   // Clear offset if feature is disabled so that it is recalculated next time it is enabled
   if (!do_offset_) {
      offset_ = {};
   }
}

void LikelihoodWrapper::setOffsettingMode(OffsettingMode mode)
{
   offsetting_mode_ = mode;
   if (isOffsetting()) {
      oocoutI(static_cast<RooAbsArg *>(nullptr), Minimization) << "LikelihoodWrapper::setOffsettingMode(" << GetName() << "): changed offsetting mode while offsetting was enabled; resetting offset values" << std::endl;
      offset_ = {};
   }
}

ROOT::Math::KahanSum<double> LikelihoodWrapper::applyOffsetting(ROOT::Math::KahanSum<double> current_value)
{
   if (do_offset_) {

      // If no offset is stored enable this feature now
      if (offset_ == 0 && current_value != 0) {
         offset_ = current_value;
         if (offsetting_mode_ == OffsettingMode::legacy) {
            auto sum_likelihood = dynamic_cast<RooSumL*>(likelihood_.get());
            if (sum_likelihood != nullptr) {
               auto subsidiary_value = sum_likelihood->getSubsidiaryValue();
               // "undo" the addition of the subsidiary value to emulate legacy behavior
               offset_ -= subsidiary_value;
               // manually calculate result with zero carry, again to emulate legacy behavior
               return {current_value.Result() - offset_.Result()};
            }
         }
         oocoutI(static_cast<RooAbsArg *>(nullptr), Minimization)
            << "LikelihoodWrapper::applyOffsetting(" << GetName() << "): Likelihood offset now set to " << offset_
            << std::endl;
      }

      return current_value - offset_;
   } else {
      return current_value;
   }
}

/// When calculating an unbinned likelihood with square weights applied, a different offset
/// is necessary. Similar situations may ask for a separate offset as well. This function
/// switches between the two sets of offset values.
void LikelihoodWrapper::swapOffsets()
{
   std::swap(offset_, offset_save_);
}

void LikelihoodWrapper::setApplyWeightSquared(bool flag)
{
   RooUnbinnedL *unbinned_likelihood = dynamic_cast<RooUnbinnedL*>(likelihood_.get());
   if (unbinned_likelihood == nullptr) {
      throw std::logic_error("LikelihoodWrapper::setApplyWeightSquared can only be used on unbinned likelihoods, but the wrapped likelihood_ member is not a RooUnbinnedL!");
   }
   bool flag_was_changed = unbinned_likelihood->setApplyWeightSquared(flag);

   if (flag_was_changed) {
      swapOffsets();
   }
}

void LikelihoodWrapper::updateMinuitInternalParameterValues(const std::vector<double>& /*minuit_internal_x*/) {}
void LikelihoodWrapper::updateMinuitExternalParameterValues(const std::vector<double>& /*minuit_external_x*/) {}

} // namespace TestStatistics
} // namespace RooFit
