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

namespace RooFit {
namespace TestStatistics {

LikelihoodWrapper::LikelihoodWrapper(std::shared_ptr<RooAbsL> likelihood,
                                     std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean/*,
                                     RooMinimizer *minimizer*/)
   : likelihood_(std::move(likelihood)),/* minimizer_(minimizer),*/
     calculation_is_clean_(std::move(calculation_is_clean)) /*, minimizer_fcn_(minimizer_fcn)*/
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
      offset_ = 0;
      offset_carry_ = 0;
   }
}

void LikelihoodWrapper::setOffsettingMode(OffsettingMode mode)
{
   offsetting_mode_ = mode;
   if (isOffsetting()) {
      oocoutI(static_cast<RooAbsArg *>(nullptr), Minimization) << "LikelihoodWrapper::setOffsettingMode(" << GetName() << "): changed offsetting mode while offsetting was enabled; resetting offset values" << std::endl;
      offset_ = 0;
      offset_carry_ = 0;
   }
}

void LikelihoodWrapper::applyOffsetting(double &current_value, double &carry)
{
   if (do_offset_) {

      // If no offset is stored enable this feature now
      if (offset_ == 0 && current_value != 0) {
         offset_ = current_value;
         offset_carry_ = carry;
         if (offsetting_mode_ == OffsettingMode::legacy) {
            auto sum_likelihood = dynamic_cast<RooSumL*>(likelihood_.get());
            if (sum_likelihood != nullptr) {
               double subsidiary_value, subsidiary_carry;
               std::tie(subsidiary_value, subsidiary_carry) = sum_likelihood->getSubsidiaryValue();
               // "undo" the addition of the subsidiary value to emulate legacy behavior
               offset_ -= subsidiary_value;
               offset_carry_ -= subsidiary_carry;
               // then add 0 in Kahan summation way to make sure the carry gets taken up into the value if it should be
               double y = 0 - offset_carry_;
               double t = offset_ + y;
               offset_carry_ = (t - offset_) - y;
               offset_ = t;
               // also set carry to this value, again to emulate legacy behavior
               carry = offset_carry_;
            }
         }
         oocoutI(static_cast<RooAbsArg *>(nullptr), Minimization)
            << "LikelihoodWrapper::applyOffsetting(" << GetName() << "): Likelihood offset now set to " << offset_
            << std::endl;
      }

      // Subtract offset
      // old method:
//      {
//         double y = -offset_ - (carry + offset_carry_);
//         double t = current_value + y;
//         carry = (t - current_value) - y;
//         current_value = t;
//      }
      // Jonas method:
      {
         double new_value = current_value - offset_;
         double new_carry = carry - offset_carry_;
         // then add 0 in Kahan summation way to make sure the carry gets taken up into the value if it should be
         double y = 0 - new_carry;
         double t = new_value + y;
         carry = (t - new_value) - y;
         current_value = t;
      }
   }
}

/// When calculating an unbinned likelihood with square weights applied, a different offset
/// is necessary. Similar situations may ask for a separate offset as well. This function
/// switches between the two sets of offset values.
void LikelihoodWrapper::swapOffsets()
{
   std::swap(offset_, offset_save_);
   std::swap(offset_carry_, offset_carry_save_);
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
