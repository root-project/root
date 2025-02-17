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

#include <RooFit/TestStatistics/LikelihoodWrapper.h>

#include <RooFit/TestStatistics/RooAbsL.h> // need complete type for likelihood->...
#include <RooFit/TestStatistics/RooUnbinnedL.h>
#include <RooFit/TestStatistics/RooSumL.h> // need complete type for dynamic cast
#include <RooFit/TestStatistics/RooBinnedL.h>
#include <RooFit/TestStatistics/RooSubsidiaryL.h>

#include <RooMsgService.h>

#include "MinuitFcnGrad.h"

// including derived classes for factory method
#include "LikelihoodSerial.h"
#ifdef ROOFIT_MULTIPROCESS
#include "LikelihoodJob.h"
#endif // ROOFIT_MULTIPROCESS

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
 * \param[in] calculation_is_clean Shared pointer to the object that keeps track of what has been evaluated for the
 * current parameter set provided by Minuit. This information can be used by different calculators, so must be shared
 * between them.
 */
LikelihoodWrapper::LikelihoodWrapper(std::shared_ptr<RooAbsL> likelihood,
                                     std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean,
                                     SharedOffset offset)
   : likelihood_(std::move(likelihood)),
     calculation_is_clean_(std::move(calculation_is_clean)),
     shared_offset_(std::move(offset))
{
   // determine likelihood type
   if (dynamic_cast<RooUnbinnedL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::unbinned;
   } else if (dynamic_cast<RooBinnedL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::binned;
   } else if (dynamic_cast<RooSumL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::sum;
   } else if (dynamic_cast<RooSubsidiaryL *>(likelihood_.get()) != nullptr) {
      likelihood_type_ = LikelihoodType::subsidiary;
   } else {
      throw std::logic_error("in LikelihoodWrapper constructor: _likelihood is not of a valid subclass!");
   }
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
   // Clear offsets if feature is disabled so that it is recalculated next time it is enabled
   if (!do_offset_) {
      shared_offset_.clear();
   }
}

void LikelihoodWrapper::setOffsettingMode(OffsettingMode mode)
{
   offsetting_mode_ = mode;
   if (isOffsetting()) {
      oocoutI(nullptr, Minimization)
         << "LikelihoodWrapper::setOffsettingMode(" << GetName()
         << "): changed offsetting mode while offsetting was enabled; resetting offset values" << std::endl;
      shared_offset_.clear();
   }
}

/// (Re)calculate (on each worker) all component offsets.
///
/// Note that these are calculated over the full event range! This will decrease the effectiveness of offsetting
/// proportionally to the number of splits over the event range. The alternative, however, becomes very complex to
/// implement and maintain, so this is a compromise.
void LikelihoodWrapper::calculate_offsets()
{
   shared_offset_.clear();

   switch (likelihood_type_) {
   case LikelihoodType::unbinned: {
      shared_offset_.offsets().push_back(likelihood_->evaluatePartition({0, 1}, 0, 0));
      shared_offset_.offsets_save().emplace_back();
      break;
   }
   case LikelihoodType::binned: {
      shared_offset_.offsets().push_back(likelihood_->evaluatePartition({0, 1}, 0, 0));
      break;
   }
   case LikelihoodType::subsidiary: {
      if (offsetting_mode_ == OffsettingMode::full) {
         shared_offset_.offsets().push_back(likelihood_->evaluatePartition({0, 1}, 0, 0));
      } else {
         shared_offset_.offsets().emplace_back();
      }
      break;
   }
   case LikelihoodType::sum: {
      auto sum_likelihood = dynamic_cast<RooSumL *>(likelihood_.get());
      assert(sum_likelihood != nullptr);
      for (std::size_t comp_ix = 0; comp_ix < likelihood_->getNComponents(); ++comp_ix) {
         auto component_subsidiary_cast =
            dynamic_cast<RooSubsidiaryL *>(sum_likelihood->GetComponents()[comp_ix].get());
         if (offsetting_mode_ == OffsettingMode::full || component_subsidiary_cast == nullptr) {
            // Note: we leave out here the check for whether the calculated value is zero to reduce complexity, which
            // RooNLLVar does do at this (equivalent) point. Instead, we check whether the offset is zero when
            // subtracting it in evaluations.
            shared_offset_.offsets().push_back(likelihood_->evaluatePartition({0, 1}, comp_ix, comp_ix + 1));
            oocoutI(nullptr, Minimization)
               << "LikelihoodSerial::evaluate(" << GetName() << "): Likelihood offset now set to "
               << shared_offset_.offsets().back().Sum() << std::endl;
         } else {
            shared_offset_.offsets().emplace_back();
         }
         // default initialize the save offsets, just in case we have an unbinned component
         shared_offset_.offsets_save().emplace_back();
      }
      break;
   }
   }
}

/// \note Currently we do not recalculate the offset value, so in practice swapped offsets
///       are zero/disabled. This differs from using RooNLLVar, so your fit may yield slightly
///       different values.
void LikelihoodWrapper::setApplyWeightSquared(bool flag)
{
   std::vector<std::size_t> comp_was_changed;
   switch (likelihood_type_) {
   case LikelihoodType::unbinned: {
      auto unbinned_likelihood = dynamic_cast<RooUnbinnedL *>(likelihood_.get());
      assert(unbinned_likelihood != nullptr);
      if (unbinned_likelihood->setApplyWeightSquared(flag))
         comp_was_changed.emplace_back(0);
      break;
   }
   case LikelihoodType::sum: {
      auto sum_likelihood = dynamic_cast<RooSumL *>(likelihood_.get());
      assert(sum_likelihood != nullptr);
      for (std::size_t comp_ix = 0; comp_ix < likelihood_->getNComponents(); ++comp_ix) {
         auto component_unbinned_cast = dynamic_cast<RooUnbinnedL *>(sum_likelihood->GetComponents()[comp_ix].get());
         if (component_unbinned_cast != nullptr) {
            if (component_unbinned_cast->setApplyWeightSquared(flag))
               comp_was_changed.emplace_back(comp_ix);
         }
      }
      break;
   }
   default: {
      throw std::logic_error("LikelihoodWrapper::setApplyWeightSquared can only be used on unbinned likelihoods, but "
                             "the wrapped likelihood_ member is not a RooUnbinnedL nor a RooSumL containing an unbinned"
                             "component!");
   }
   }
   if (!comp_was_changed.empty()) {
      shared_offset_.swap(comp_was_changed);
   }
}

void LikelihoodWrapper::updateMinuitInternalParameterValues(const std::vector<double> & /*minuit_internal_x*/) {}
void LikelihoodWrapper::updateMinuitExternalParameterValues(const std::vector<double> & /*minuit_external_x*/) {}

/// Factory method.
std::unique_ptr<LikelihoodWrapper>
LikelihoodWrapper::create(LikelihoodMode likelihoodMode, std::shared_ptr<RooAbsL> likelihood,
                          std::shared_ptr<WrapperCalculationCleanFlags> calculationIsClean, SharedOffset offset)
{
   switch (likelihoodMode) {
   case LikelihoodMode::serial: {
      return std::make_unique<LikelihoodSerial>(std::move(likelihood), std::move(calculationIsClean),
                                                std::move(offset));
   }
   case LikelihoodMode::multiprocess: {
#ifdef ROOFIT_MULTIPROCESS
      return std::make_unique<LikelihoodJob>(std::move(likelihood), std::move(calculationIsClean), std::move(offset));
#else
      throw std::runtime_error("MinuitFcnGrad ctor with LikelihoodMode::multiprocess is not available in this build "
                               "without RooFit::Multiprocess!");
#endif
   }
   default: {
      throw std::logic_error("In MinuitFcnGrad constructor: likelihoodMode has an unsupported value!");
   }
   }
}

} // namespace TestStatistics
} // namespace RooFit
