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

#include "LikelihoodSerial.h"

#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooFit/TestStatistics/RooUnbinnedL.h>
#include <RooFit/TestStatistics/RooBinnedL.h>
#include <RooFit/TestStatistics/RooSubsidiaryL.h>
#include <RooFit/TestStatistics/RooSumL.h>
#include "RooRealVar.h"

namespace RooFit {
namespace TestStatistics {

/** \class LikelihoodSerial
 * \brief Serial likelihood calculation strategy implementation
 *
 * This class serves as a baseline reference implementation of the LikelihoodWrapper. It reimplements the previous
 * RooNLLVar "BulkPartition" single CPU strategy in the new RooFit::TestStatistics framework.
 *
 * \note The class is not intended for use by end-users. We recommend to either use RooMinimizer with a RooAbsL derived
 * likelihood object, or to use a higher level entry point like RooAbsPdf::fitTo() or RooAbsPdf::createNLL().
 */

LikelihoodSerial::LikelihoodSerial(std::shared_ptr<RooAbsL> likelihood, std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean)
   : LikelihoodWrapper(std::move(likelihood), std::move(calculation_is_clean))
{
   initVars();
   // determine likelihood type
   if (dynamic_cast<RooUnbinnedL *>(likelihood_.get()) != nullptr) {
      likelihood_type = LikelihoodType::unbinned;
   } else if (dynamic_cast<RooBinnedL *>(likelihood_.get()) != nullptr) {
      likelihood_type = LikelihoodType::binned;
   } else if (dynamic_cast<RooSumL *>(likelihood_.get()) != nullptr) {
      likelihood_type = LikelihoodType::sum;
   } else if (dynamic_cast<RooSubsidiaryL *>(likelihood_.get()) != nullptr) {
      likelihood_type = LikelihoodType::subsidiary;
   } else {
      throw std::logic_error("in LikelihoodSerial constructor: _likelihood is not of a valid subclass!");
   }
   // Note to future maintainers: take care when storing the minimizer_fcn pointer. The
   // RooAbsMinimizerFcn subclasses may get cloned inside MINUIT, which means the pointer
   // should also somehow be updated in this class.
}

/// \brief Helper function for the constuctor.
///
/// This is a separate function (instead of just in ctor) for historical reasons.
/// Its predecessor RooRealMPFE::initVars() was used from multiple ctors, but also
/// from RooRealMPFE::constOptimizeTestStatistic at the end, which makes sense,
/// because it might change the set of variables. We may at some point want to do
/// this here as well.
void LikelihoodSerial::initVars()
{
   // Empty current lists
   _vars.removeAll();
   _saveVars.removeAll();

   // Retrieve non-constant parameters
   auto vars = std::make_unique<RooArgSet>(*likelihood_->getParameters());

   RooArgList varList(*vars);

   // Save in lists
   _vars.add(varList);
   _saveVars.addClone(varList);
}

void LikelihoodSerial::evaluate() {
   switch (likelihood_type) {
   case LikelihoodType::unbinned:
   case LikelihoodType::binned: {
      result = likelihood_->evaluatePartition({0, 1}, 0, 0);
      break;
   }
   case LikelihoodType::sum: {
      result = likelihood_->evaluatePartition({0, 1}, 0, likelihood_->getNComponents());
      break;
   }
   default: {
      throw std::logic_error("in LikelihoodSerial::evaluate_task: likelihood types other than binned, unbinned and simultaneous not yet implemented!");
      break;
   }
   }

   result = applyOffsetting(result);
}

} // namespace TestStatistics
} // namespace RooFit
