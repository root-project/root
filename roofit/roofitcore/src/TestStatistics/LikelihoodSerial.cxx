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
#include "RooRealVar.h"
#include "RooNaNPacker.h"

#include "TMath.h" // IsNaN

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

LikelihoodSerial::LikelihoodSerial(std::shared_ptr<RooAbsL> likelihood,
                                   std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean,
                                   SharedOffset offset)
   : LikelihoodWrapper(std::move(likelihood), std::move(calculation_is_clean), std::move(offset))
{
   initVars();
}

/// \brief Helper function for the constructor.
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
   std::unique_ptr<RooArgSet> vars{likelihood_->getParameters()};

   RooArgList varList(*vars);

   // Save in lists
   _vars.add(varList);
   _saveVars.addClone(varList);
}

void LikelihoodSerial::evaluate()
{
   if (do_offset_ && shared_offset_.offsets().empty()) {
      calculate_offsets();
   }

   switch (likelihood_type_) {
   case LikelihoodType::unbinned:
   case LikelihoodType::binned: {
      result = likelihood_->evaluatePartition({0, 1}, 0, 0);
      if (do_offset_) {
         result -= shared_offset_.offsets()[0];
      }
      break;
   }
   case LikelihoodType::subsidiary: {
      result = likelihood_->evaluatePartition({0, 1}, 0, 0);
      if (do_offset_ && offsetting_mode_ == OffsettingMode::full) {
         result -= shared_offset_.offsets()[0];
      }
      break;
   }
   case LikelihoodType::sum: {
      result = ROOT::Math::KahanSum<double>();
      RooNaNPacker packedNaN;
      for (std::size_t comp_ix = 0; comp_ix < likelihood_->getNComponents(); ++comp_ix) {
         auto component_result = likelihood_->evaluatePartition({0, 1}, comp_ix, comp_ix + 1);
         packedNaN.accumulate(component_result.Sum());

         if (do_offset_ && shared_offset_.offsets()[comp_ix] != ROOT::Math::KahanSum<double>(0, 0)) {
            result += (component_result - shared_offset_.offsets()[comp_ix]);
         } else {
            result += component_result;
         }
      }
      if (packedNaN.getPayload() != 0) {
         result = ROOT::Math::KahanSum<double>(packedNaN.getNaNWithPayload());
      }
      break;
   }
   }

   if (TMath::IsNaN(result.Sum())) {
      RooAbsReal::logEvalError(nullptr, GetName().c_str(), "function value is NAN");
   }
}

} // namespace TestStatistics
} // namespace RooFit
