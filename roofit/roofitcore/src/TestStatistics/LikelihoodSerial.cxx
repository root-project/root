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


#include <TestStatistics/LikelihoodSerial.h>
#include <TestStatistics/kahan_sum.h>
#include <TestStatistics/RooAbsL.h>
#include <TestStatistics/RooUnbinnedL.h>
#include <TestStatistics/RooBinnedL.h>
#include <TestStatistics/RooSubsidiaryL.h>
#include <TestStatistics/RooSumL.h>
#include "RooRealVar.h"

namespace RooFit {
namespace TestStatistics {

LikelihoodSerial::LikelihoodSerial(std::shared_ptr<RooAbsL> likelihood, std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean/*,
                                   RooMinimizer *minimizer*/)
   : LikelihoodWrapper(std::move(likelihood), std::move(calculation_is_clean)/*, minimizer*/)
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

// This is a separate function (instead of just in ctor) for historical reasons.
// Its predecessor RooRealMPFE::initVars() was used from multiple ctors, but also
// from RooRealMPFE::constOptimizeTestStatistic at the end, which makes sense,
// because it might change the set of variables. We may at some point want to do
// this here as well.
void LikelihoodSerial::initVars()
{
   // Empty current lists
   _vars.removeAll();
   _saveVars.removeAll();

   // Retrieve non-constant parameters
   auto vars = std::make_unique<RooArgSet>(
      *likelihood_->getParameters()); // TODO: make sure this is the right list of parameters, compare to original
                                           // implementation in RooRealMPFE.cxx

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
      carry = likelihood_->getCarry();
      break;
   }
   case LikelihoodType::sum: {
      result = likelihood_->evaluatePartition({0, 1}, 0, likelihood_->getNComponents());
      carry = likelihood_->getCarry();
      // TODO: this normalization part below came from RooOptTestStatistic::evaluate, probably this just means you need to do the normalization on master only when doing parallel calculation. Make sure of this! In any case, it is currently not relevant, because the norm term is 1 by default and is only overridden for the RooDataWeightAverage class.
//      // Only apply global normalization if SimMaster doesn't have MP master
//      if (numSets() == 1) {
//         const Double_t norm = globalNormalization();
//         result /= norm;
//         carry /= norm;
//      }
      break;
   }
   default: {
      throw std::logic_error("in LikelihoodSerial::evaluate_task: likelihood types other than binned, unbinned and simultaneous not yet implemented!");
      break;
   }
   }

   applyOffsetting(result, carry);
}

} // namespace TestStatistics
} // namespace RooFit