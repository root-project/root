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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper
#define ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper

#include "RooArgSet.h"
#include "RooAbsArg.h"  // enum ConstOpCode

#include <Fit/ParameterSettings.h>
#include "Math/MinimizerOptions.h"

#include <memory>  // shared_ptr
#include <string>

// forward declaration
class RooMinimizer;

namespace RooFit {
namespace TestStatistics {

// forward declaration
class RooAbsL;
struct WrapperCalculationCleanFlags;

enum class LikelihoodType {
   unbinned,
   binned,
   subsidiary,
   sum
};

// Previously, offsetting was only implemented for RooNLLVar components of a likelihood,
// not for RooConstraintSum terms. To emulate this behavior, use OffsettingMode::legacy. To
// also offset the RooSubsidiaryL component (equivalent of RooConstraintSum) of RooSumL
// likelihoods, use OffsettingMode::full.
enum class OffsettingMode {
   legacy,
   full
};

class LikelihoodWrapper {
public:
   LikelihoodWrapper(std::shared_ptr<RooAbsL> likelihood, std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean/*, RooMinimizer* minimizer*/);
   virtual ~LikelihoodWrapper() = default;
   virtual LikelihoodWrapper* clone() const = 0;

   virtual void evaluate() = 0;
   virtual double getResult() const = 0;

   // synchronize minimizer settings with calculators in child classes
   virtual void synchronizeWithMinimizer(const ROOT::Math::MinimizerOptions & options);
   virtual void synchronizeParameterSettings(const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings);
   // Minuit passes in parameter values that may not conform to RooFit internal standards (like applying range clipping),
   // but that the specific calculator does need. This function can be implemented to receive these Minuit-internal values:
   virtual void updateMinuitInternalParameterValues(const std::vector<double>& minuit_internal_x);
   virtual void updateMinuitExternalParameterValues(const std::vector<double>& minuit_external_x);

   // necessary from MinuitFcnGrad to reach likelihood properties:
   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt);
   double defaultErrorLevel() const;
   virtual std::string GetName() const;
   virtual std::string GetTitle() const;
   inline virtual bool isOffsetting() const { return do_offset_; }
   virtual void enableOffsetting(bool flag);
   void setOffsettingMode(OffsettingMode mode);
   inline double offset() const { return offset_; }
   inline double offsetCarry() const { return offset_carry_; }
   void setApplyWeightSquared(bool flag);

protected:
   std::shared_ptr<RooAbsL> likelihood_;
   std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean_;

   bool do_offset_ = false;
   double offset_ = 0;
   double offset_carry_ = 0;
   double offset_save_ = 0;      //!
   double offset_carry_save_ = 0; //!
   OffsettingMode offsetting_mode_ = OffsettingMode::legacy;
   void applyOffsetting(double &current_value, double &carry);
   void swapOffsets();
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper
