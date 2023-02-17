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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
#define ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad

#include "RooArgList.h"
#include "RooRealVar.h"
#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooFit/TestStatistics/LikelihoodWrapper.h>
#include <RooFit/TestStatistics/LikelihoodGradientWrapper.h>
#include "../RooAbsMinimizerFcn.h"

#include <Fit/ParameterSettings.h>
#include "Math/IFunction.h" // ROOT::Math::IMultiGradFunction

// forward declaration
class RooAbsReal;
class RooMinimizer;

namespace RooFit {
namespace TestStatistics {

class MinuitFcnGrad : public RooAbsMinimizerFcn {
public:
   MinuitFcnGrad(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &_likelihood, RooMinimizer *context,
                 std::vector<ROOT::Fit::ParameterSettings> &parameters, LikelihoodMode likelihoodMode,
                 LikelihoodGradientMode likelihoodGradientMode);

   /// Overridden from RooAbsMinimizerFcn to include gradient strategy synchronization.
   bool Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) override;

   // used inside Minuit:
   inline bool returnsInMinuit2ParameterSpace() const { return gradient->usesMinuitInternalValues(); }

   inline void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override
   {
      likelihood->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
      if (likelihood != likelihood_in_gradient) {
         likelihood_in_gradient->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
      }
   }

   ROOT::Math::IMultiGenFunction *getMultiGenFcn() override { return _multiGenFcn.get(); }

   double operator()(const double *x) const;

   /// IMultiGradFunction overrides necessary for Minuit
   void Gradient(const double *x, double *grad) const;
   void GradientWithPrevResult(const double *x, double *grad, double *previous_grad, double *previous_g2,
                               double *previous_gstep) const;

   inline std::string getFunctionName() const override { return likelihood->GetName(); }

   inline std::string getFunctionTitle() const override { return likelihood->GetTitle(); }

   inline void setOffsetting(bool flag) override
   {
      likelihood->enableOffsetting(flag);
      if (likelihood != likelihood_in_gradient) {
         likelihood_in_gradient->enableOffsetting(flag);
      }
   }

private:
   bool syncParameterValuesFromMinuitCalls(const double *x, bool minuit_internal) const;

   // members
   std::shared_ptr<LikelihoodWrapper> likelihood;
   std::shared_ptr<LikelihoodWrapper> likelihood_in_gradient;
   std::shared_ptr<LikelihoodGradientWrapper> gradient;
   mutable bool calculating_gradient_ = false;

public:
   mutable std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean;

private:
   mutable std::vector<double> minuit_internal_x_;
   mutable std::vector<double> minuit_external_x_;

   std::unique_ptr<ROOT::Math::IMultiGradFunction> _multiGenFcn;

public:
   mutable bool minuit_internal_roofit_x_mismatch_ = false;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
