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

class RooMinimizer;

namespace RooFit {
namespace TestStatistics {

class MinuitFcnGrad : public RooAbsMinimizerFcn {
public:
   MinuitFcnGrad(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &absL, RooMinimizer *context,
                 std::vector<ROOT::Fit::ParameterSettings> &parameters, LikelihoodMode likelihoodMode,
                 LikelihoodGradientMode likelihoodGradientMode);

   /// Overridden from RooAbsMinimizerFcn to include gradient strategy synchronization.
   bool Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings) override;

   // used inside Minuit:
   inline bool returnsInMinuit2ParameterSpace() const { return _gradient->usesMinuitInternalValues(); }

   inline void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override
   {
      applyToLikelihood([&](auto &l) { l.constOptimizeTestStatistic(opcode, doAlsoTrackingOpt); });
   }

   ROOT::Math::IMultiGenFunction *getMultiGenFcn() override { return _multiGenFcn.get(); }

   double operator()(const double *x) const;

   /// IMultiGradFunction overrides necessary for Minuit
   void Gradient(const double *x, double *grad) const;
   void GradientWithPrevResult(const double *x, double *grad, double *previous_grad, double *previous_g2,
                               double *previous_gstep) const;

   inline std::string getFunctionName() const override { return _likelihood->GetName(); }

   inline std::string getFunctionTitle() const override { return _likelihood->GetTitle(); }

   inline void setOffsetting(bool flag) override
   {
      applyToLikelihood([&](auto &l) { l.enableOffsetting(flag); });
      if (!flag) {
         offsets_reset_ = true;
      }
   }

private:
   bool syncParameterValuesFromMinuitCalls(const double *x, bool minuit_internal) const;

   template <class Func>
   void applyToLikelihood(Func &&func) const
   {
      func(*_likelihood);
      if (_likelihoodInGradient && _likelihood != _likelihoodInGradient) {
         func(*_likelihoodInGradient);
      }
   }

   // members
   // the likelihoods are shared_ptrs because they may point to the same object
   std::shared_ptr<LikelihoodWrapper> _likelihood;
   std::shared_ptr<LikelihoodWrapper> _likelihoodInGradient;
   std::unique_ptr<LikelihoodGradientWrapper> _gradient;
   mutable bool _calculatingGradient = false;

   mutable std::shared_ptr<WrapperCalculationCleanFlags> _calculationIsClean;

   mutable std::vector<double> _minuitInternalX;
   mutable std::vector<double> _minuitExternalX;
   // offsets_reset_ should be reset also when applyWeightSquared is activated in LikelihoodWrappers;
   // currently setting this is not supported, so it doesn't happen.
   mutable bool offsets_reset_ = false;
   void syncOffsets() const;

   std::unique_ptr<ROOT::Math::IMultiGradFunction> _multiGenFcn;

   mutable bool _minuitInternalRooFitXMismatch = false;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
