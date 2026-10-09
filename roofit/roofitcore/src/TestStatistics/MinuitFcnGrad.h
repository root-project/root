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

   bool returnsInMinuit2ParameterSpace() const { return _gradient->usesMinuitInternalValues(); }

   /// \name ROOT::Minuit2::FCNBase interface
   /// @{
   double operator()(std::vector<double> const &x) const override;
   bool HasGradient() const override { return true; }
   std::vector<double> Gradient(std::vector<double> const &x) const override;
   // Unhide the 4-argument overload from FCNBase, which forwards to Gradient().
   // Otherwise GCC's -Woverloaded-virtual (enabled with -Werror on some CI
   // targets) complains because we only override the 5-argument overload here.
   using FCNBase::GradientWithPrevResult;
   std::vector<double> GradientWithPrevResult(std::vector<double> const &x, double *previous_grad, double *previous_g2,
                                              double *previous_gstep, double fValAtX) const override;
   ROOT::Minuit2::GradientParameterSpace gradParameterSpace() const override;
   /// @}

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

   mutable bool _minuitInternalRooFitXMismatch = false;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
