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

class MinuitFcnGrad : public ROOT::Math::IMultiGradFunction, public RooAbsMinimizerFcn {
public:
   MinuitFcnGrad(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &_likelihood, RooMinimizer *context,
                 std::vector<ROOT::Fit::ParameterSettings> &parameters, LikelihoodMode likelihoodMode,
                 LikelihoodGradientMode likelihoodGradientMode, bool verbose = false);

   inline ROOT::Math::IMultiGradFunction *Clone() const override { return new MinuitFcnGrad(*this); }

   /// Overridden from RooAbsMinimizerFcn to include gradient strategy synchronization.
   bool Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings, bool optConst,
                      bool verbose = false) override;

   // used inside Minuit:
   inline bool returnsInMinuit2ParameterSpace() const override { return gradient->usesMinuitInternalValues(); }

   inline void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override
   {
      likelihood->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
   }

   bool fit(ROOT::Fit::Fitter& fitter) const override { return fitter.FitFCN(*this); };
   ROOT::Math::IMultiGenFunction* getMultiGenFcn() override { return this; };

private:
   /// IMultiGradFunction override necessary for Minuit
   double DoEval(const double *x) const override;

public:
   /// IMultiGradFunction overrides necessary for Minuit
   void Gradient(const double *x, double *grad) const override;
   void GradientWithPrevResult(const double *x, double *grad, double *previous_grad, double *previous_g2,
                               double *previous_gstep) const override;

   /// Part of IMultiGradFunction interface, used widely both in Minuit and in RooFit.
   inline unsigned int NDim() const override { return _nDim; }

   inline std::string getFunctionName() const override { return likelihood->GetName(); }

   inline std::string getFunctionTitle() const override { return likelihood->GetTitle(); }

   inline void setOffsetting(bool flag) override { likelihood->enableOffsetting(flag); }

private:
   /// This override should not be used in this class, so it throws.
   double DoDerivative(const double *x, unsigned int icoord) const override;

   bool syncParameterValuesFromMinuitCalls(const double *x, bool minuit_internal) const;

   // members
   std::shared_ptr<LikelihoodWrapper> likelihood;
   std::shared_ptr<LikelihoodGradientWrapper> gradient;

public:
   mutable std::shared_ptr<WrapperCalculationCleanFlags> calculation_is_clean;

private:
   mutable std::vector<double> minuit_internal_x_;
   mutable std::vector<double> minuit_external_x_;

public:
   mutable bool minuit_internal_roofit_x_mismatch_ = false;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
