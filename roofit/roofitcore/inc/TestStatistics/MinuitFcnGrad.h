/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
#define ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad

#include <Fit/ParameterSettings.h>
#include "ROOT/RMakeUnique.hxx"
#include "Math/IFunction.h" // ROOT::Math::IMultiGradFunction
#include "RooArgList.h"
#include "RooRealVar.h"
#include "TestStatistics/LikelihoodWrapper.h"
#include "TestStatistics/LikelihoodGradientWrapper.h"
#include "TestStatistics/LikelihoodJob.h"
#include "TestStatistics/LikelihoodGradientJob.h"
#include "TestStatistics/RooAbsL.h"
#include "RooMinimizer.h"
#include "RooAbsMinimizerFcn.h"

// forward declaration
class RooAbsL;
class RooAbsReal;

namespace RooFit {
namespace TestStatistics {

class MinuitFcnGrad : public ROOT::Math::IMultiGradFunction, public RooAbsMinimizerFcn {
public:
   MinuitFcnGrad(LikelihoodWrapper *_likelihood, LikelihoodGradientWrapper *_gradient, RooMinimizer* context,
                 bool verbose = false);
   ROOT::Math::IMultiGradFunction *Clone() const override;

   // override to include gradient strategy synchronization:
   Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings, Bool_t optConst = kTRUE,
                      Bool_t verbose = kFALSE) override;

   // used inside Minuit:
   bool returnsInMinuit2ParameterSpace() const override;

private:
   // IMultiGradFunction overrides necessary for Minuit: DoEval, Gradient, (has)G2ndDerivative and (has)GStepSize
   double DoEval(const double *x) const override;

public:
   void Gradient(const double *x, double *grad) const override;
   void G2ndDerivative(const double *x, double *g2) const override;
   void GStepSize(const double *x, double *gstep) const override;
   bool hasG2ndDerivative() const override;
   bool hasGStepSize() const override;

   // part of IMultiGradFunction interface, used widely both in Minuit and in RooFit:
   unsigned int NDim() const override;

private:
   // The following three overrides will not actually be used in this class, so they will throw:
   double DoDerivative(const double *x, unsigned int icoord) const override;
   double DoSecondDerivative(const double * /*x*/, unsigned int /*icoord*/) const override;
   double DoStepSize(const double * /*x*/, unsigned int /*icoord*/) const override;

   void optimizeConstantTerms(bool constStatChange, bool constValChange) override;

   // members
   std::unique_ptr<LikelihoodWrapper> likelihood;
   std::unique_ptr<LikelihoodGradientWrapper> gradient;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_MinuitFcnGrad
