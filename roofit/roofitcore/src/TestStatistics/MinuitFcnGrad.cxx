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

#include "RooMsgService.h"
#include "RooMinimizer.h"
#include "TestStatistics/MinuitFcnGrad.h"

namespace RooFit {
namespace TestStatistics {

// Note: MinuitFcnGrad takes ownership of the wrappers, i.e. it will destroy them when it dies!
MinuitFcnGrad::MinuitFcnGrad(LikelihoodWrapper *_likelihood, LikelihoodGradientWrapper *_gradient,
                             RooMinimizer* context, bool verbose)
   : RooAbsMinimizerFcn(RooArgList(*likelihood->getParameters()), context, verbose),
   likelihood(_likelihood), gradient(_gradient)
{
   auto parameters = _context->fitter()->Config().ParamsSettings();
   synchronize_parameter_settings(parameters, kTRUE, verbose);
   likelihood->synchronize_parameter_settings(parameters);
   gradient->synchronize_parameter_settings(parameters);

   std::cerr << "Possibly the following code (see code) does not give the same values as the code it replaced from "
                "RooGradMinimizerFcn (commented out below), make sure!"
             << std::endl;
   //      set_strategy(ROOT::Math::MinimizerOptions::DefaultStrategy());
   //      set_error_level(ROOT::Math::MinimizerOptions::DefaultErrorDef());
   likelihood->synchronize_with_minimizer(ROOT::Math::MinimizerOptions());
   gradient->synchronize_with_minimizer(ROOT::Math::MinimizerOptions());
}

MinuitFcnGrad::MinuitFcnGrad(const MinuitFcnGrad & other)
: RooAbsMinimizerFcn(other), likelihood(other.likelihood->clone()), gradient(other.gradient->clone()) {};

// IMultiGradFunction overrides necessary for Minuit: DoEval, Gradient, G2ndDerivative and GStepSize
// The likelihood and gradient wrappers do the actual calculations.

double MinuitFcnGrad::DoEval(const double *x) const
{
   _evalCounter++;
   return likelihood->get_value(x);
}

void MinuitFcnGrad::Gradient(const double *x, double *grad) const
{
   gradient->fill_gradient(x, grad);
}

void MinuitFcnGrad::G2ndDerivative(const double *x, double *g2) const
{
   gradient->fill_second_derivative(x, g2);
}

void MinuitFcnGrad::GStepSize(const double *x, double *gstep) const
{
   gradient->fill_step_size(x, gstep);
}

ROOT::Math::IMultiGradFunction *MinuitFcnGrad::Clone() const
{
   return new MinuitFcnGrad(*this);
}

double MinuitFcnGrad::DoDerivative(const double * /*x*/, unsigned int /*icoord*/) const
{
   throw std::runtime_error("MinuitFcnGrad::DoDerivative is not implemented, please use Gradient instead.");
}

double MinuitFcnGrad::DoSecondDerivative(const double * /*x*/, unsigned int /*icoord*/) const
{
   throw std::runtime_error("MinuitFcnGrad::DoSecondDerivative is not implemented, please use G2ndDerivative instead.");
}

double MinuitFcnGrad::DoStepSize(const double * /*x*/, unsigned int /*icoord*/) const
{
   throw std::runtime_error("MinuitFcnGrad::DoStepSize is not implemented, please use GStepSize instead.");
}

bool MinuitFcnGrad::hasG2ndDerivative() const
{
   return true;
}

bool MinuitFcnGrad::hasGStepSize() const
{
   return true;
}

unsigned int MinuitFcnGrad::NDim() const
{
   return _nDim;
}

bool MinuitFcnGrad::returnsInMinuit2ParameterSpace() const
{
   return true;
}


void MinuitFcnGrad::optimizeConstantTerms(bool constStatChange, bool constValChange) {
   if (constStatChange) {

      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);

      oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
         << "MinuitFcnGrad::synchronize: set of constant parameters changed, rerunning const optimizer" << std::endl;
      likelihood->constOptimizeTestStatistic(RooAbsArg::ConfigChange);
   } else if (constValChange) {
      oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
         << "MinuitFcnGrad::synchronize: constant parameter values changed, rerunning const optimizer" << std::endl;
      likelihood->constOptimizeTestStatistic(RooAbsArg::ValueChange);
   }

   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors);
}


Bool_t
MinuitFcnGrad::Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters, Bool_t optConst, Bool_t verbose)
{
   Bool_t returnee = synchronize_parameter_settings(parameters, optConst, verbose);
   likelihood->synchronize_parameter_settings(parameters);
   gradient->synchronize_parameter_settings(parameters);

   likelihood->synchronize_with_minimizer(_context->fitter()->Config().MinimizerOptions());
   gradient->synchronize_with_minimizer(_context->fitter()->Config().MinimizerOptions());
   return returnee;
}

} // namespace TestStatistics
} // namespace RooFit
