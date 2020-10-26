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
#include "RooAbsPdf.h"
#include "TestStatistics/MinuitFcnGrad.h"
#include "RooMinimizer.h"

#define DEBUG_STREAM(var) << " " #var "=" << var
#include <sys/types.h>
#include <unistd.h>

namespace RooFit {
namespace TestStatistics {

//MinuitFcnGrad::MinuitFcnGrad(const MinuitFcnGrad &other)
//   : RooAbsMinimizerFcn(other), likelihood(other.likelihood->clone()), gradient(other.gradient->clone()){};

// IMultiGradFunction overrides necessary for Minuit: DoEval, Gradient, G2ndDerivative and GStepSize
// The likelihood and gradient wrappers do the actual calculations.

double MinuitFcnGrad::DoEval(const double *x) const
{
   Bool_t parameters_changed = set_roofit_parameter_values(x);

   std::cout << "MinuitFcnGrad::DoEval @ PID" << getpid() << ": " DEBUG_STREAM(parameters_changed);

   // Calculate the function for these parameters
   RooAbsReal::setHideOffset(kFALSE);
   likelihood->evaluate();
   double fvalue = likelihood->return_result();
   calculation_is_clean->likelihood = true;
   RooAbsReal::setHideOffset(kTRUE);

   std::cout DEBUG_STREAM(fvalue) << std::endl;

   if (!parameters_changed) {
      return fvalue;
   }

   if (RooAbsPdf::evalError() || RooAbsReal::numEvalErrors() > 0 || fvalue > 1e30) {

      if (_printEvalErrors >= 0) {

         if (_doEvalErrorWall) {
            oocoutW(static_cast<RooAbsArg *>(nullptr), Eval)
               << "RooGradMinimizerFcn: Minimized function has error status." << std::endl
               << "Returning maximum FCN so far (" << _maxFCN
               << ") to force MIGRAD to back out of this region. Error log follows" << std::endl;
         } else {
            oocoutW(static_cast<RooAbsArg *>(nullptr), Eval)
               << "RooGradMinimizerFcn: Minimized function has error status but is ignored" << std::endl;
         }

         TIterator *iter = _floatParamList->createIterator();
         RooRealVar *var;
         Bool_t first(kTRUE);
         ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << "Parameter values: ";
         while ((var = (RooRealVar *)iter->Next())) {
            if (first) {
               first = kFALSE;
            } else
               ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << ", ";
            ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << var->GetName() << "=" << var->getVal();
         }
         delete iter;
         ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << std::endl;

         RooAbsReal::printEvalErrors(ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval), _printEvalErrors);
         ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << std::endl;
      }

      if (_doEvalErrorWall) {
         fvalue = _maxFCN + 1;
      }

      RooAbsPdf::clearEvalError();
      RooAbsReal::clearEvalErrorLog();
      _numBadNLL++;
   } else if (fvalue > _maxFCN) {
      _maxFCN = fvalue;
   }

   // Optional logging
   if (_verbose) {
      std::cout << "\nprevFCN" << (likelihood->is_offsetting() ? "-offset" : "") << " = " << std::setprecision(10)
                << fvalue << std::setprecision(4) << "  ";
      std::cout.flush();
   }

   _evalCounter++;
   //#ifndef NDEBUG
   //  std::cout << "RooGradMinimizerFcn " << this << " evaluations (in DoEval): " << _evalCounter <<
   //  std::endl;
   //#endif
   return fvalue;
}

/// Parameters here are the variables in the likelihood, i.e. the parameters
/// of the PDF that can vary given the observables in the dataset that constrains
/// it. This sets the RooFit RooAbsArg parameters, not values inside Minuit or
/// the Likelihood(Gradient)Wrapper implementations.
bool MinuitFcnGrad::set_roofit_parameter_values(const double *x) const
{
   bool a_parameter_has_been_updated = false;

   for (std::size_t ix = 0; ix < NDim(); ++ix) {
      bool update_this_parameter = SetPdfParamVal(ix, x[ix]);
      a_parameter_has_been_updated |= update_this_parameter;
   }

   if(a_parameter_has_been_updated) {
      std::cout << "setting stuff back to false" << std::endl;
      calculation_is_clean->set_all(false);
   }

   return a_parameter_has_been_updated;
}

void MinuitFcnGrad::Gradient(const double *x, double *grad) const
{
   std::cout << "MinuitFcnGrad::Gradient x[0] = " << x[0] << std::endl;
   set_roofit_parameter_values(x);
   gradient->fill_gradient(grad);
}

void MinuitFcnGrad::G2ndDerivative(const double *x, double *g2) const
{
   set_roofit_parameter_values(x);
   gradient->fill_second_derivative(g2);
}

void MinuitFcnGrad::GStepSize(const double *x, double *gstep) const
{
   set_roofit_parameter_values(x);
   gradient->fill_step_size(gstep);
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

void MinuitFcnGrad::optimizeConstantTerms(bool constStatChange, bool constValChange)
{
   if (constStatChange) {

      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);

      oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
         << "MinuitFcnGrad::synchronize: set of constant parameters changed, rerunning const optimizer" << std::endl;
      likelihood->constOptimizeTestStatistic(RooAbsArg::ConfigChange, true);
   } else if (constValChange) {
      oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
         << "MinuitFcnGrad::synchronize: constant parameter values changed, rerunning const optimizer" << std::endl;
      likelihood->constOptimizeTestStatistic(RooAbsArg::ValueChange, true);
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

std::string MinuitFcnGrad::getFunctionName() const
{
   return likelihood->GetName();
}

std::string MinuitFcnGrad::getFunctionTitle() const
{
   return likelihood->GetTitle();
}

void MinuitFcnGrad::setOptimizeConst(Int_t flag)
{
   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);

   if (_optConst && !flag) {
      if (_context->getPrintLevel() > -1)
         oocoutI(_context, Minimization) << "MinuitFcnGrad::setOptimizeConst: deactivating const optimization"
                                         << std::endl;
      likelihood->constOptimizeTestStatistic(RooAbsArg::DeActivate, true);
      _optConst = flag;
   } else if (!_optConst && flag) {
      if (_context->getPrintLevel() > -1)
         oocoutI(_context, Minimization) << "MinuitFcnGrad::setOptimizeConst: activating const optimization"
                                         << std::endl;
      likelihood->constOptimizeTestStatistic(RooAbsArg::Activate, flag > 1);
      _optConst = flag;
   } else if (_optConst && flag) {
      if (_context->getPrintLevel() > -1)
         oocoutI(_context, Minimization) << "MinuitFcnGrad::setOptimizeConst: const optimization already active"
                                         << std::endl;
   } else {
      if (_context->getPrintLevel() > -1)
         oocoutI(_context, Minimization) << "MinuitFcnGrad::setOptimizeConst: const optimization wasn't active"
                                         << std::endl;
   }

   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors);
}

} // namespace TestStatistics
} // namespace RooFit
