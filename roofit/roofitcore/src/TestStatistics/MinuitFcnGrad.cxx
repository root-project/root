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

#include "TestStatistics/MinuitFcnGrad.h"

namespace RooFit {
namespace TestStatistics {

MinuitFcnGrad::MinuitFcnGrad(const MinuitFcnGrad &other)
   : ROOT::Math::IMultiGradFunction(other), _evalCounter(other._evalCounter), _maxFCN(other._maxFCN),
     _numBadNLL(other._numBadNLL), _printEvalErrors(other._printEvalErrors), _doEvalErrorWall(other._doEvalErrorWall),
     _nDim(other._nDim), _floatParamVec(other._floatParamVec), _context(other._context), _verbose(other._verbose)
{
   _floatParamList = new RooArgList(*other._floatParamList);
   _constParamList = new RooArgList(*other._constParamList);
   _initFloatParamList = (RooArgList *)other._initFloatParamList->snapshot(kFALSE);
   _initConstParamList = (RooArgList *)other._initConstParamList->snapshot(kFALSE);
}

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

MinuitFcnGrad::~MinuitFcnGrad()
{
   delete _floatParamList;
   delete _initFloatParamList;
   delete _constParamList;
   delete _initConstParamList;
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

Bool_t MinuitFcnGrad::synchronize_parameter_settings(std::vector<ROOT::Fit::ParameterSettings> &parameter_settings,
                                                     Bool_t optConst, Bool_t verbose)
{
   // Update parameter_settings with current information in RooAbsReal function parameters
   auto get_time = []() {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
         .count();
   };
   decltype(get_time()) t1, t2, t3, t4, t5, t6, t7;

   Bool_t constValChange(kFALSE);
   Bool_t constStatChange(kFALSE);

   Int_t index(0);

   t1 = get_time();

   // Handle eventual migrations from constParamList -> floatParamList
   for (index = 0; index < _constParamList->getSize(); index++) {

      auto par = dynamic_cast<RooRealVar *>(_constParamList->at(index));
      if (!par)
         continue;

      auto oldpar = dynamic_cast<RooRealVar *>(_initConstParamList->at(index));
      if (!oldpar)
         continue;

      // Test if constness changed
      if (!par->isConstant()) {

         // Remove from constList, add to floatList
         _constParamList->remove(*par);
         _floatParamList->add(*par);
         _initFloatParamList->addClone(*oldpar);
         _initConstParamList->remove(*oldpar);
         constStatChange = kTRUE;
         _nDim++;

         if (verbose) {
            oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
               << "MinuitFcnGrad::synchronize: parameter " << par->GetName() << " is now floating." << std::endl;
         }
      }

      // Test if value changed
      if (par->getVal() != oldpar->getVal()) {
         constValChange = kTRUE;
         if (verbose) {
            oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
               << "RooGradMinimizerFcn::synchronize: value of constant parameter " << par->GetName() << " changed from "
               << oldpar->getVal() << " to " << par->getVal() << std::endl;
         }
      }
   }

   t2 = get_time();

   // Update reference list
   *_initConstParamList = *_constParamList;

   t3 = get_time();

   // Synchronize MINUIT with function state
   // Handle floatParamList
   for (index = 0; index < _floatParamList->getSize(); index++) {
      auto par = dynamic_cast<RooRealVar *>(_floatParamList->at(index));

      if (!par)
         continue;

      Double_t pstep(0);
      Double_t pmin(0);
      Double_t pmax(0);

      if (!par->isConstant()) {

         // Verify that floating parameter is indeed of type RooRealVar
         if (!par->IsA()->InheritsFrom(RooRealVar::Class())) {
            oocoutW(static_cast<RooAbsArg *>(nullptr), Eval)
               << "MinuitFcnGrad::fit: Error, non-constant parameter " << par->GetName()
               << " is not of type RooRealVar, skipping" << std::endl;
            _floatParamList->remove(*par);
            index--;
            _nDim--;
            continue;
         }

         // Set the limits, if not infinite
         if (par->hasMin())
            pmin = par->getMin();
         if (par->hasMax())
            pmax = par->getMax();

         // Calculate step size
         pstep = par->getError();
         if (pstep <= 0) {
            // Floating parameter without error estitimate
            if (par->hasMin() && par->hasMax()) {
               pstep = 0.1 * (pmax - pmin);

               // Trim default choice of error if within 2 sigma of limit
               if (pmax - par->getVal() < 2 * pstep) {
                  pstep = (pmax - par->getVal()) / 2;
               } else if (par->getVal() - pmin < 2 * pstep) {
                  pstep = (par->getVal() - pmin) / 2;
               }

               // If trimming results in zero error, restore default
               if (pstep == 0) {
                  pstep = 0.1 * (pmax - pmin);
               }

            } else {
               pstep = 1;
            }
            if (verbose) {
               oocoutW(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: WARNING: no initial error estimate available for " << par->GetName()
                  << ": using " << pstep << std::endl;
            }
         }
      } else {
         pmin = par->getVal();
         pmax = par->getVal();
      }

      // new parameter
      if (index >= Int_t(parameter_settings.size())) {

         if (par->hasMin() && par->hasMax()) {
            parameter_settings.emplace_back(par->GetName(), par->getVal(), pstep, pmin, pmax);
         } else {
            parameter_settings.emplace_back(par->GetName(), par->getVal(), pstep);
            if (par->hasMin())
               parameter_settings.back().SetLowerLimit(pmin);
            else if (par->hasMax())
               parameter_settings.back().SetUpperLimit(pmax);
         }

         continue;
      }

      Bool_t oldFixed = parameter_settings[index].IsFixed();
      Double_t oldVar = parameter_settings[index].Value();
      Double_t oldVerr = parameter_settings[index].StepSize();
      Double_t oldVlo = parameter_settings[index].LowerLimit();
      Double_t oldVhi = parameter_settings[index].UpperLimit();

      if (par->isConstant() && !oldFixed) {

         // Parameter changes floating -> constant : update only value if necessary
         if (oldVar != par->getVal()) {
            parameter_settings[index].SetValue(par->getVal());
            if (verbose) {
               oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: value of parameter " << par->GetName() << " changed from " << oldVar
                  << " to " << par->getVal() << std::endl;
            }
         }
         parameter_settings[index].Fix();
         constStatChange = kTRUE;
         if (verbose) {
            oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
               << "MinuitFcnGrad::synchronize: parameter " << par->GetName() << " is now fixed." << std::endl;
         }

      } else if (par->isConstant() && oldFixed) {

         // Parameter changes constant -> constant : update only value if necessary
         if (oldVar != par->getVal()) {
            parameter_settings[index].SetValue(par->getVal());
            constValChange = kTRUE;

            if (verbose) {
               oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: value of fixed parameter " << par->GetName() << " changed from "
                  << oldVar << " to " << par->getVal() << std::endl;
            }
         }

      } else {
         // Parameter changes constant -> floating
         if (!par->isConstant() && oldFixed) {
            parameter_settings[index].Release();
            constStatChange = kTRUE;

            if (verbose) {
               oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: parameter " << par->GetName() << " is now floating." << std::endl;
            }
         }

         // Parameter changes constant -> floating : update all if necessary
         if (oldVar != par->getVal() || oldVlo != pmin || oldVhi != pmax || oldVerr != pstep) {
            parameter_settings[index].SetValue(par->getVal());
            parameter_settings[index].SetStepSize(pstep);
            if (par->hasMin() && par->hasMax())
               parameter_settings[index].SetLimits(pmin, pmax);
            else if (par->hasMin())
               parameter_settings[index].SetLowerLimit(pmin);
            else if (par->hasMax())
               parameter_settings[index].SetUpperLimit(pmax);
         }

         // Inform user about changes in verbose mode
         if (verbose) {
            // if ierr<0, par was moved from the const list and a message was already printed

            if (oldVar != par->getVal()) {
               oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: value of parameter " << par->GetName() << " changed from " << oldVar
                  << " to " << par->getVal() << std::endl;
            }
            if (oldVlo != pmin || oldVhi != pmax) {
               oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: limits of parameter " << par->GetName() << " changed from [" << oldVlo
                  << "," << oldVhi << "] to [" << pmin << "," << pmax << "]" << std::endl;
            }

            // If oldVerr=0, then parameter was previously fixed
            if (oldVerr != pstep && oldVerr != 0) {
               oocoutI(static_cast<RooAbsArg *>(nullptr), Eval)
                  << "MinuitFcnGrad::synchronize: error/step size of parameter " << par->GetName() << " changed from "
                  << oldVerr << " to " << pstep << std::endl;
            }
         }
      }
   }

   t4 = get_time();

   if (optConst) {
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

   t5 = get_time();

   updateFloatVec();

   t6 = get_time();

   likelihood->synchronize_parameter_settings(parameter_settings);
   gradient->synchronize_parameter_settings(parameter_settings);

   t7 = get_time();

   oocxcoutD((TObject *)nullptr, Benchmarking2)
      << "MinuitFcnGrad::synchronize_parameter_settings timestamps: " << t1 << " " << t2 << " " << t3 << " " << t4
      << " " << t5 << " " << t6 << " " << t7 << std::endl;

   return kFALSE;
}

void MinuitFcnGrad::updateFloatVec()
{
   _floatParamVec.clear();
   RooFIter iter = _floatParamList->fwdIterator();
   RooAbsArg *arg;
   _floatParamVec = std::vector<RooAbsArg *>(_floatParamList->getSize());
   Int_t i(0);
   while ((arg = iter.next())) {
      _floatParamVec[i++] = arg;
   }
}

Bool_t MinuitFcnGrad::Synchronize(std::vector<ROOT::Fit::ParameterSettings>& parameters,
                                        Bool_t optConst, Bool_t verbose) {
   Bool_t returnee = synchronize_parameter_settings(parameters, optConst, verbose);
   likelihood->synchronize_with_minimizer(_context.fitter()->Config().MinimizerOptions());
   gradient->synchronize_with_minimizer(_context.fitter()->Config().MinimizerOptions());
   return returnee;
}

RooArgList *MinuitFcnGrad::GetFloatParamList()
{
   return _floatParamList;
}
RooArgList *MinuitFcnGrad::GetConstParamList()
{
   return _constParamList;
}
RooArgList *MinuitFcnGrad::GetInitFloatParamList()
{
   return _initFloatParamList;
}
RooArgList *MinuitFcnGrad::GetInitConstParamList()
{
   return _initConstParamList;
}

void MinuitFcnGrad::SetEvalErrorWall(Bool_t flag)
{
   _doEvalErrorWall = flag;
}
void MinuitFcnGrad::SetPrintEvalErrors(Int_t numEvalErrors)
{
   _printEvalErrors = numEvalErrors;
}

Double_t &MinuitFcnGrad::GetMaxFCN()
{
   return _maxFCN;
}
Int_t MinuitFcnGrad::GetNumInvalidNLL()
{
   return _numBadNLL;
}

Int_t MinuitFcnGrad::evalCounter() const
{
   return _evalCounter;
}
void MinuitFcnGrad::zeroEvalCount()
{
   _evalCounter = 0;
}

void MinuitFcnGrad::SetVerbose(Bool_t flag)
{
   _verbose = flag;
}

void MinuitFcnGrad::SetPdfParamErr(Int_t index, Double_t value)
{
   // Modify PDF parameter error by ordinal index (needed by MINUIT)
   ((RooRealVar *)_floatParamList->at(index))->setError(value);
}

void MinuitFcnGrad::ClearPdfParamAsymErr(Int_t index)
{
   // Modify PDF parameter error by ordinal index (needed by MINUIT)
   ((RooRealVar *)_floatParamList->at(index))->removeAsymError();
}

void MinuitFcnGrad::SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal)
{
   // Modify PDF parameter error by ordinal index (needed by MINUIT)
   ((RooRealVar *)_floatParamList->at(index))->setAsymError(loVal, hiVal);
}

void MinuitFcnGrad::BackProp(const ROOT::Fit::FitResult &results)
{
   // Transfer MINUIT fit results back into RooFit objects

   for (unsigned index = 0; index < NDim(); index++) {
      Double_t value = results.Value(index);
      SetPdfParamVal(index, value);

      // Set the parabolic error
      Double_t err = results.Error(index);
      SetPdfParamErr(index, err);

      Double_t eminus = results.LowerError(index);
      Double_t eplus = results.UpperError(index);

      if (eplus > 0 || eminus < 0) {
         // Store the asymmetric error, if it is available
         SetPdfParamErr(index, eminus, eplus);
      } else {
         // Clear the asymmetric error
         ClearPdfParamAsymErr(index);
      }
   }
}

void MinuitFcnGrad::ApplyCovarianceMatrix(TMatrixDSym &V)
{
   // Apply results of given external covariance matrix. i.e. propagate its errors
   // to all RRV parameter representations and give this matrix instead of the
   // HESSE matrix at the next save() call

   for (unsigned i = 0; i < NDim(); i++) {
      // Skip fixed parameters
      if (GetFloatParamList()->at(i)->isConstant()) {
         continue;
      }
      SetPdfParamErr(i, sqrt(V(i, i)));
   }
}

} // namespace TestStatistics
} // namespace RooFit
