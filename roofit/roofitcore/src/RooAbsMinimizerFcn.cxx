/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl   *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// RooAbsMinimizerFcn is an interface class to the ROOT::Math function
// for minimization. It contains only the "logistics" of synchronizing
// between Minuit and RooFit. Its subclasses implement actual interfacing
// to Minuit by subclassing IMultiGenFunction or IMultiGradFunction.
//

#include "RooAbsMinimizerFcn.h"

#include "RooAbsArg.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooMsgService.h"
#include "RooNaNPacker.h"

#include "TClass.h"
#include "TMatrixDSym.h"

#include <fstream>
#include <iomanip>

RooAbsMinimizerFcn::RooAbsMinimizerFcn(RooArgList paramList, RooMinimizer *context) : _context{context}
{
   _allParams.add(paramList);

   RooArgList initFloatParams;

   // Examine parameter list
   for (RooAbsArg *param : _allParams) {

      // Treat all non-RooRealVar parameters as constants (MINUIT cannot handle them)
      if (!param->isConstant() && !canBeFloating(*param)) {
         oocoutW(_context, Minimization) << "RooAbsMinimizerFcn::RooAbsMinimizerFcn: removing parameter "
                                         << param->GetName() << " from list because it is not of type RooRealVar"
                                         << std::endl;
      }
   }

   _allParams.snapshot(_allParamsInit, false);

   std::size_t iParam = 0;
   for (RooAbsArg *param : _allParamsInit) {
      if (!treatAsConstant(*param)) {
         _floatableParamIndices.push_back(iParam);
      }
      ++iParam;
   }
}

/// Internal function to synchronize TMinimizer with current
/// information in RooAbsReal function parameters
bool RooAbsMinimizerFcn::synchronizeParameterSettings(std::vector<ROOT::Fit::ParameterSettings> &parameters,
                                                      bool optConst)
{
   // Synchronize MINUIT with function state

   for (std::size_t i = 0; i < _allParams.size(); ++i) {
      if (treatAsConstant(_allParamsInit[i]) && !treatAsConstant(_allParams[i])) {
         std::stringstream ss;
         ss << "RooMinimzer: the parameter named " << _allParams[i].GetName()
            << " is not constant anymore, but it was constant at the time where the RooMinimizer was constructed."
               " This is illegal. The other way around is supported: you can always change the constant flag of "
               "parameters that were floating at the time the minimizer was instantiated.";
         oocxcoutF(nullptr, LinkStateMgmt) << ss.str() << std::endl;
         throw std::runtime_error(ss.str());
      }
   }

   std::vector<ROOT::Fit::ParameterSettings> oldParameters = parameters;
   parameters.clear();

   for (std::size_t index = 0; index < getNDim(); index++) {

      auto &par = floatableParam(index);

      // make sure the parameter are in dirty state to enable
      // a real NLL computation when the minimizer calls the function the first time
      // (see issue #7659)
      par.setValueDirty();

      // Set the limits, if not infinite
      double pmin = par.hasMin() ? par.getMin() : 0.0;
      double pmax = par.hasMax() ? par.getMax() : 0.0;

      // Calculate step size
      double pstep = par.getError();
      if (pstep <= 0) {
         // Floating parameter without error estimate
         if (par.hasMin() && par.hasMax()) {
            pstep = 0.1 * (pmax - pmin);

            // Trim default choice of error if within 2 sigma of limit
            if (pmax - par.getVal() < 2 * pstep) {
               pstep = (pmax - par.getVal()) / 2;
            } else if (par.getVal() - pmin < 2 * pstep) {
               pstep = (par.getVal() - pmin) / 2;
            }

            // If trimming results in zero error, restore default
            if (pstep == 0) {
               pstep = 0.1 * (pmax - pmin);
            }

         } else {
            pstep = 1;
         }
         if (cfg().verbose) {
            oocoutW(_context, Minimization)
               << "RooAbsMinimizerFcn::synchronize: WARNING: no initial error estimate available for " << par.GetName()
               << ": using " << pstep << std::endl;
         }
      }

      if (par.hasMin() && par.hasMax()) {
         parameters.emplace_back(par.GetName(), par.getVal(), pstep, pmin, pmax);
      } else {
         parameters.emplace_back(par.GetName(), par.getVal(), pstep);
         if (par.hasMin()) {
            parameters.back().SetLowerLimit(pmin);
         } else if (par.hasMax()) {
            parameters.back().SetUpperLimit(pmax);
         }
      }

      par.isConstant() ? parameters.back().Fix() : parameters.back().Release();
   }

   if (optConst) {
      bool constStateChange = false;
      bool constValChange = false;
      for (std::size_t i = 0; i < oldParameters.size(); ++i) {
         auto const &newParam = parameters[i];
         auto const &oldParam = oldParameters[i];
         constStateChange &= (newParam.IsFixed() != oldParam.IsFixed());
         constValChange &= (newParam.IsFixed() && (newParam.Value() != oldParam.Value()));
      }
      optimizeConstantTerms(constStateChange, constValChange);
   }

   return false;
}

bool RooAbsMinimizerFcn::Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters)
{
   return synchronizeParameterSettings(parameters, _optConst);
}

/// Transfer MINUIT fit results back into RooFit objects.
void RooAbsMinimizerFcn::BackProp()
{
   auto const &results = _context->fitter()->Result();

   for (std::size_t index = 0; index < getNDim(); index++) {

      auto &param = floatableParam(index);

      double value = results.fParams[index];
      SetPdfParamVal(index, value);

      // Set the parabolic error
      double err = results.fErrors[index];
      param.setError(err);

      double eminus = results.lowerError(index);
      double eplus = results.upperError(index);

      if (eplus > 0 || eminus < 0) {
         // Store the asymmetric error, if it is available
         param.setAsymError(eminus, eplus);
      } else {
         // Clear the asymmetric error
         param.removeAsymError();
      }
   }
}

/// Change the file name for logging of a RooMinimizer of all MINUIT steppings
/// through the parameter space. If inLogfile is null, the current log file
/// is closed and logging is stopped.
bool RooAbsMinimizerFcn::SetLogFile(const char *inLogfile)
{
   if (_logfile) {
      oocoutI(_context, Minimization) << "RooAbsMinimizerFcn::setLogFile: closing previous log file" << std::endl;
      _logfile->close();
      delete _logfile;
      _logfile = nullptr;
   }
   _logfile = new std::ofstream(inLogfile);
   if (!_logfile->good()) {
      oocoutI(_context, Minimization) << "RooAbsMinimizerFcn::setLogFile: cannot open file " << inLogfile << std::endl;
      _logfile->close();
      delete _logfile;
      _logfile = nullptr;
   }

   return false;
}

/// Apply results of given external covariance matrix. i.e. propagate its errors
/// to all RRV parameter representations and give this matrix instead of the
/// HESSE matrix at the next save() call
void RooAbsMinimizerFcn::ApplyCovarianceMatrix(TMatrixDSym &V)
{
   for (unsigned int i = 0; i < getNDim(); i++) {
      floatableParam(i).setError(std::sqrt(V(i, i)));
   }
}

/// Set value of parameter i.
bool RooAbsMinimizerFcn::SetPdfParamVal(int index, double value) const
{
   auto &par = floatableParam(index);

   if (par.getVal() != value) {
      if (cfg().verbose)
         std::cout << par.GetName() << "=" << value << ", ";

      par.setVal(value);
      return true;
   }

   return false;
}

/// Print information about why evaluation failed.
/// Using _printEvalErrors, the number of errors printed can be steered.
/// Negative values disable printing.
void RooAbsMinimizerFcn::printEvalErrors() const
{
   if (cfg().printEvalErrors < 0)
      return;

   std::ostringstream msg;
   if (cfg().doEEWall) {
      msg << "RooAbsMinimizerFcn: Minimized function has error status." << std::endl
          << "Returning maximum FCN so far (" << _maxFCN
          << ") to force MIGRAD to back out of this region. Error log follows.\n";
   } else {
      msg << "RooAbsMinimizerFcn: Minimized function has error status but is ignored.\n";
   }

   msg << "Parameter values: ";
   for (std::size_t i = 0; i < getNDim(); ++i) {
      auto &var = floatableParam(i);
      msg << "\t" << var.GetName() << "=" << var.getVal();
   }
   msg << std::endl;

   RooAbsReal::printEvalErrors(msg, cfg().printEvalErrors);
   ooccoutW(_context, Minimization) << msg.str() << std::endl;
}

/// Apply corrections on the fvalue if errors were signaled.
///
/// Two kinds of errors are possible: 1. infinite or nan values (the latter
/// can be a signaling nan, using RooNaNPacker) or 2. logEvalError-type errors.
/// Both are caught here and fvalue is updated so that Minuit in turn is nudged
/// to move the search outside of the problematic parameter space area.
double RooAbsMinimizerFcn::applyEvalErrorHandling(double fvalue) const
{
   if (!std::isfinite(fvalue) || RooAbsReal::numEvalErrors() > 0 || fvalue > 1e30) {
      printEvalErrors();
      RooAbsReal::clearEvalErrorLog();
      _numBadNLL++;

      if (cfg().doEEWall) {
         const double badness = RooNaNPacker::unpackNaN(fvalue);
         fvalue = (std::isfinite(_maxFCN) ? _maxFCN : 0.) + cfg().recoverFromNaN * badness;
      }
   } else {
      if (_evalCounter > 0 && _evalCounter == _numBadNLL) {
         // This is the first time we get a valid function value; while before, the
         // function was always invalid. For invalid  cases, we returned values > 0.
         // Now, we offset valid values such that they are < 0.
         _funcOffset = -fvalue;
      }
      fvalue += _funcOffset;
      _maxFCN = std::max(fvalue, _maxFCN);
   }
   return fvalue;
}

void RooAbsMinimizerFcn::finishDoEval() const
{
   _evalCounter++;
}

void RooAbsMinimizerFcn::setOptimizeConst(int flag)
{
   auto ctx = _context->makeEvalErrorContext();

   if (_optConst && !flag) {
      if (_context->getPrintLevel() > -1) {
         oocoutI(_context, Minimization) << "RooAbsMinimizerFcn::setOptimizeConst: deactivating const optimization"
                                         << std::endl;
      }
      setOptimizeConstOnFunction(RooAbsArg::DeActivate, true);
      _optConst = flag;
   } else if (!_optConst && flag) {
      if (_context->getPrintLevel() > -1) {
         oocoutI(_context, Minimization) << "RooAbsMinimizerFcn::setOptimizeConst: activating const optimization"
                                         << std::endl;
      }
      setOptimizeConstOnFunction(RooAbsArg::Activate, flag > 1);
      _optConst = flag;
   } else if (_optConst && flag) {
      if (_context->getPrintLevel() > -1) {
         oocoutI(_context, Minimization) << "RooAbsMinimizerFcn::setOptimizeConst: const optimization already active"
                                         << std::endl;
      }
   } else {
      if (_context->getPrintLevel() > -1) {
         oocoutI(_context, Minimization) << "RooAbsMinimizerFcn::setOptimizeConst: const optimization wasn't active"
                                         << std::endl;
      }
   }
}

void RooAbsMinimizerFcn::optimizeConstantTerms(bool constStatChange, bool constValChange)
{
   auto ctx = _context->makeEvalErrorContext();

   if (constStatChange) {

      oocoutI(_context, Minimization)
         << "RooAbsMinimizerFcn::optimizeConstantTerms: set of constant parameters changed, rerunning const optimizer"
         << std::endl;
      setOptimizeConstOnFunction(RooAbsArg::ConfigChange, true);
   } else if (constValChange) {
      oocoutI(_context, Minimization)
         << "RooAbsMinimizerFcn::optimizeConstantTerms: constant parameter values changed, rerunning const optimizer"
         << std::endl;
      setOptimizeConstOnFunction(RooAbsArg::ValueChange, true);
   }
}

RooArgList RooAbsMinimizerFcn::floatParams() const
{
   RooArgList out;
   for (RooAbsArg *param : _allParams) {
      if (!treatAsConstant(*param))
         out.add(*param);
   }
   return out;
}

RooArgList RooAbsMinimizerFcn::constParams() const
{
   RooArgList out;
   for (RooAbsArg *param : _allParams) {
      if (treatAsConstant(*param))
         out.add(*param);
   }
   return out;
}

RooArgList RooAbsMinimizerFcn::initFloatParams() const
{
   RooArgList initFloatableParams;

   for (RooAbsArg *param : _allParamsInit) {
      if (!treatAsConstant(*param))
         initFloatableParams.add(*param);
   }

   // Make sure we only return the initial parameters
   // corresponding to currently floating parameters.
   RooArgList out;
   initFloatableParams.selectCommon(floatParams(), out);

   return out;
}
