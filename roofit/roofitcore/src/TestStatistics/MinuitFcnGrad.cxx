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

#include "TestStatistics/MinuitFcnGrad.h"

#include "RooMinimizer.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"

#include <iomanip>  // std::setprecision

namespace RooFit {
namespace TestStatistics {

// IMultiGradFunction overrides necessary for Minuit: DoEval, Gradient
// The likelihood and gradient wrappers do the actual calculations.

double MinuitFcnGrad::DoEval(const double *x) const
{
   Bool_t parameters_changed = syncParameterValuesFromMinuitCalls(x, false);

   // Calculate the function for these parameters
//   RooAbsReal::setHideOffset(kFALSE);
   likelihood->evaluate();
   double fvalue = likelihood->getResult();
   calculation_is_clean->likelihood = true;
//   RooAbsReal::setHideOffset(kTRUE);

   if (!parameters_changed) {
      return fvalue;
   }

   if (!std::isfinite(fvalue) || RooAbsReal::numEvalErrors() > 0 || fvalue > 1e30) {

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

         Bool_t first(kTRUE);
         ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << "Parameter values: ";
         for (const auto rooAbsArg : *_floatParamList) {
            auto var = static_cast<const RooRealVar*>(rooAbsArg);
            if (first) {
               first = kFALSE;
            } else {
               ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << ", ";
            }
            ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << var->GetName() << "=" << var->getVal();
         }
         ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << std::endl;

         RooAbsReal::printEvalErrors(ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval), _printEvalErrors);
         ooccoutW(static_cast<RooAbsArg *>(nullptr), Eval) << std::endl;
      }

      if (_doEvalErrorWall) {
         fvalue = _maxFCN + 1;
      }

      RooAbsReal::clearEvalErrorLog();
      _numBadNLL++;
   } else if (fvalue > _maxFCN) {
      _maxFCN = fvalue;
   }

   // Optional logging
   if (_verbose) {
      std::cout << "\nprevFCN" << (likelihood->isOffsetting() ? "-offset" : "") << " = " << std::setprecision(10)
                << fvalue << std::setprecision(4) << "  ";
      std::cout.flush();
   }

   _evalCounter++;
   return fvalue;
}

/// Minuit calls (via FcnAdapters etc) DoEval or Gradient with a set of parameters x.
/// This function syncs these values to the proper places in RooFit.
///
/// The first twist, and reason this function is more complicated than one may imagine, is that Minuit internally uses a
/// transformed parameter space to account for parameter boundaries. Whether we receive these Minuit "internal"
/// parameter values or "regular"/untransformed RooFit parameter space values depends on the situation.
/// - The values that arrive here via DoEval are always "normal" parameter values, since Minuit transforms these
///   back into regular space before passing to DoEval (see MnUserFcn::operator() which wraps the Fcn(Gradient)Base
///   in ModularFunctionMinimizer::Minimize and is used for direct function calls from that point on in the minimizer).
///   These can thus always be safely synced with this function's RooFit parameters using SetPdfParamVal.
/// - The values that arrive here via Gradient will be in internal coordinates if that is
///   what this class expects, and indeed this is the case for MinuitFcnGrad's current implementation. This is
///   communicated to Minuit via MinuitFcnGrad::returnsInMinuit2ParameterSpace. Inside Minuit, that function determines
///   whether this class's gradient calculator is wrapped inside a AnalyticalGradientCalculator, to which Minuit passes
///   "external" parameter values, or as an ExternalInternalGradientCalculator, which gets "internal" parameter values.
///   Long story short: when MinuitFcnGrad::returnsInMinuit2ParameterSpace() returns true, Minuit will pass "internal"
///   values to Gradient. These cannot be synced with this function's RooFit parameters using
///   SetPdfParamVal, unless a manual transformation step is performed in advance. However, they do need to be passed
///   on to the gradient calculator, since indeed we expect values there to be in "internal" space. However, this is
///   calculator dependent. Note that in the current MinuitFcnGrad implementation we do not actually allow for
///   calculators in "external" (i.e. regular RooFit parameter space) values, since
///   MinuitFcnGrad::returnsInMinuit2ParameterSpace is hardcoded to true. This should in a future version be changed so
///   that the calculator (the wrapper) is queried for this information.
/// Because some gradient calculators may also use the regular RooFit parameters (e.g. for calculating the likelihood's
/// value itself), this information is also passed on to the gradient wrapper. Vice versa, when updated "internal"
/// parameters are passed to Gradient, the likelihood may be affected as well. Even though a
/// transformation from internal to "external" may be necessary before the values can be used, the likelihood can at
/// least log that its parameter values are possibly no longer in sync with those of the gradient.
///
/// The second twist is that the Minuit external parameters may still be different from the ones used in RooFit. This
/// happens when Minuit tries out values that lay outside the RooFit parameter's range(s). RooFit's setVal (called
/// inside SetPdfParamVal) then clips the RooAbsArg's value to one of the range limits, instead of setting it to the
/// value Minuit intended. When this happens, i.e. syncParameterValuesFromMinuitCalls is called with
/// minuit_internal = false and the values do not match the previous values stored in minuit_internal_x_ *but* the
/// values after SetPdfParamVal did not get set to the intended value, the minuit_internal_roofit_x_mismatch_ flag is
/// set. This information can be used by calculators, if desired, for instance when a calculator does not want to make
/// use of the range information in the RooAbsArg parameters.
bool MinuitFcnGrad::syncParameterValuesFromMinuitCalls(const double *x, bool minuit_internal) const
{
   bool a_parameter_has_been_updated = false;
   if (minuit_internal) {
      if (!returnsInMinuit2ParameterSpace()) {
         throw std::logic_error("Updating Minuit-internal parameters only makes sense for (gradient) calculators that are defined in Minuit-internal parameter space.");
      }

      for (std::size_t ix = 0; ix < NDim(); ++ix) {
         bool parameter_changed = (x[ix] != minuit_internal_x_[ix]);
         if (parameter_changed) {
            minuit_internal_x_[ix] = x[ix];
         }
         a_parameter_has_been_updated |= parameter_changed;
      }

      if(a_parameter_has_been_updated) {
         calculation_is_clean->set_all(false);
         likelihood->updateMinuitInternalParameterValues(minuit_internal_x_);
         gradient->updateMinuitInternalParameterValues(minuit_internal_x_);
      }
   } else {
      bool a_parameter_is_mismatched = false;

      for (std::size_t ix = 0; ix < NDim(); ++ix) {
         // Note: the return value of SetPdfParamVal does not always mean that the parameter's value in the RooAbsReal changed since last
         // time! If the value was out of range bin, setVal was still called, but the value was not updated.
         SetPdfParamVal(ix, x[ix]);
         minuit_external_x_[ix] = x[ix];
         // The above is why we need minuit_external_x_. The minuit_external_x_ vector can also be passed to
         // LikelihoodWrappers, if needed, but typically they will make use of the RooFit parameters directly. However,
         // we log in the flag below whether they are different so that calculators can use this information.
         bool parameter_changed = (x[ix] != minuit_external_x_[ix]);
         a_parameter_has_been_updated |= parameter_changed;
         a_parameter_is_mismatched |= (((RooRealVar *)_floatParamList->at(ix))->getVal() != minuit_external_x_[ix]);
      }

      minuit_internal_roofit_x_mismatch_ = a_parameter_is_mismatched;

      if(a_parameter_has_been_updated) {
         calculation_is_clean->set_all(false);
         likelihood->updateMinuitExternalParameterValues(minuit_external_x_);
         gradient->updateMinuitExternalParameterValues(minuit_external_x_);
      }
   }
   return a_parameter_has_been_updated;
}


void MinuitFcnGrad::Gradient(const double *x, double *grad) const
{
   syncParameterValuesFromMinuitCalls(x, returnsInMinuit2ParameterSpace());
   gradient->fillGradient(grad);
}

double MinuitFcnGrad::DoDerivative(const double * /*x*/, unsigned int /*icoord*/) const
{
   throw std::runtime_error("MinuitFcnGrad::DoDerivative is not implemented, please use Gradient instead.");
}

Bool_t
MinuitFcnGrad::Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters, Bool_t optConst, Bool_t verbose)
{
   Bool_t returnee = synchronizeParameterSettings(parameters, optConst, verbose);
   likelihood->synchronizeParameterSettings(parameters);
   gradient->synchronizeParameterSettings(parameters);

   likelihood->synchronizeWithMinimizer(_context->fitter()->Config().MinimizerOptions());
   gradient->synchronizeWithMinimizer(_context->fitter()->Config().MinimizerOptions());
   return returnee;
}

} // namespace TestStatistics
} // namespace RooFit
