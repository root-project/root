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

#include "MinuitFcnGrad.h"

#include "RooMinimizer.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"
#include "RooNaNPacker.h"

#include <iomanip> // std::setprecision

namespace RooFit {
namespace TestStatistics {

namespace {

class MinuitGradFunctor : public ROOT::Math::IMultiGradFunction {

public:
   MinuitGradFunctor(MinuitFcnGrad const &fcn) : _fcn{fcn} {}

   ROOT::Math::IMultiGradFunction *Clone() const override { return new MinuitGradFunctor(_fcn); }

   unsigned int NDim() const override { return _fcn.getNDim(); }

   void Gradient(const double *x, double *grad) const override { return _fcn.Gradient(x, grad); }

   void GradientWithPrevResult(const double *x, double *grad, double *previous_grad, double *previous_g2,
                               double *previous_gstep) const override
   {
      return _fcn.GradientWithPrevResult(x, grad, previous_grad, previous_g2, previous_gstep);
   }

   bool returnsInMinuit2ParameterSpace() const override { return _fcn.returnsInMinuit2ParameterSpace(); }

private:
   double DoEval(const double *x) const override { return _fcn(x); }

   double DoDerivative(double const * /*x*/, unsigned int /*icoord*/) const override
   {
      throw std::runtime_error("MinuitGradFunctor::DoDerivative is not implemented, please use Gradient instead.");
   }

   MinuitFcnGrad const &_fcn;
};

} // namespace

/** \class MinuitFcnGrad
 *
 * \brief Minuit-RooMinimizer interface which synchronizes parameter data and coordinates evaluation of likelihood
 * (gradient) values
 *
 * This class provides an interface between RooFit and Minuit. It synchronizes parameter values from Minuit, calls
 * calculator classes to evaluate likelihood and likelihood gradient values and returns them to Minuit. The Wrapper
 * objects do the actual calculations. These are constructed inside the MinuitFcnGrad constructor using the RooAbsL
 * likelihood passed in to the constructor, usually directly from RooMinimizer, with which this class is intimately
 * coupled, being a RooAbsMinimizerFcn implementation. MinuitFcnGrad inherits from ROOT::Math::IMultiGradFunction as
 * well, which allows it to be used as the FCN and GRAD parameters Minuit expects.
 *
 * \note The class is not intended for use by end-users. We recommend to either use RooMinimizer with a RooAbsL derived
 * likelihood object, or to use a higher level entry point like RooAbsPdf::fitTo() or RooAbsPdf::createNLL().
 */

/// \param[in] absL The input likelihood.
/// \param[in] context RooMinimizer that creates and owns this class.
/// \param[in] parameters The vector of ParameterSettings objects that describe the parameters used in the Minuit
/// \param[in] likelihoodMode Lmode
/// \param[in] likelihoodGradientMode Lgrad
/// \param[in] verbose true for verbose output
/// Fitter. Note that these must match the set used in the Fitter used by \p context! It can be passed in from
/// RooMinimizer with fitter()->Config().ParamsSettings().
MinuitFcnGrad::MinuitFcnGrad(const std::shared_ptr<RooFit::TestStatistics::RooAbsL> &absL, RooMinimizer *context,
                             std::vector<ROOT::Fit::ParameterSettings> &parameters, LikelihoodMode likelihoodMode,
                             LikelihoodGradientMode likelihoodGradientMode)
   : RooAbsMinimizerFcn(*absL->getParameters(), context),
     _minuitInternalX(getNDim(), 0),
     _minuitExternalX(getNDim(), 0),
     _multiGenFcn{std::make_unique<MinuitGradFunctor>(*this)}
{
   synchronizeParameterSettings(parameters, true);

   _calculationIsClean = std::make_unique<WrapperCalculationCleanFlags>();

   SharedOffset shared_offset;

   if (likelihoodMode == LikelihoodMode::multiprocess &&
       likelihoodGradientMode == LikelihoodGradientMode::multiprocess) {
      _likelihood = LikelihoodWrapper::create(likelihoodMode, absL, _calculationIsClean, shared_offset);
      _likelihoodInGradient =
         LikelihoodWrapper::create(LikelihoodMode::serial, absL, _calculationIsClean, shared_offset);
   } else {
      _likelihood = LikelihoodWrapper::create(likelihoodMode, absL, _calculationIsClean, shared_offset);
      _likelihoodInGradient = _likelihood;
   }

   _gradient = LikelihoodGradientWrapper::create(likelihoodGradientMode, absL, _calculationIsClean, getNDim(), _context,
                                                 shared_offset);

   applyToLikelihood([&](auto &l) { l.synchronizeParameterSettings(parameters); });
   _gradient->synchronizeParameterSettings(getMultiGenFcn(), parameters);

   // Note: can be different than RooGradMinimizerFcn/LikelihoodGradientSerial, where default options are passed
   // (ROOT::Math::MinimizerOptions::DefaultStrategy() and ROOT::Math::MinimizerOptions::DefaultErrorDef())
   applyToLikelihood([&](auto &l) { l.synchronizeWithMinimizer(ROOT::Math::MinimizerOptions()); });
   _gradient->synchronizeWithMinimizer(ROOT::Math::MinimizerOptions());
}

/// Make sure the offsets are up to date
///
/// If the offsets need to be updated, this function triggers a likelihood evaluation.
/// The likelihood will make sure the offset is set correctly in their shared_ptr
/// offsets object, that is also shared with possible other LikelihoodWrapper members
/// of MinuitFcnGrad and also the LikelihoodGradientWrapper member. Other necessary
/// synchronization steps are also performed from the Wrapper child classes (e.g.
/// sending the values to workers from MultiProcess::Jobs).
void MinuitFcnGrad::syncOffsets() const
{
   if (_likelihood->isOffsetting() && (_evalCounter == 0 || offsets_reset_)) {
      _likelihoodInGradient->evaluate();
      offsets_reset_ = false;
   }
}

double MinuitFcnGrad::operator()(const double *x) const
{
   syncParameterValuesFromMinuitCalls(x, false);

   syncOffsets();

   // Calculate the function for these parameters
   auto &likelihoodHere(_likelihoodInGradient && _gradient->isCalculating() ? *_likelihoodInGradient : *_likelihood);
   likelihoodHere.evaluate();
   double fvalue = likelihoodHere.getResult().Sum();
   _calculationIsClean->likelihood = true;

   fvalue = applyEvalErrorHandling(fvalue);

   // Optional logging
   if (cfg().verbose) {
      std::cout << "\nprevFCN" << (likelihoodHere.isOffsetting() ? "-offset" : "") << " = " << std::setprecision(10)
                << fvalue << std::setprecision(4) << "  ";
      std::cout.flush();
   }

   finishDoEval();
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
/// minuit_internal = false and the values do not match the previous values stored in _minuitInternalX *but* the
/// values after SetPdfParamVal did not get set to the intended value, the _minuitInternalRooFitXMismatch flag is
/// set. This information can be used by calculators, if desired, for instance when a calculator does not want to make
/// use of the range information in the RooAbsArg parameters.
bool MinuitFcnGrad::syncParameterValuesFromMinuitCalls(const double *x, bool minuit_internal) const
{
   bool aParamWasUpdated = false;
   if (minuit_internal) {
      if (!returnsInMinuit2ParameterSpace()) {
         throw std::logic_error("Updating Minuit-internal parameters only makes sense for (gradient) calculators that "
                                "are defined in Minuit-internal parameter space.");
      }

      for (std::size_t ix = 0; ix < getNDim(); ++ix) {
         bool parameter_changed = (x[ix] != _minuitInternalX[ix]);
         if (parameter_changed) {
            _minuitInternalX[ix] = x[ix];
         }
         aParamWasUpdated |= parameter_changed;
      }

      if (aParamWasUpdated) {
         _calculationIsClean->set_all(false);
         applyToLikelihood([&](auto &l) { l.updateMinuitInternalParameterValues(_minuitInternalX); });
         _gradient->updateMinuitInternalParameterValues(_minuitInternalX);
      }
   } else {
      bool aParamIsMismatched = false;

      for (std::size_t ix = 0; ix < getNDim(); ++ix) {
         // Note: the return value of SetPdfParamVal does not always mean that the parameter's value in the RooAbsReal
         // changed since last time! If the value was out of range bin, setVal was still called, but the value was not
         // updated.
         SetPdfParamVal(ix, x[ix]);
         _minuitExternalX[ix] = x[ix];
         // The above is why we need _minuitExternalX. The _minuitExternalX vector can also be passed to
         // LikelihoodWrappers, if needed, but typically they will make use of the RooFit parameters directly. However,
         // we log in the flag below whether they are different so that calculators can use this information.
         bool parameter_changed = (x[ix] != _minuitExternalX[ix]);
         aParamWasUpdated |= parameter_changed;
         aParamIsMismatched |= (floatableParam(ix).getVal() != _minuitExternalX[ix]);
      }

      _minuitInternalRooFitXMismatch = aParamIsMismatched;

      if (aParamWasUpdated) {
         _calculationIsClean->set_all(false);
         applyToLikelihood([&](auto &l) { l.updateMinuitExternalParameterValues(_minuitExternalX); });
         _gradient->updateMinuitExternalParameterValues(_minuitExternalX);
      }
   }
   return aParamWasUpdated;
}

void MinuitFcnGrad::Gradient(const double *x, double *grad) const
{
   _calculatingGradient = true;
   syncParameterValuesFromMinuitCalls(x, returnsInMinuit2ParameterSpace());
   syncOffsets();
   _gradient->fillGradient(grad);
   _calculatingGradient = false;
}

void MinuitFcnGrad::GradientWithPrevResult(const double *x, double *grad, double *previous_grad, double *previous_g2,
                                           double *previous_gstep) const
{
   _calculatingGradient = true;
   syncParameterValuesFromMinuitCalls(x, returnsInMinuit2ParameterSpace());
   syncOffsets();
   _gradient->fillGradientWithPrevResult(grad, previous_grad, previous_g2, previous_gstep);
   _calculatingGradient = false;
}

bool MinuitFcnGrad::Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters)
{
   bool returnee = synchronizeParameterSettings(parameters, _optConst);
   applyToLikelihood([&](auto &l) { l.synchronizeParameterSettings(parameters); });
   _gradient->synchronizeParameterSettings(parameters);

   applyToLikelihood([&](auto &l) { l.synchronizeWithMinimizer(_context->fitter()->Config().MinimizerOptions()); });
   _gradient->synchronizeWithMinimizer(_context->fitter()->Config().MinimizerOptions());
   return returnee;
}

} // namespace TestStatistics
} // namespace RooFit
