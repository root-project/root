/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *   Emmanouil Michalainas, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\file RooNLLVarNew.cxx
\class RooNLLVarNew
\ingroup Roofitcore

This is a simple class designed to produce the nll values needed by the fitter.
This class calls functions from `RooBatchCompute` library to provide faster
computation times.
**/

#include "RooFit/Detail/RooNLLVarNew.h"

#include <RooHistPdf.h>
#include <RooBatchCompute.h>
#include <RooDataHist.h>
#include <RooNaNPacker.h>
#include <RooConstVar.h>
#include <RooRealVar.h>
#include <RooSetProxy.h>
#include <RooFit/Detail/MathFuncs.h>

#include "RooFitImplHelpers.h"

#include <ROOT/StringUtils.hxx>

#include <TClass.h>
#include <TMath.h>
#include <Math/Util.h>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace RooFit::Detail {

// Declare constexpr static members to make them available if odr-used in C++14.
constexpr const char *RooNLLVarNew::weightVarName;
constexpr const char *RooNLLVarNew::weightVarNameSumW2;
constexpr const char *RooNLLVarNew::binVolumeVarName;
constexpr const char *RooNLLVarNew::weightErrorLoVarName;
constexpr const char *RooNLLVarNew::weightErrorHiVarName;

namespace {

// Use RooConstVar for dummies such that they don't get included in getParameters().
std::unique_ptr<RooConstVar> dummyVar(const char *name)
{
   return std::make_unique<RooConstVar>(name, name, 1.0);
}

// Helper class to represent a template pdf based on the fit dataset.
class RooOffsetPdf : public RooAbsPdf {
public:
   RooOffsetPdf(const char *name, const char *title, RooArgSet const &observables, RooAbsReal &weightVar)
      : RooAbsPdf(name, title),
        _observables("!observables", "List of observables", this),
        _weightVar{"!weightVar", "weightVar", this, weightVar, true, false}
   {
      for (RooAbsArg *obs : observables) {
         _observables.add(*obs);
      }
   }
   RooOffsetPdf(const RooOffsetPdf &other, const char *name = nullptr)
      : RooAbsPdf(other, name),
        _observables("!servers", this, other._observables),
        _weightVar{"!weightVar", this, other._weightVar}
   {
   }
   TObject *clone(const char *newname) const override { return new RooOffsetPdf(*this, newname); }

   void doEval(RooFit::EvalContext &ctx) const override
   {
      std::span<double> output = ctx.output();
      std::size_t nEvents = output.size();

      std::span<const double> weights = ctx.at(_weightVar);

      // Create the template histogram from the data. This operation is very
      // expensive, but since the offset only depends on the observables it
      // only has to be done once.

      RooDataHist dataHist{"data", "data", _observables};
      // Loop over events to fill the histogram
      for (std::size_t i = 0; i < nEvents; ++i) {
         for (auto *var : static_range_cast<RooRealVar *>(_observables)) {
            var->setVal(ctx.at(var)[i]);
         }
         dataHist.add(_observables, weights[weights.size() == 1 ? 0 : i]);
      }

      // Lookup bin weights via RooHistPdf
      RooHistPdf pdf{"offsetPdf", "offsetPdf", _observables, dataHist};
      for (std::size_t i = 0; i < nEvents; ++i) {
         for (auto *var : static_range_cast<RooRealVar *>(_observables)) {
            var->setVal(ctx.at(var)[i]);
         }
         output[i] = pdf.getVal(_observables);
      }
   }

private:
   double evaluate() const override { return 0.0; } // should never be called

   RooSetProxy _observables;
   RooTemplateProxy<RooAbsReal> _weightVar;
};

} // namespace

/// Construct either an NLL or a chi-squared test statistic.
/// \param func The pdf or function to evaluate. For `Statistic::NLL` a
/// RooAbsPdf is required, and for `Statistic::Chi2` any RooAbsReal is accepted.
RooNLLVarNew::RooNLLVarNew(const char *name, const char *title, RooAbsReal &func, RooArgSet const &observables,
                           Config const &cfg)
   : RooAbsReal(name, title),
     _func{"func", "func", this, func},
     _weightVar{"weightVar", "weightVar", this, dummyVar(weightVarName)},
     _weightSquaredVar{weightVarNameSumW2, weightVarNameSumW2, this, dummyVar("weightSquardVar")},
     _statistic{cfg.statistic},
     _chi2ErrorType{cfg.chi2ErrorType}
{
   auto *pdf = dynamic_cast<RooAbsPdf *>(&func);

   if (_statistic == Statistic::Chi2) {
      // Signal to RooEvaluatorWrapper::setData that zero-weight bins must be
      // retained (chi2 needs every bin's prediction, even where the data is
      // empty).
      setAttribute("Chi2EvaluationActive");
      _funcMode = !pdf ? FuncMode::Function : (cfg.extended ? FuncMode::ExtendedPdf : FuncMode::Pdf);
   } else {
      _binnedL = pdf && pdf->getAttribute("BinnedLikelihoodActive");
   }

   RooArgSet obs;
   func.getObservables(&observables, obs);

   // Extended mode needs an expected-events function for both NLL and chi2
   // (NLL adds it as an extra additive term, chi2 uses it as the predicted
   // yield normalisation). Skip it for binned NLL (where the yields come
   // directly from the pdf) and for chi2 Function mode (where the function
   // values are directly the predicted yields).
   const bool wantsExpectedEvents = pdf && ((_statistic == Statistic::NLL && cfg.extended && !_binnedL) ||
                                            (_statistic == Statistic::Chi2 && _funcMode == FuncMode::ExtendedPdf));
   if (wantsExpectedEvents) {
      std::unique_ptr<RooAbsReal> expectedEvents = pdf->createExpectedEventsFunc(&obs);
      if (expectedEvents) {
         _expectedEvents =
            std::make_unique<RooTemplateProxy<RooAbsReal>>("expectedEvents", "expectedEvents", this, *expectedEvents);
         addOwnedComponents(std::move(expectedEvents));
      }
   }

   if (_statistic == Statistic::NLL) {
      // In the "BinnedLikelihoodActiveYields" mode, the pdf values can
      // directly be interpreted as yields and don't need to be multiplied by
      // the bin widths. That's why we don't need to even fill them in this
      // case.
      if (_binnedL && !pdf->getAttribute("BinnedLikelihoodActiveYields")) {
         fillBinWidthsFromPdfBoundaries(*pdf, obs);
      }

      enableOffsetting(cfg.offsetMode == RooFit::OffsetMode::Initial);
      enableBinOffsetting(cfg.offsetMode == RooFit::OffsetMode::Bin);

      // In the binned likelihood code path, we directly use that data weights
      // for the offsetting.
      if (!_binnedL && _doBinOffset) {
         auto offsetPdf = std::make_unique<RooOffsetPdf>("_offset_func", "_offset_func", obs, *_weightVar);
         _offsetPdf = std::make_unique<RooTemplateProxy<RooAbsPdf>>("offsetPdf", "offsetPdf", this, *offsetPdf);
         addOwnedComponents(std::move(offsetPdf));
      }
   } else {
      // Chi2-only proxies: per-bin volumes, plus per-bin asymmetric errors
      // when Poisson error mode is requested.
      auto binVolumeDummy = std::make_unique<RooConstVar>(binVolumeVarName, binVolumeVarName, 1.0);
      _binVolumes = std::make_unique<RooTemplateProxy<RooAbsReal>>(binVolumeVarName, binVolumeVarName, this,
                                                                   *binVolumeDummy, true, false);
      addOwnedComponents(std::move(binVolumeDummy));

      if (_chi2ErrorType == RooDataHist::Poisson) {
         auto errLoDummy = std::make_unique<RooConstVar>(weightErrorLoVarName, weightErrorLoVarName, 1.0);
         auto errHiDummy = std::make_unique<RooConstVar>(weightErrorHiVarName, weightErrorHiVarName, 1.0);
         _weightErrLo = std::make_unique<RooTemplateProxy<RooAbsReal>>(weightErrorLoVarName, weightErrorLoVarName, this,
                                                                       *errLoDummy, true, false);
         _weightErrHi = std::make_unique<RooTemplateProxy<RooAbsReal>>(weightErrorHiVarName, weightErrorHiVarName, this,
                                                                       *errHiDummy, true, false);
         addOwnedComponents(std::move(errLoDummy));
         addOwnedComponents(std::move(errHiDummy));
      }
   }

   resetWeightVarNames();
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name),
     _func{"func", this, other._func},
     _weightVar{"weightVar", this, other._weightVar},
     _weightSquaredVar{"weightSquaredVar", this, other._weightSquaredVar},
     _weightSquared{other._weightSquared},
     _binnedL{other._binnedL},
     _doOffset{other._doOffset},
     _doBinOffset{other._doBinOffset},
     _statistic{other._statistic},
     _funcMode{other._funcMode},
     _chi2ErrorType{other._chi2ErrorType},
     _simCount{other._simCount},
     _prefix{other._prefix},
     _binw{other._binw}
{
   if (other._expectedEvents) {
      _expectedEvents = std::make_unique<RooTemplateProxy<RooAbsReal>>("expectedEvents", this, *other._expectedEvents);
   }
   if (other._binVolumes) {
      _binVolumes = std::make_unique<RooTemplateProxy<RooAbsReal>>(binVolumeVarName, this, *other._binVolumes);
   }
   if (other._weightErrLo) {
      _weightErrLo = std::make_unique<RooTemplateProxy<RooAbsReal>>(weightErrorLoVarName, this, *other._weightErrLo);
   }
   if (other._weightErrHi) {
      _weightErrHi = std::make_unique<RooTemplateProxy<RooAbsReal>>(weightErrorHiVarName, this, *other._weightErrHi);
   }
}

void RooNLLVarNew::fillBinWidthsFromPdfBoundaries(RooAbsReal const &pdf, RooArgSet const &observables)
{
   // Check if the bin widths were already filled
   if (!_binw.empty()) {
      return;
   }

   if (observables.size() != 1) {
      throw std::runtime_error("BinnedPdf optimization only works with a 1D pdf.");
   } else {
      auto *var = static_cast<RooRealVar *>(observables.first());
      std::list<double> *boundaries = pdf.binBoundaries(*var, var->getMin(), var->getMax());
      std::list<double>::iterator biter = boundaries->begin();
      _binw.resize(boundaries->size() - 1);
      double lastBound = (*biter);
      ++biter;
      int ibin = 0;
      while (biter != boundaries->end()) {
         _binw[ibin] = (*biter) - lastBound;
         lastBound = (*biter);
         ibin++;
         ++biter;
      }
   }
}

void RooNLLVarNew::doEvalBinnedL(RooFit::EvalContext &ctx, std::span<const double> preds,
                                 std::span<const double> weights) const
{
   ROOT::Math::KahanSum<double> result{0.0};
   ROOT::Math::KahanSum<double> sumWeightKahanSum{0.0};

   const bool predsAreYields = _binw.empty();

   for (std::size_t i = 0; i < preds.size(); ++i) {

      // Calculate log(Poisson(N|mu) for this bin
      double N = weights[i];
      double mu = preds[i];
      if (!predsAreYields) {
         mu *= _binw[i];
      }

      if (mu <= 0 && N > 0) {
         // Catch error condition: data present where zero events are predicted
         logEvalError(Form("Observed %f events in bin %lu with zero event yield", N, (unsigned long)i));
      } else {
         result += RooFit::Detail::MathFuncs::nll(mu, N, true, _doBinOffset);
         sumWeightKahanSum += N;
      }
   }

   finalizeResult(ctx, result, sumWeightKahanSum.Sum());
}

void RooNLLVarNew::doEvalChi2(RooFit::EvalContext &ctx, std::span<const double> preds, std::span<const double> weights,
                              std::span<const double> weightsSumW2) const
{
   // Error type None implies zero sigma for every bin: the chi2 is undefined
   // everywhere but empty bins. Match the legacy behaviour of returning zero.
   if (_chi2ErrorType == RooDataHist::None) {
      finalizeResult(ctx, ROOT::Math::KahanSum<double>{0.0}, 0.0);
      return;
   }

   auto config = ctx.config(this);
   std::span<const double> binVol = ctx.at(*_binVolumes);
   std::span<const double> errLo = _weightErrLo ? ctx.at(*_weightErrLo) : std::span<const double>{};
   std::span<const double> errHi = _weightErrHi ? ctx.at(*_weightErrHi) : std::span<const double>{};

   const double sumWeight = RooBatchCompute::reduceSum(config, weights.data(), weights.size());

   double normFactor = 1.0;
   switch (_funcMode) {
   case FuncMode::Pdf: normFactor = sumWeight; break;
   case FuncMode::ExtendedPdf: normFactor = ctx.at(*_expectedEvents)[0]; break;
   case FuncMode::Function: normFactor = 1.0; break;
   }

   ROOT::Math::KahanSum<double> result{0.0};
   ROOT::Math::KahanSum<double> sumWeightKahanSum{0.0};

   for (std::size_t i = 0; i < preds.size(); ++i) {
      const double N = weights[i];
      const double mu = preds[i] * normFactor * binVol[i];
      const double diff = mu - N;

      double sigma2;
      switch (_chi2ErrorType) {
      case RooDataHist::SumW2: sigma2 = weightsSumW2[i]; break;
      case RooDataHist::Poisson: {
         // Poisson errors are asymmetric: choose the side facing the prediction.
         const double err = diff > 0 ? errHi[i] : errLo[i];
         sigma2 = err * err;
         break;
      }
      default: sigma2 = mu; break; // Expected
      }

      // Skip bins where data, prediction and error are all zero (matches legacy RooChi2Var).
      if (sigma2 == 0.0 && N == 0.0 && mu == 0.0) {
         continue;
      }
      if (sigma2 <= 0.0) {
         logEvalError(Form("chi2 bin %lu has non-positive error; term replaced with NaN", (unsigned long)i));
         result += std::numeric_limits<double>::quiet_NaN();
         continue;
      }

      result += diff * diff / sigma2;
      sumWeightKahanSum += N;
   }

   finalizeResult(ctx, result, sumWeightKahanSum.Sum());
}

void RooNLLVarNew::doEval(RooFit::EvalContext &ctx) const
{
   std::span<const double> weights = ctx.at(_weightVar);
   std::span<const double> weightsSumW2 = ctx.at(_weightSquaredVar);

   if (_statistic == Statistic::Chi2) {
      return doEvalChi2(ctx, ctx.at(&*_func), weights, weightsSumW2);
   }

   if (_binnedL) {
      return doEvalBinnedL(ctx, ctx.at(&*_func), _weightSquared ? weightsSumW2 : weights);
   }

   auto config = ctx.config(this);

   auto probas = ctx.at(_func);

   double sumWeight = RooBatchCompute::reduceSum(config, weights.data(), weights.size());
   double sumWeight2 = 0.;
   if (_expectedEvents && _weightSquared) {
      sumWeight2 = RooBatchCompute::reduceSum(config, weightsSumW2.data(), weightsSumW2.size());
   }

   auto nllOut = RooBatchCompute::reduceNLL(config, probas, _weightSquared ? weightsSumW2 : weights,
                                            _doBinOffset ? ctx.at(*_offsetPdf) : std::span<const double>{});

   if (nllOut.nInfiniteValues > 0) {
      oocoutW(&*_func, Eval) << "RooAbsPdf::getLogVal(" << _func->GetName()
                             << ") WARNING: top-level pdf has some infinite values" << std::endl;
   }
   for (std::size_t i = 0; i < nllOut.nNonPositiveValues; ++i) {
      _func->logEvalError("getLogVal() top-level p.d.f not greater than zero");
   }
   for (std::size_t i = 0; i < nllOut.nNaNValues; ++i) {
      _func->logEvalError("getLogVal() top-level p.d.f evaluates to NaN");
   }

   if (_expectedEvents) {
      // The unbinned NLL path is only reached for pdf inputs, so the cast is safe.
      auto &pdf = static_cast<RooAbsPdf &>(const_cast<RooAbsReal &>(*_func));
      std::span<const double> expected = ctx.at(*_expectedEvents);
      nllOut.nllSum += pdf.extendedTerm(sumWeight, expected[0], _weightSquared ? sumWeight2 : 0.0, _doBinOffset);
   }

   finalizeResult(ctx, {nllOut.nllSum, nllOut.nllSumCarry}, sumWeight);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the prefix for the special variables of this NLL, like weights or bin
/// volumes.
/// \param[in] prefix The prefix to add to the observables and weight names.
void RooNLLVarNew::setPrefix(std::string const &prefix)
{
   _prefix = prefix;

   resetWeightVarNames();
}

void RooNLLVarNew::resetWeightVarNames()
{
   _weightVar->SetName((_prefix + weightVarName).c_str());
   _weightSquaredVar->SetName((_prefix + weightVarNameSumW2).c_str());
   if (_offsetPdf) {
      (*_offsetPdf)->SetName((_prefix + "_offset_func").c_str());
   }
   if (_binVolumes) {
      (*_binVolumes)->SetName((_prefix + binVolumeVarName).c_str());
   }
   if (_weightErrLo) {
      (*_weightErrLo)->SetName((_prefix + weightErrorLoVarName).c_str());
   }
   if (_weightErrHi) {
      (*_weightErrHi)->SetName((_prefix + weightErrorHiVarName).c_str());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Toggles the weight square correction.
void RooNLLVarNew::applyWeightSquared(bool flag)
{
   if (_statistic == Statistic::Chi2) {
      if (flag) {
         coutW(Fitting) << "RooNLLVarNew::applyWeightSquared(" << GetName()
                        << ") has no effect on a chi-squared evaluator; ignoring." << std::endl;
      }
      return;
   }
   _weightSquared = flag;
}

void RooNLLVarNew::enableOffsetting(bool flag)
{
   _doOffset = flag;
   _offset = ROOT::Math::KahanSum<double>{};
}

void RooNLLVarNew::finalizeResult(RooFit::EvalContext &ctx, ROOT::Math::KahanSum<double> result, double weightSum) const
{
   // If part of simultaneous PDF normalize probability over
   // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n)
   // If we do bin-by bin offsetting, we don't do this because it cancels out.
   // The correction is specific to NLL; it has no meaning for chi2.
   if (_statistic == Statistic::NLL && !_doBinOffset && _simCount > 1) {
      result += weightSum * std::log(static_cast<double>(_simCount));
   }

   // Check if value offset flag is set.
   if (_doOffset) {

      // If no offset is stored enable this feature now
      if (_offset.Sum() == 0 && _offset.Carry() == 0 && (result.Sum() != 0 || result.Carry() != 0)) {
         _offset = result;
      }
   }
   ctx.setOutputWithOffset(this, result, _offset);
}

} // namespace RooFit::Detail

/// \endcond
