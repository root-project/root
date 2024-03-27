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

#include "RooNLLVarNew.h"

#include <RooHistPdf.h>
#include <RooBatchCompute.h>
#include <RooDataHist.h>
#include <RooNaNPacker.h>
#include <RooConstVar.h>
#include <RooRealVar.h>
#include <RooSetProxy.h>
#include "RooFit/Detail/Buffers.h"
#include <RooFit/Detail/EvaluateFuncs.h>

#include "RooFitImplHelpers.h"

#include <ROOT/StringUtils.hxx>

#include <TClass.h>
#include <TMath.h>
#include <Math/Util.h>

#include <numeric>
#include <stdexcept>
#include <vector>

// Declare constexpr static members to make them available if odr-used in C++14.
constexpr const char *RooNLLVarNew::weightVarName;
constexpr const char *RooNLLVarNew::weightVarNameSumW2;

namespace {

// Use RooConstVar for dummies such that they don't get included in getParameters().
RooConstVar *dummyVar(const char *name)
{
   return new RooConstVar(name, name, 1.0);
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

/** Construct a RooNLLVarNew
\param name the name
\param title the title
\param pdf The pdf for which the nll is computed for
\param observables The observabes of the pdf
\param isExtended Set to true if this is an extended fit
**/
RooNLLVarNew::RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables,
                           bool isExtended, RooFit::OffsetMode offsetMode)
   : RooAbsReal(name, title),
     _pdf{"pdf", "pdf", this, pdf},
     _weightVar{"weightVar", "weightVar", this, *dummyVar(weightVarName), true, false, true},
     _weightSquaredVar{weightVarNameSumW2, weightVarNameSumW2, this, *dummyVar("weightSquardVar"), true, false, true},
     _binnedL{pdf.getAttribute("BinnedLikelihoodActive")}
{
   RooArgSet obs;
   pdf.getObservables(&observables, obs);

   // In the "BinnedLikelihoodActiveYields" mode, the pdf values can directly
   // be interpreted as yields and don't need to be multiplied by the bin
   // widths. That's why we don't need to even fill them in this case.
   if (_binnedL && !pdf.getAttribute("BinnedLikelihoodActiveYields")) {
      fillBinWidthsFromPdfBoundaries(pdf, obs);
   }

   if (isExtended && !_binnedL) {
      std::unique_ptr<RooAbsReal> expectedEvents = pdf.createExpectedEventsFunc(&obs);
      if (expectedEvents) {
         _expectedEvents =
            std::make_unique<RooTemplateProxy<RooAbsReal>>("expectedEvents", "expectedEvents", this, *expectedEvents);
         addOwnedComponents(std::move(expectedEvents));
      }
   }

   resetWeightVarNames();
   enableOffsetting(offsetMode == RooFit::OffsetMode::Initial);
   enableBinOffsetting(offsetMode == RooFit::OffsetMode::Bin);

   // In the binned likelihood code path, we directly use that data weights for
   // the offsetting.
   if (!_binnedL && _doBinOffset) {
      auto offsetPdf = std::make_unique<RooOffsetPdf>("_offset_pdf", "_offset_pdf", obs, *_weightVar);
      _offsetPdf = std::make_unique<RooTemplateProxy<RooAbsPdf>>("offsetPdf", "offsetPdf", this, *offsetPdf);
      addOwnedComponents(std::move(offsetPdf));
   }
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name),
     _pdf{"pdf", this, other._pdf},
     _weightVar{"weightVar", this, other._weightVar},
     _weightSquaredVar{"weightSquaredVar", this, other._weightSquaredVar},
     _weightSquared{other._weightSquared},
     _binnedL{other._binnedL},
     _doOffset{other._doOffset},
     _simCount{other._simCount},
     _prefix{other._prefix},
     _binw{other._binw}
{
   if (other._expectedEvents) {
      _expectedEvents = std::make_unique<RooTemplateProxy<RooAbsReal>>("expectedEvents", this, *other._expectedEvents);
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
         result += RooFit::Detail::EvaluateFuncs::nllEvaluate(mu, N, true, _doBinOffset);
         sumWeightKahanSum += N;
      }
   }

   finalizeResult(ctx, result, sumWeightKahanSum.Sum());
}

void RooNLLVarNew::doEval(RooFit::EvalContext &ctx) const
{
   std::span<const double> weights = ctx.at(_weightVar);
   std::span<const double> weightsSumW2 = ctx.at(_weightSquaredVar);

   if (_binnedL) {
      return doEvalBinnedL(ctx, ctx.at(&*_pdf), _weightSquared ? weightsSumW2 : weights);
   }

   auto config = ctx.config(this);

   auto probas = ctx.at(_pdf);

   _sumWeight = weights.size() == 1 ? weights[0] * probas.size()
                                    : RooBatchCompute::reduceSum(config, weights.data(), weights.size());
   if (_expectedEvents && _weightSquared && _sumWeight2 == 0.0) {
      _sumWeight2 = weights.size() == 1 ? weightsSumW2[0] * probas.size()
                                        : RooBatchCompute::reduceSum(config, weightsSumW2.data(), weightsSumW2.size());
   }

   auto nllOut = RooBatchCompute::reduceNLL(config, probas, _weightSquared ? weightsSumW2 : weights,
                                            _doBinOffset ? ctx.at(*_offsetPdf) : std::span<const double>{});

   if (nllOut.nLargeValues > 0) {
      oocoutW(&*_pdf, Eval) << "RooAbsPdf::getLogVal(" << _pdf->GetName()
                            << ") WARNING: top-level pdf has unexpectedly large values" << std::endl;
   }
   for (std::size_t i = 0; i < nllOut.nNonPositiveValues; ++i) {
      _pdf->logEvalError("getLogVal() top-level p.d.f not greater than zero");
   }
   for (std::size_t i = 0; i < nllOut.nNaNValues; ++i) {
      _pdf->logEvalError("getLogVal() top-level p.d.f evaluates to NaN");
   }

   if (_expectedEvents) {
      std::span<const double> expected = ctx.at(*_expectedEvents);
      nllOut.nllSum += _pdf->extendedTerm(_sumWeight, expected[0], _weightSquared ? _sumWeight2 : 0.0, _doBinOffset);
   }

   finalizeResult(ctx, {nllOut.nllSum, nllOut.nllSumCarry}, _sumWeight);
}

void RooNLLVarNew::getParametersHook(const RooArgSet * /*nset*/, RooArgSet *params, bool /*stripDisconnected*/) const
{
   // strip away the special variables
   params->remove(RooArgList{*_weightVar, *_weightSquaredVar}, true, true);
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
      (*_offsetPdf)->SetName((_prefix + "_offset_pdf").c_str());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Toggles the weight square correction.
void RooNLLVarNew::applyWeightSquared(bool flag)
{
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
   // If we do bin-by bin offsetting, we don't do this because it cancels out
   if (!_doBinOffset && _simCount > 1) {
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

void RooNLLVarNew::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   if (_binnedL && !_pdf->getAttribute("BinnedLikelihoodActiveYields")) {
      std::stringstream errorMsg;
      errorMsg << "RooNLLVarNew::translate(): binned likelihood optimization is only supported when raw pdf "
                  "values can be interpreted as yields."
               << " This is not the case for HistFactory models written with ROOT versions before 6.26.00";
      coutE(InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }

   std::string weightSumName = RooFit::Detail::makeValidVarName(GetName()) + "WeightSum";
   std::string resName = RooFit::Detail::makeValidVarName(GetName()) + "Result";
   ctx.addResult(this, resName);
   ctx.addToGlobalScope("double " + weightSumName + " = 0.0;\n");
   ctx.addToGlobalScope("double " + resName + " = 0.0;\n");

   const bool needWeightSum = _expectedEvents || _simCount > 1;

   if (needWeightSum) {
      auto scope = ctx.beginLoop(this);
      ctx.addToCodeBody(weightSumName + " += " + ctx.getResult(*_weightVar) + ";\n");
   }
   if (_simCount > 1) {
      std::string simCountStr = std::to_string(static_cast<double>(_simCount));
      ctx.addToCodeBody(resName + " += " + weightSumName + " * std::log(" + simCountStr + ");\n");
   }

   // Begin loop scope for the observables and weight variable. If the weight
   // is a scalar, the context will ignore it for the loop scope. The closing
   // brackets of the loop is written at the end of the scopes lifetime.
   {
      auto scope = ctx.beginLoop(this);
      std::string term = ctx.buildCall("RooFit::Detail::EvaluateFuncs::nllEvaluate", _pdf, _weightVar, _binnedL, 0);
      ctx.addToCodeBody(this, resName + " += " + term + ";");
   }
   if (_expectedEvents) {
      std::string expected = ctx.getResult(**_expectedEvents);
      ctx.addToCodeBody(resName + " += " + expected + " - " + weightSumName + " * std::log(" + expected + ");\n");
   }
}

/// \endcond
