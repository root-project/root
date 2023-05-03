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
In contrast to the `RooNLLVar` class, any logic except the bare minimum has been
transfered away to other classes, like the `RooFitDriver`. This class also calls
functions from `RooBatchCompute` library to provide faster computation times.
**/

#include "RooNLLVarNew.h"

#include <RooBatchCompute.h>
#include <RooNaNPacker.h>
#include <RooRealVar.h>
#include "RooFit/Detail/Buffers.h"

#include <ROOT/StringUtils.hxx>

#include <TClass.h>
#include <TMath.h>
#include <Math/Util.h>

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ROOT::Experimental;

// Declare constexpr static members to make them available if odr-used in C++14.
constexpr const char *RooNLLVarNew::weightVarName;
constexpr const char *RooNLLVarNew::weightVarNameSumW2;

namespace {

RooArgSet getObs(RooAbsArg const &arg, RooArgSet const &observables)
{
   RooArgSet out;
   arg.getObservables(&observables, out);
   return out;
}

RooRealVar *dummyVar(const char *name)
{
   return new RooRealVar(name, name, 1.0);
}

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
     _observables{getObs(pdf, observables)},
     _isExtended{isExtended},
     _binnedL{pdf.getAttribute("BinnedLikelihoodActive")},
     _weightVar{"weightVar", "weightVar", this, *dummyVar(weightVarName), true, false, true},
     _weightSquaredVar{weightVarNameSumW2, weightVarNameSumW2, this, *dummyVar("weightSquardVar"), true, false, true},
     _binVolumeVar{"binVolumeVar", "binVolumeVar", this, *dummyVar("_bin_volume"), true, false, true}
{
   if (_binnedL) {
      fillBinWidthsFromPdfBoundaries(pdf);
   }

   resetWeightVarNames();
   enableOffsetting(offsetMode == RooFit::OffsetMode::Initial);
   enableBinOffsetting(offsetMode == RooFit::OffsetMode::Bin);
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name),
     _pdf{"pdf", this, other._pdf},
     _observables{other._observables},
     _isExtended{other._isExtended},
     _weightSquared{other._weightSquared},
     _binnedL{other._binnedL},
     _doOffset{other._doOffset},
     _simCount{other._simCount},
     _prefix{other._prefix},
     _weightVar{"weightVar", this, other._weightVar},
     _weightSquaredVar{"weightSquaredVar", this, other._weightSquaredVar}
{
}

void RooNLLVarNew::fillBinWidthsFromPdfBoundaries(RooAbsReal const &pdf)
{
   // Check if the bin widths were already filled
   if (!_binw.empty()) {
      return;
   }

   if (_observables.size() != 1) {
      throw std::runtime_error("BinnedPdf optimization only works with a 1D pdf.");
   } else {
      auto *var = static_cast<RooRealVar *>(_observables.first());
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

double RooNLLVarNew::computeBatchBinnedL(RooSpan<const double> preds, RooSpan<const double> weights) const
{
   ROOT::Math::KahanSum<double> result{0.0};
   ROOT::Math::KahanSum<double> sumWeightKahanSum{0.0};

   for (std::size_t i = 0; i < preds.size(); ++i) {

      double eventWeight = weights[i];

      // Calculate log(Poisson(N|mu) for this bin
      double N = eventWeight;
      double mu = preds[i] * _binw[i];

      if (mu <= 0 && N > 0) {

         // Catch error condition: data present where zero events are predicted
         logEvalError(Form("Observed %f events in bin %lu with zero event yield", N, (unsigned long)i));

      } else if (std::abs(mu) < 1e-10 && std::abs(N) < 1e-10) {

         // Special handling of this case since log(Poisson(0,0)=0 but can't be calculated with usual log-formula
         // since log(mu)=0. No update of result is required since term=0.

      } else {

         result += -1 * (-mu + N * log(mu) - TMath::LnGamma(N + 1));
         sumWeightKahanSum += eventWeight;
      }
   }

   return finalizeResult(result, sumWeightKahanSum.Sum());
}

/** Compute multiple negative logs of propabilities

\param output An array of doubles where the computation results will be stored
\param nOut not used
\note nEvents is the number of events to be processed (the dataMap size)
\param dataMap A map containing spans with the input data for the computation
**/
void RooNLLVarNew::computeBatch(cudaStream_t *stream, double *output, size_t /*nOut*/,
                                RooFit::Detail::DataMap const &dataMap) const
{
   RooSpan<const double> weights = dataMap.at(_weightVar);
   RooSpan<const double> weightsSumW2 = dataMap.at(_weightSquaredVar);

   if (_binnedL) {
      output[0] = computeBatchBinnedL(dataMap.at(&*_pdf), _weightSquared ? weightsSumW2 : weights);
      return;
   }

   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;

   auto probas = dataMap.at(_pdf);

   _sumWeight =
      weights.size() == 1 ? weights[0] * probas.size() : dispatch->reduceSum(stream, weights.data(), weights.size());
   if (_isExtended && _weightSquared && _sumWeight2 == 0.0) {
      _sumWeight2 = weights.size() == 1 ? weightsSumW2[0] * probas.size()
                                        : dispatch->reduceSum(stream, weightsSumW2.data(), weightsSumW2.size());
   }

   auto nllOut = dispatch->reduceNLL(stream, probas, _weightSquared ? weightsSumW2 : weights, weights, _sumWeight,
                                     _doBinOffset ? dataMap.at(_binVolumeVar) : RooSpan<const double>{});

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

   if (_isExtended) {
      double expected = _pdf->expectedEvents(&_observables);
      nllOut.nllSum += _pdf->extendedTerm(_sumWeight, expected, _weightSquared ? _sumWeight2 : 0.0, _doBinOffset);
   }

   output[0] = finalizeResult(nllOut.nllSum, _sumWeight);
}

void RooNLLVarNew::getParametersHook(const RooArgSet * /*nset*/, RooArgSet *params, bool /*stripDisconnected*/) const
{
   // strip away the special variables
   params->remove(RooArgList{*_weightVar, *_weightSquaredVar, *_binVolumeVar}, true, true);
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
   _binVolumeVar->SetName((_prefix + "_bin_volume").c_str());
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

double RooNLLVarNew::finalizeResult(ROOT::Math::KahanSum<double> result, double weightSum) const
{
   // If part of simultaneous PDF normalize probability over
   // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n)
   if (_simCount > 1) {
      result += weightSum * std::log(static_cast<double>(_simCount));
   }

   // Check if value offset flag is set.
   if (_doOffset) {

      // If no offset is stored enable this feature now
      if (_offset.Sum() == 0 && _offset.Carry() == 0 && (result.Sum() != 0 || result.Carry() != 0)) {
         _offset = result;
      }

      // Subtract offset
      if (!RooAbsReal::hideOffset()) {
         result -= _offset;
      }
   }
   return result.Sum();
}

void RooNLLVarNew::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   std::string resName = ctx.makeValidVarName(GetName()) + "Result";
   ctx.addResult(this, resName);
   ctx.addToGlobalScope("double " + resName + " = 0;\n");

   if (_simCount > 1) {

      {
         auto scope = ctx.beginLoop(this);
         ctx.addToCodeBody(resName + " += " + ctx.getResult(_weightVar.arg()) + ";\n");
      }
      ctx.addToCodeBody(resName + " *= std::log(" + std::to_string(static_cast<double>(_simCount)) + ");\n");
   }

   // Begin loop scope for the observables and weight variable. If the weight
   // is a scalar, the context will ignore it for the loop scope. The closing
   // brackets of the loop is written at the end of the scopes lifetime.
   {
      auto scope = ctx.beginLoop(this);
      ctx.addToCodeBody(this, resName + " -= " + ctx.getResult(_weightVar.arg()) + " * std::log(" +
                                 ctx.getResult(_pdf.arg()) + ");\n");
   }
}
