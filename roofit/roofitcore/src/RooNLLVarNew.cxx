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

#include <RooAddition.h>
#include <RooFormulaVar.h>
#include <RooNaNPacker.h>
#include <RooRealVar.h>
#include "RooFit/Detail/Buffers.h"

#include <ROOT/StringUtils.hxx>

#include <TClass.h>
#include <TMath.h>
#include <Math/Util.h>
#include <TMath.h>

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ROOT::Experimental;

// Declare constexpr static members to make them available if odr-used in C++14.
constexpr const char *RooNLLVarNew::weightVarName;
constexpr const char *RooNLLVarNew::weightVarNameSumW2;

namespace {

template <class Input>
double kahanSum(Input const &input)
{
   return ROOT::Math::KahanSum<double, 4u>::Accumulate(input.begin(), input.end()).Sum();
}

RooArgSet getObs(RooAbsArg const &arg, RooArgSet const &observables)
{
   RooArgSet out;
   arg.getObservables(&observables, out);
   return out;
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
                           bool isExtended, RooFit::OffsetMode offsetMode, bool binnedL)
   : RooAbsReal(name, title), _pdf{"pdf", "pdf", this, pdf}, _observables{getObs(pdf, observables)},
     _isExtended{isExtended}, _binnedL{binnedL},
     _weightVar{"weightVar", "weightVar", this, *new RooRealVar(weightVarName, weightVarName, 1.0), true, false, true},
     _weightSquaredVar{weightVarNameSumW2,
                       weightVarNameSumW2,
                       this,
                       *new RooRealVar("weightSquardVar", "weightSquaredVar", 1.0),
                       true,
                       false,
                       true}
{
   if (_binnedL) {
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

   resetWeightVarNames();
   enableOffsetting(offsetMode == RooFit::OffsetMode::Initial);
   // TODO: implement template offsetting mode as well
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name), _pdf{"pdf", this, other._pdf}, _observables{other._observables},
     _isExtended{other._isExtended}, _weightSquared{other._weightSquared}, _binnedL{other._binnedL},
     _doOffset{other._doOffset}, _simCount{other._simCount}, _prefix{other._prefix},
     _weightVar{"weightVar", this, other._weightVar}, _weightSquaredVar{"weightSquaredVar", this,
                                                                        other._weightSquaredVar}
{
}

/** Compute multiple negative logs of propabilities

\param output An array of doubles where the computation results will be stored
\param nOut not used
\note nEvents is the number of events to be processed (the dataMap size)
\param dataMap A map containing spans with the input data for the computation
**/
void RooNLLVarNew::computeBatch(cudaStream_t * /*stream*/, double *output, size_t /*nOut*/,
                                RooFit::Detail::DataMap const &dataMap) const
{
   std::size_t nEvents = dataMap.at(_pdf).size();

   auto weights = dataMap.at(_weightVar);
   auto weightsSumW2 = dataMap.at(_weightSquaredVar);
   auto weightSpan = _weightSquared ? weightsSumW2 : weights;

   if (_binnedL) {
      ROOT::Math::KahanSum<double> result{0.0};
      ROOT::Math::KahanSum<double> sumWeightKahanSum{0.0};
      auto preds = dataMap.at(&*_pdf);

      for (std::size_t i = 0; i < nEvents; ++i) {

         double eventWeight = weightSpan[i];

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

      output[0] = finalizeResult(std::move(result), sumWeightKahanSum.Sum());

      return;
   }

   auto probas = dataMap.at(_pdf);

   _logProbasBuffer.resize(nEvents);
   (*_pdf).getLogProbabilities(probas, _logProbasBuffer.data());

   _sumWeight = weights.size() == 1 ? weights[0] * nEvents : kahanSum(weights);

   if (_isExtended && _weightSquared && _sumWeight2 == 0.0) {
      _sumWeight2 = weights.size() == 1 ? weightsSumW2[0] * nEvents : kahanSum(weightsSumW2);
   }

   ROOT::Math::KahanSum<double> kahanProb;
   RooNaNPacker packedNaN(0.f);

   for (std::size_t i = 0; i < nEvents; ++i) {

      double eventWeight = weightSpan.size() > 1 ? weightSpan[i] : weightSpan[0];
      if (0. == eventWeight * eventWeight)
         continue;

      const double term = -eventWeight * _logProbasBuffer[i];

      kahanProb.Add(term);
      packedNaN.accumulate(term);
   }

   if (packedNaN.getPayload() != 0.) {
      // Some events with evaluation errors. Return "badness" of errors.
      kahanProb = Math::KahanSum<double>(packedNaN.getNaNWithPayload());
   }

   if (_isExtended) {
      double expected = _pdf->expectedEvents(&_observables);
      kahanProb += _pdf->extendedTerm(_sumWeight, expected, _weightSquared ? _sumWeight2 : 0.0);
   }

   output[0] = finalizeResult(std::move(kahanProb), _sumWeight);
}

void RooNLLVarNew::getParametersHook(const RooArgSet * /*nset*/, RooArgSet *params, bool /*stripDisconnected*/) const
{
   // strip away the observables and weights
   params->remove(_observables, true, true);
   params->remove(RooArgList{*_weightVar, *_weightSquaredVar}, true, true);
}

////////////////////////////////////////////////////////////////////////////////
/// Clones the PDF recursively and prefixes the names of all nodes, except for
/// parameter nodes. Used for simultaneous fits.
/// \return A RooArgSet with the new observable args.
/// \param[in] prefix The prefix to add to the observables and weight names.
RooArgSet RooNLLVarNew::prefixArgNames(std::string const &prefix)
{
   _prefix = prefix;

   std::unique_ptr<RooAbsReal> pdfClone = RooHelpers::cloneTreeWithSameParameters(*_pdf, &_observables);

   redirectServers(RooArgList{*pdfClone});

   RooArgSet parameters;
   pdfClone->getParameters(&_observables, parameters);

   _observables.clear();

   RooArgSet nodes;
   pdfClone->treeNodeServerList(&nodes);
   for (RooAbsArg *arg : nodes) {
      if (!parameters.find(*arg)) {
         arg->SetName((prefix + arg->GetName()).c_str());
         if (dynamic_cast<RooRealVar *>(arg)) {
            // It's an observable
            static_cast<RooRealVar *>(arg)->setConstant();
            _observables.add(*arg);
            arg->setAttribute("__obs__");
         }
      }
   }

   addOwnedComponents(std::move(pdfClone));

   resetWeightVarNames();

   return _observables;
}

void RooNLLVarNew::resetWeightVarNames()
{
   _weightVar->SetName((_prefix + weightVarName).c_str());
   _weightSquaredVar->SetName((_prefix + weightVarNameSumW2).c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// Toggles the weight square correction.
void RooNLLVarNew::applyWeightSquared(bool flag)
{
   _weightSquared = flag;
}

std::unique_ptr<RooArgSet>
RooNLLVarNew::fillNormSetForServer(RooArgSet const & /*normSet*/, RooAbsArg const & /*server*/) const
{
   if (_binnedL) {
      return std::make_unique<RooArgSet>();
   }
   return nullptr;
}

void RooNLLVarNew::enableOffsetting(bool flag)
{
   _doOffset = flag;
   _offset = ROOT::Math::KahanSum<double>{};
}

double RooNLLVarNew::finalizeResult(ROOT::Math::KahanSum<double> &&result, double weightSum) const
{
   // If part of simultaneous PDF normalize probability over
   // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n)
   if (_simCount > 1) {
      result += weightSum * std::log(static_cast<double>(_simCount));
   }

   // Check if value offset flag is set.
   if (_offset) {

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
