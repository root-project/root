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

#include "RooAddition.h"
#include "RooFormulaVar.h"
#include "RooNaNPacker.h"

#include "RooFit/Detail/Buffers.h"

#include "ROOT/StringUtils.hxx"

#include "Math/Util.h"

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ROOT::Experimental;

namespace {

std::unique_ptr<RooAbsReal> createRangeNormTerm(RooAbsPdf const &pdf, RooArgSet const &observables,
                                                std::string const &baseName, std::string const &rangeNames)
{

   RooArgSet observablesInPdf;
   pdf.getObservables(&observables, observablesInPdf);

   RooArgList termList;

   auto pdfIntegralCurrent = pdf.createIntegral(observablesInPdf, &observablesInPdf, nullptr, rangeNames.c_str());
   auto term =
      new RooFormulaVar((baseName + "_correctionTerm").c_str(), "(log(x[0]))", RooArgList(*pdfIntegralCurrent));
   termList.add(*term);

   auto integralFull = pdf.createIntegral(observablesInPdf, &observablesInPdf, nullptr);
   auto fullRangeTerm = new RooFormulaVar((baseName + "_foobar").c_str(), "-(log(x[0]))", RooArgList(*integralFull));
   termList.add(*fullRangeTerm);

   auto out =
      std::unique_ptr<RooAbsReal>{new RooAddition((baseName + "_correction").c_str(), "correction", termList, true)};
   return out;
}

template <class Input>
double kahanSum(Input const &input)
{
   return ROOT::Math::KahanSum<double, 4u>::Accumulate(input.begin(), input.end()).Sum();
}

} // namespace

/** Construct a RooNLLVarNew
\param name the name
\param title the title
\param pdf The pdf for which the nll is computed for
\param observables The observabes of the pdf
\param weight A pointer to the weight variable (if exists)
\param isExtended Set to true if this is an extended fit
\param rangeName the range name
**/
RooNLLVarNew::RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables,
                           RooAbsReal *weight, bool isExtended, std::string const &rangeName)
   : RooAbsReal(name, title), _pdf{"pdf", "pdf", this, pdf}, _observables{observables}, _isExtended{isExtended}
//_rangeNormTerm{rangeName.empty() ? nullptr : createRangeNormTerm(pdf, observables, pdf.GetName(), rangeName)}
{
   if (weight)
      _weight = std::make_unique<RooTemplateProxy<RooAbsReal>>("_weight", "_weight", this, *weight);
   if (!rangeName.empty()) {
      auto term = createRangeNormTerm(pdf, observables, pdf.GetName(), rangeName);
      _rangeNormTerm = std::make_unique<RooTemplateProxy<RooAbsReal>>("_rangeNormTerm", "_rangeNormTerm", this, *term);
      this->addOwnedComponents(std::move(term));
   }
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name), _pdf{"pdf", this, other._pdf}, _observables{other._observables}
{
   if (other._weight)
      _weight = std::make_unique<RooTemplateProxy<RooAbsReal>>("_weight", this, *other._weight);
   if (other._rangeNormTerm)
      _rangeNormTerm = std::make_unique<RooTemplateProxy<RooAbsReal>>("_rangeNormTerm", this, *other._rangeNormTerm);
}

/** Compute multiple negative logs of propabilities

\param output An array of doubles where the computation results will be stored
\param nOut not used
\note nEvents is the number of events to be processed (the dataMap size)
\param dataMap A map containing spans with the input data for the computation
**/
void RooNLLVarNew::computeBatch(cudaStream_t * /*stream*/, double *output, size_t /*nOut*/,
                                RooBatchCompute::DataMap &dataMap) const
{
   std::size_t nEvents = dataMap[&*_pdf].size();
   auto probas = dataMap[&*_pdf];

   auto logProbasBuffer = ROOT::Experimental::Detail::makeCpuBuffer(nEvents);
   RooSpan<double> logProbas{logProbasBuffer->cpuWritePtr(), nEvents};
   (*_pdf).getLogProbabilities(probas, logProbas.data());

   std::vector<double> nlls(nEvents);
   nlls.reserve(nEvents);

   if (_weight) {
      double const *weights = dataMap[&**_weight].data();
      for (std::size_t i = 0; i < nEvents; ++i) {
         // Explicitely add zero if zero weight to get rid of eventual NaNs in
         // logProbas that have no weight anyway.
         nlls.push_back(weights[i] == 0.0 ? 0.0 : -logProbas[i] * weights[i]);
      }
   } else {
      for (auto const &p : logProbas)
         nlls.push_back(-p);
   }

   if ((_isExtended || _rangeNormTerm) && _sumWeight == 0.0) {
      if (!_weight) {
         _sumWeight = nEvents;
      } else {
         auto weightSpan = dataMap[&**_weight];
         _sumWeight = weightSpan.size() == 1 ? weightSpan[0] * nEvents : kahanSum(dataMap[&**_weight]);
      }
   }
   if (_rangeNormTerm) {
      auto rangeNormTermSpan = dataMap[&**_rangeNormTerm];
      if (rangeNormTermSpan.size() == 1) {
         _sumCorrectionTerm = _sumWeight * rangeNormTermSpan[0];
      } else {
         if (!_weight) {
            _sumCorrectionTerm = kahanSum(rangeNormTermSpan);
         } else {
            auto weightSpan = dataMap[&**_weight];
            if (weightSpan.size() == 1) {
               _sumCorrectionTerm = weightSpan[0] * kahanSum(rangeNormTermSpan);
            } else {
               // We don't need to use the library for now because the weights and
               // correction term integrals are always in the CPU map.
               _sumCorrectionTerm = 0.0;
               for (std::size_t i = 0; i < nEvents; ++i) {
                  _sumCorrectionTerm += weightSpan[i] * rangeNormTermSpan[i];
               }
            }
         }
      }
   }

   double nll = kahanSum(nlls);

   if (std::isnan(nll)) {
      // Special handling of evaluation errors.
      // We can recover if the bin/event that results in NaN has a weight of zero:
      RooNaNPacker nanPacker;
      for (std::size_t i = 0; i < probas.size(); ++i) {
         if (_weight) {
            double const *weights = dataMap[&**_weight].data();
            if (std::isnan(logProbas[i]) && weights[i] != 0.0) {
               nanPacker.accumulate(logProbas[i]);
            }
         }
         if (std::isnan(logProbas[i])) {
            nanPacker.accumulate(logProbas[i]);
         }
      }

      // Some events with evaluation errors. Return "badness" of errors.
      if (nanPacker.getPayload() > 0.) {
         nll = nanPacker.getNaNWithPayload();
      }
   }

   if (_isExtended) {
      assert(_sumWeight != 0.0);
      nll += _pdf->extendedTerm(_sumWeight, &_observables);
   }
   if (_rangeNormTerm) {
      nll += _sumCorrectionTerm;
   }
   output[0] = nll;

   // Since the output of this node is always of size one, it is possible that it is
   // evaluated in scalar mode. We need to set the cached value and clear
   // the dirty flag.
   const_cast<RooNLLVarNew *>(this)->setCachedValue(nll);
   const_cast<RooNLLVarNew *>(this)->clearValueDirty();
}

double RooNLLVarNew::evaluate() const
{
   throw std::runtime_error("RooNLLVarNew::evaluate was called directly which should not happen!");
   return _value;
}

void RooNLLVarNew::getParametersHook(const RooArgSet * /*nset*/, RooArgSet *params, Bool_t /*stripDisconnected*/) const
{
   // strip away the observables and weights
   params->remove(_observables, true, true);
   if (_weight)
      params->remove(**_weight, true, true);
}
