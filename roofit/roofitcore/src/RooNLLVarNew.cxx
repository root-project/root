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

#include "ROOT/StringUtils.hxx"

#include "RooFit/Detail/Buffers.h"

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

} // namespace

/** Contstruct a RooNLLVarNew

\param pdf The pdf for which the nll is computed for
\param observables The observabes of the pdf
\param weight A pointer to the weight variable (if exists)
\param isExtended Set to true if this is an extended fit
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
      this->addOwnedComponents(*term.release());
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

\param dispatch A pointer to the RooBatchCompute library interface used for this computation
\param output An array of doubles where the computation results will be stored
\param nEvents The number of events to be processed
\param dataMap A map containing spans with the input data for the computation
**/
void RooNLLVarNew::computeBatch(cudaStream_t *stream, double *output, size_t /*nOut*/,
                                RooBatchCompute::DataMap &dataMap) const
{
   using namespace ROOT::Experimental::Detail;

   std::size_t nEvents = dataMap[&*_pdf].size();

   RooBatchCompute::VarVector vars = {&*_pdf};
   if (_weight)
      vars.push_back(&**_weight);
   RooBatchCompute::ArgVector args = {static_cast<double>(vars.size() - 1)};
   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;

   std::unique_ptr<AbsBuffer> logsBuffer = stream ? makeGpuBuffer(nEvents) : makeCpuBuffer(nEvents);
   double *logsBufferDataPtr = stream ? logsBuffer->gpuWritePtr() : logsBuffer->cpuWritePtr();
   dispatch->compute(stream, RooBatchCompute::NegativeLogarithms, logsBufferDataPtr, nEvents, dataMap, vars, args);

   if ((_isExtended || _rangeNormTerm) && _sumWeight == 0.0) {
      if (!_weight) {
         _sumWeight = nEvents;
      } else {
         auto weightSpan = dataMap[&**_weight];
         _sumWeight = weightSpan.size() == 1 ? weightSpan[0] * nEvents
                                             : dispatch->sumReduce(stream, dataMap[&**_weight].data(), nEvents);
      }
   }
   if (_rangeNormTerm) {
      auto rangeNormTermSpan = dataMap[&**_rangeNormTerm];
      if (rangeNormTermSpan.size() == 1) {
         _sumCorrectionTerm = _sumWeight * rangeNormTermSpan[0];
      } else {
         if (!_weight) {
            _sumCorrectionTerm = dispatch->sumReduce(stream, rangeNormTermSpan.data(), nEvents);
         } else {
            auto weightSpan = dataMap[&**_weight];
            if (weightSpan.size() == 1) {
               _sumCorrectionTerm = weightSpan[0] * dispatch->sumReduce(stream, rangeNormTermSpan.data(), nEvents);
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

   output[0] = reduce(stream, logsBufferDataPtr, nEvents);
   // std::cout << "RooNLLVar::computeBatch() " << output[0] << std::endl;
}

/** Reduce an array of nll values to the sum of them

\param dispatch A pointer to the RooBatchCompute library interface used for this computation
\param input The input array with the nlls to be reduced
\param nEvents the number of events to be processed
**/
double RooNLLVarNew::reduce(cudaStream_t *stream, const double *input, size_t nEvents) const
{
   auto dispatch = stream ? RooBatchCompute::dispatchCUDA : RooBatchCompute::dispatchCPU;
   double nll = dispatch->sumReduce(stream, input, nEvents);
   if (_isExtended) {
      assert(_sumWeight != 0.0);
      nll += _pdf->extendedTerm(_sumWeight, &_observables);
   }
   if (_rangeNormTerm) {
      nll += _sumCorrectionTerm;
   }
   return nll;
}

double RooNLLVarNew::getValV(const RooArgSet *) const
{
   // throw std::runtime_error("RooNLLVarNew::getValV was called directly which should not happen!");
   return 0.0;
}

double RooNLLVarNew::evaluate() const
{
   throw std::runtime_error("RooNLLVarNew::evaluate was called directly which should not happen!");
}

RooSpan<double> RooNLLVarNew::evaluateSpan(RooBatchCompute::RunContext &, const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::evaluatSpan was called directly which should not happen!");
}

RooSpan<const double> RooNLLVarNew::getValues(RooBatchCompute::RunContext &, const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::getValues was called directly which should not happen!");
}

void RooNLLVarNew::getParametersHook(const RooArgSet * /*nset*/, RooArgSet *params, Bool_t /*stripDisconnected*/) const
{
   // strip away the observables and weights
   params->remove(_observables, true, true);
   if (_weight)
      params->remove(**_weight, true, true);
}
