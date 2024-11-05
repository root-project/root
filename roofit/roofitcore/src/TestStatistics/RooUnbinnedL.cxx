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

/**
\file RooUnbinnedL.cxx
\class RooUnbinnedL
\ingroup Roofitcore

A -log(likelihood) calculation from a dataset
(assumed to be unbinned) and a PDF. The NLL is calculated as
\f[
 \sum_\mathrm{data} -\log( \mathrm{pdf}(x_\mathrm{data}))
\f]
In extended mode, a
\f$ N_\mathrm{expect} - N_\mathrm{observed}*log(N_\mathrm{expect}) \f$ term is added.
**/

#include <RooFit/TestStatistics/RooUnbinnedL.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAbsDataStore.h>
#include <RooChangeTracker.h>
#include <RooNaNPacker.h>
#include <RooFit/Evaluator.h>

#include "../RooFit/BatchModeDataHelpers.h"

namespace RooFit {
namespace TestStatistics {

namespace {

RooAbsL::ClonePdfData clonePdfData(RooAbsPdf &pdf, RooAbsData &data, RooFit::EvalBackend evalBackend)
{
   if (evalBackend.value() == RooFit::EvalBackend::Value::Legacy) {
      return {&pdf, &data};
   }
   // For the evaluation with the BatchMode, the pdf needs to be "compiled" for
   // a given normalization set.
   return {RooFit::Detail::compileForNormSet(pdf, *data.get()), &data};
}

} // namespace

RooUnbinnedL::RooUnbinnedL(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended,
                           RooFit::EvalBackend evalBackend)
   : RooAbsL(clonePdfData(*pdf, *data, evalBackend), data->numEntries(), 1, extended)
{
   std::unique_ptr<RooArgSet> params(pdf->getParameters(data));
   paramTracker_ = std::make_unique<RooChangeTracker>("chtracker", "change tracker", *params, true);

   if (evalBackend.value() != RooFit::EvalBackend::Value::Legacy) {
      evaluator_ = std::make_unique<RooFit::Evaluator>(*pdf_, evalBackend.value() == RooFit::EvalBackend::Value::Cuda);
      std::stack<std::vector<double>>{}.swap(_vectorBuffers);
      auto dataSpans =
         RooFit::BatchModeDataHelpers::getDataSpans(*data, "", nullptr, /*skipZeroWeights=*/true,
                                                    /*takeGlobalObservablesFromData=*/false, _vectorBuffers);
      for (auto const &item : dataSpans) {
         evaluator_->setInput(item.first->GetName(), item.second, false);
      }
   }
}

RooUnbinnedL::RooUnbinnedL(const RooUnbinnedL &other)
   : RooAbsL(other),
     apply_weight_squared(other.apply_weight_squared),
     _first(other._first),
     lastSection_(other.lastSection_),
     cachedResult_(other.cachedResult_),
     evaluator_(other.evaluator_)
{
   paramTracker_ = std::make_unique<RooChangeTracker>(*other.paramTracker_);
}

RooUnbinnedL::~RooUnbinnedL() = default;

//////////////////////////////////////////////////////////////////////////////////

/// Returns true if value was changed, false otherwise.
bool RooUnbinnedL::setApplyWeightSquared(bool flag)
{
   if (apply_weight_squared != flag) {
      apply_weight_squared = flag;
      return true;
   }
   //   setValueDirty();
   return false;
}

namespace {

using ComputeResult = std::pair<ROOT::Math::KahanSum<double>, double>;

// Copy of RooNLLVar::computeScalarFunc.
ComputeResult computeScalarFunc(const RooAbsPdf *pdfClone, RooAbsData *dataClone, RooArgSet *normSet, bool weightSq,
                                std::size_t stepSize, std::size_t firstEvent, std::size_t lastEvent,
                                RooAbsPdf const *offsetPdf = nullptr)
{
   ROOT::Math::KahanSum<double> kahanWeight;
   ROOT::Math::KahanSum<double> kahanProb;
   RooNaNPacker packedNaN(0.f);

   for (auto i = firstEvent; i < lastEvent; i += stepSize) {
      dataClone->get(i);

      double weight = dataClone->weight(); // FIXME

      if (0. == weight * weight)
         continue;
      if (weightSq)
         weight = dataClone->weightSquared();

      double logProba = pdfClone->getLogVal(normSet);

      if (offsetPdf) {
         logProba -= offsetPdf->getLogVal(normSet);
      }

      const double term = -weight * logProba;

      kahanWeight.Add(weight);
      kahanProb.Add(term);
      packedNaN.accumulate(term);
   }

   if (packedNaN.getPayload() != 0.) {
      // Some events with evaluation errors. Return "badness" of errors.
      return {ROOT::Math::KahanSum<double>{packedNaN.getNaNWithPayload()}, kahanWeight.Sum()};
   }

   return {kahanProb, kahanWeight.Sum()};
}

// For now, almost exact copy of computeScalarFunc.
ComputeResult computeBatchFunc(std::span<const double> probas, RooAbsData *dataClone, bool weightSq,
                               std::size_t stepSize, std::size_t firstEvent, std::size_t lastEvent)
{
   ROOT::Math::KahanSum<double> kahanWeight;
   ROOT::Math::KahanSum<double> kahanProb;
   RooNaNPacker packedNaN(0.f);

   for (auto i = firstEvent; i < lastEvent; i += stepSize) {
      dataClone->get(i);

      double weight = dataClone->weight();

      if (0. == weight * weight)
         continue;
      if (weightSq)
         weight = dataClone->weightSquared();

      double logProba = std::log(probas[i]);
      const double term = -weight * logProba;

      kahanWeight.Add(weight);
      kahanProb.Add(term);
      packedNaN.accumulate(term);
   }

   if (packedNaN.getPayload() != 0.) {
      // Some events with evaluation errors. Return "badness" of errors.
      return {ROOT::Math::KahanSum<double>{packedNaN.getNaNWithPayload()}, kahanWeight.Sum()};
   }

   return {kahanProb, kahanWeight.Sum()};
}

} // namespace

//////////////////////////////////////////////////////////////////////////////////
/// Calculate and return likelihood on subset of data from firstEvent to lastEvent
/// processed with a step size of 'stepSize'. If this an extended likelihood and
/// and the zero event is processed the extended term is added to the return
/// likelihood.
///
ROOT::Math::KahanSum<double>
RooUnbinnedL::evaluatePartition(Section events, std::size_t /*components_begin*/, std::size_t /*components_end*/)
{
   // Throughout the calculation, we use Kahan's algorithm for summing to
   // prevent loss of precision - this is a factor four more expensive than
   // straight addition, but since evaluating the PDF is usually much more
   // expensive than that, we tolerate the additional cost...
   ROOT::Math::KahanSum<double> result;
   double sumWeight;
   auto numEvalErrorsBefore = RooAbsReal::numEvalErrors();

   // Do not reevaluate likelihood if parameters nor event range have changed
   if (!paramTracker_->hasChanged(true) && events == lastSection_ &&
       (cachedResult_.Sum() != 0 || cachedResult_.Carry() != 0))
      return cachedResult_;

   if (evaluator_) {
      // Here, we have a memory allocation that should be avoided when this
      // code needs to be optimized.
      std::span<const double> probas = evaluator_->run();
      std::tie(result, sumWeight) =
         computeBatchFunc(probas, data_.get(), apply_weight_squared, 1, events.begin(N_events_), events.end(N_events_));
   } else {
      data_->store()->recalculateCache(nullptr, events.begin(N_events_), events.end(N_events_), 1, true);
      std::tie(result, sumWeight) = computeScalarFunc(pdf_.get(), data_.get(), normSet_.get(), apply_weight_squared, 1,
                                                      events.begin(N_events_), events.end(N_events_));
   }

   // include the extended maximum likelihood term, if requested
   if (extended_ && events.begin_fraction == 0) {
      result += pdf_->extendedTerm(*data_, apply_weight_squared);
   }

   // If part of simultaneous PDF normalize probability over
   // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n)
   if (sim_count_ > 1) {
      result += sumWeight * log(1.0 * sim_count_);
   }

   // At the end of the first full calculation, wire the caches. This doesn't
   // need to be done in BatchMode with the RooFit driver.
   if (_first && !evaluator_) {
      _first = false;
      pdf_->wireAllCaches();
   }

   if ((RooAbsReal::evalErrorLoggingMode() == RooAbsReal::CollectErrors ||
        RooAbsReal::evalErrorLoggingMode() == RooAbsReal::CountErrors) &&
       numEvalErrorsBefore == RooAbsReal::numEvalErrors()) {
      cachedResult_ = result;
      lastSection_ = events;
   }
   return result;
}

} // namespace TestStatistics
} // namespace RooFit
