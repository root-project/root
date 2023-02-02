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

Class RooUnbinnedL implements a -log(likelihood) calculation from a dataset
(assumed to be unbinned) and a PDF. The NLL is calculated as
\f[
 \sum_\mathrm{data} -\log( \mathrm{pdf}(x_\mathrm{data}))
\f]
In extended mode, a
\f$ N_\mathrm{expect} - N_\mathrm{observed}*log(N_\mathrm{expect}) \f$ term is added.
**/

#include <RooFit/TestStatistics/RooUnbinnedL.h>

#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooAbsDataStore.h"
#include "RooNLLVar.h"  // RooNLLVar::ComputeScalar
#include "RunContext.h"
#include "RooChangeTracker.h"

namespace RooFit {
namespace TestStatistics {

RooUnbinnedL::RooUnbinnedL(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended,
                           bool useBatchedEvaluations)
   : RooAbsL(RooAbsL::ClonePdfData{pdf, data}, data->numEntries(), 1, extended),
     useBatchedEvaluations_(useBatchedEvaluations)
{
   std::unique_ptr<RooArgSet> params(pdf->getParameters(data));
   paramTracker_ = std::make_unique<RooChangeTracker>("chtracker","change tracker",*params,true);
}

RooUnbinnedL::RooUnbinnedL(const RooUnbinnedL &other)
   : RooAbsL(other), apply_weight_squared(other.apply_weight_squared), _first(other._first),
     useBatchedEvaluations_(other.useBatchedEvaluations_), lastSection_(other.lastSection_),
     cachedResult_(other.cachedResult_)
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

void RooUnbinnedL::setUseBatchedEvaluations(bool flag) {
   useBatchedEvaluations_ = flag;
}

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

   // Do not reevaluate likelihood if parameters nor event range have changed
   if (!paramTracker_->hasChanged(true) && events == lastSection_ && (cachedResult_.Sum() != 0 || cachedResult_.Carry() != 0)) return cachedResult_;

   data_->store()->recalculateCache(nullptr, events.begin(N_events_), events.end(N_events_), 1, true);

   if (useBatchedEvaluations_) {
      std::unique_ptr<RooBatchCompute::RunContext> evalData;
      std::tie(result, sumWeight) = RooNLLVar::computeBatchedFunc(pdf_.get(), data_.get(), evalData, normSet_.get(), apply_weight_squared,
                                                                  1, events.begin(N_events_), events.end(N_events_));
   } else {
      std::tie(result, sumWeight) = RooNLLVar::computeScalarFunc(pdf_.get(), data_.get(), normSet_.get(), apply_weight_squared,
                                                                 1, events.begin(N_events_), events.end(N_events_));
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

   // At the end of the first full calculation, wire the caches
   if (_first) {
      _first = false;
      pdf_->wireAllCaches();
   }

   cachedResult_ = result;
   lastSection_ = events;
   return result;
}

} // namespace TestStatistics
} // namespace RooFit
