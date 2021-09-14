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

#include <TestStatistics/RooUnbinnedL.h>

#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooAbsDataStore.h"

namespace RooFit {
namespace TestStatistics {

RooUnbinnedL::RooUnbinnedL(RooAbsPdf *pdf, RooAbsData *data,
                           RooAbsL::Extended extended)
   : RooAbsL(RooAbsL::ClonePdfData{pdf, data}, data->numEntries(), 1, extended)
{}

RooUnbinnedL::RooUnbinnedL(const RooUnbinnedL &other)
   : RooAbsL(other), apply_weight_squared(other.apply_weight_squared), _first(other._first)
{}

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

//////////////////////////////////////////////////////////////////////////////////
/// Calculate and return likelihood on subset of data from firstEvent to lastEvent
/// processed with a step size of 'stepSize'. If this an extended likelihood and
/// and the zero event is processed the extended term is added to the return
/// likelihood.
///
double RooUnbinnedL::evaluatePartition(Section events,
                                        std::size_t /*components_begin*/, std::size_t /*components_end*/)
{
   // Throughout the calculation, we use Kahan's algorithm for summing to
   // prevent loss of precision - this is a factor four more expensive than
   // straight addition, but since evaluating the PDF is usually much more
   // expensive than that, we tolerate the additional cost...
   Double_t result(0), carry(0);

   data_->store()->recalculateCache(nullptr, events.begin(N_events_), events.end(N_events_), 1, kTRUE);

   Double_t sumWeight(0), sumWeightCarry(0);

   for (std::size_t i = events.begin(N_events_); i < events.end(N_events_); ++i) {
      data_->get(i);
      if (!data_->valid()) {
         continue;
      }

      Double_t eventWeight = data_->weight();
      if (0. == eventWeight * eventWeight) {
         continue;
      }
      if (apply_weight_squared) {
         eventWeight = data_->weightSquared();
      }

      Double_t term = -eventWeight * pdf_->getLogVal(normSet_.get());

      Double_t y = eventWeight - sumWeightCarry;
      Double_t t = sumWeight + y;
      sumWeightCarry = (t - sumWeight) - y;
      sumWeight = t;

      y = term - carry;
      t = result + y;
      carry = (t - result) - y;
      result = t;
   }

   // include the extended maximum likelihood term, if requested
   if (extended_) {
      if (apply_weight_squared) {

         // Calculate sum of weights-squared here for extended term
         Double_t sumW2(0), sumW2carry(0);
         for (Int_t i = 0; i < data_->numEntries(); i++) {
            data_->get(i);
            Double_t y = data_->weightSquared() - sumW2carry;
            Double_t t = sumW2 + y;
            sumW2carry = (t - sumW2) - y;
            sumW2 = t;
         }

         Double_t expected = pdf_->expectedEvents(data_->get());

         // Adjust calculation of extended term with W^2 weighting: adjust poisson such that
         // estimate of Nexpected stays at the same value, but has a different variance, rescale
         // both the observed and expected count of the Poisson with a factor sum[w] / sum[w^2] which is
         // the effective weight of the Poisson term.
         // i.e. change Poisson(Nobs = sum[w]| Nexp ) --> Poisson( sum[w] * sum[w] / sum[w^2] | Nexp * sum[w] /
         // sum[w^2] ) weighted by the effective weight  sum[w^2]/ sum[w] in the likelihood. Since here we compute
         // the likelihood with the weight square we need to multiply by the square of the effective weight expectedW
         // = expected * sum[w] / sum[w^2]   : effective expected entries observedW =  sum[w]  * sum[w] / sum[w^2] :
         // effective observed entries The extended term for the likelihood weighted by the square of the weight will
         // be then:
         //  (sum[w^2]/ sum[w] )^2 * expectedW -  (sum[w^2]/ sum[w] )^2 * observedW * log (expectedW)  and this is
         //  using the previous expressions for expectedW and observedW
         //  sum[w^2] / sum[w] * expected - sum[w^2] * log (expectedW)
         //  and since the weights are constants in the likelihood we can use log(expected) instead of log(expectedW)

         Double_t expectedW2 = expected * sumW2 / data_->sumEntries();
         Double_t extra = expectedW2 - sumW2 * log(expected);

         // Double_t y = pdf->extendedTerm(sumW2, data->get()) - carry;

         Double_t y = extra - carry;

         Double_t t = result + y;
         carry = (t - result) - y;
         result = t;
      } else {
         Double_t y = pdf_->extendedTerm(data_->sumEntries(), data_->get()) - carry;
         Double_t t = result + y;
         carry = (t - result) - y;
         result = t;
      }
   }

   // If part of simultaneous PDF normalize probability over
   // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n)
   if (sim_count_ > 1) {
      Double_t y = sumWeight * log(1.0 * sim_count_) - carry;
      Double_t t = result + y;
      carry = (t - result) - y;
      result = t;
   }

   // At the end of the first full calculation, wire the caches
   if (_first) {
      _first = false;
      pdf_->wireAllCaches();
   }

   eval_carry_ = carry;
   return result;
}

} // namespace TestStatistics
} // namespace RooFit
