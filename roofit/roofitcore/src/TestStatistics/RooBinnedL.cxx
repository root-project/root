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
\file RooBinnedL.cxx
\class RooBinnedL
\ingroup Roofitcore

Class RooBinnedL implements a -log(likelihood) calculation from a dataset
(assumed to be binned) and a PDF. The NLL is calculated as
\f[
 \sum_\mathrm{data} -\log( \mathrm{pdf}(x_\mathrm{data}))
\f]
In extended mode, a
\f$ N_\mathrm{expect} - N_\mathrm{observed}*log(N_\mathrm{expect}) \f$ term is added.
**/

#include <RooFit/TestStatistics/RooBinnedL.h>

#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooAbsDataStore.h"
#include "RooRealSumPdf.h"
#include "RooRealVar.h"
#include "RooChangeTracker.h"

#include "TMath.h"

namespace RooFit {
namespace TestStatistics {

RooBinnedL::RooBinnedL(RooAbsPdf *pdf, RooAbsData *data)
   : RooAbsL(RooAbsL::ClonePdfData{pdf, data}, data->numEntries(), 1)
{
   // pdf must be a RooRealSumPdf representing a yield vector for a binned likelihood calculation
   if (!dynamic_cast<RooRealSumPdf *>(pdf)) {
      throw std::logic_error("RooBinnedL can only be created from pdf of type RooRealSumPdf!");
   }

   // Retrieve and cache bin widths needed to convert unnormalized binned pdf values back to yields

   // The Active label will disable pdf integral calculations
   pdf->setAttribute("BinnedLikelihoodActive");

   RooArgSet params;
   pdf->getParameters(data->get(), params) ;
   paramTracker_ = std::make_unique<RooChangeTracker>("chtracker","change tracker",params,true);

   std::unique_ptr<RooArgSet> obs(pdf->getObservables(data));
   if (obs->getSize() != 1) {
      throw std::logic_error(
         "RooBinnedL can only be created from combination of pdf and data which has exactly one observable!");
   } else {
      RooRealVar *var = (RooRealVar *)obs->first();
      std::list<double> *boundaries = pdf->binBoundaries(*var, var->getMin(), var->getMax());
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

RooBinnedL::~RooBinnedL() = default;

//////////////////////////////////////////////////////////////////////////////////
/// Calculate and return likelihood on subset of data from firstEvent to lastEvent
/// processed with a step size of 'stepSize'. If this an extended likelihood and
/// and the zero event is processed the extended term is added to the return
/// likelihood.
//
ROOT::Math::KahanSum<double>
RooBinnedL::evaluatePartition(Section bins, std::size_t /*components_begin*/, std::size_t /*components_end*/)
{
   // Throughout the calculation, we use Kahan's algorithm for summing to
   // prevent loss of precision - this is a factor four more expensive than
   // straight addition, but since evaluating the PDF is usually much more
   // expensive than that, we tolerate the additional cost...
   ROOT::Math::KahanSum<double> result;

   // Do not reevaluate likelihood if parameters have not changed
   if (!paramTracker_->hasChanged(true) & (cachedResult_ != 0)) return cachedResult_;

//   data->store()->recalculateCache(_projDeps, firstEvent, lastEvent, stepSize, (_binnedPdf?false:true));
   // TODO: check when we might need _projDeps (it seems to be mostly empty); ties in with TODO below
   data_->store()->recalculateCache(nullptr, bins.begin(N_events_), bins.end(N_events_), 1, false);

   ROOT::Math::KahanSum<double> sumWeight;

   for (std::size_t i = bins.begin(N_events_); i < bins.end(N_events_); ++i) {

      data_->get(i);

      double eventWeight = data_->weight();

      // Calculate log(Poisson(N|mu) for this bin
      double N = eventWeight;
      double mu = pdf_->getVal() * _binw[i];

      if (mu <= 0 && N > 0) {

         // Catch error condition: data present where zero events are predicted
//         logEvalError(Form("Observed %f events in bin %d with zero event yield", N, i));
         // TODO: check if using regular stream vs logEvalError error gathering is ok
         oocoutI(nullptr, Minimization)
            << "Observed " << N << " events in bin " << i << " with zero event yield" << std::endl;

      } else if (fabs(mu) < 1e-10 && fabs(N) < 1e-10) {

         // Special handling of this case since log(Poisson(0,0)=0 but can't be calculated with usual log-formula
         // since log(mu)=0. No update of result is required since term=0.

      } else {

         double term = -1 * (-mu + N * log(mu) - TMath::LnGamma(N + 1));

         sumWeight += eventWeight;
         result += term;
      }
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
   return result;
}

} // namespace TestStatistics
} // namespace RooFit
