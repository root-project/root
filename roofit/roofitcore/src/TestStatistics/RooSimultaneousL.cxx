/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include <TestStatistics/RooSimultaneousL.h>
#include <RooSimultaneous.h>
#include <RooAbsData.h>
#include <RooRealSumPdf.h>
#include <RooProdPdf.h>
#include <TestStatistics/RooBinnedL.h>
#include <TestStatistics/RooUnbinnedL.h>
#include <ROOT/RMakeUnique.hxx>

namespace RooFit {
namespace TestStatistics {

RooSimultaneousL::RooSimultaneousL(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended)
   : RooAbsL(pdf, data,
             0,  // will be set to true value in ctor with N_components
             data->numEntries(), extended)
{
   auto sim_pdf = dynamic_cast<RooSimultaneous *>(pdf);
   if (sim_pdf == nullptr) {
      throw std::logic_error("Can only build RooSimultaneousL from RooSimultaneous pdf!");
   }

   // the rest of this constructor is an adaptation of RooAbsTestStatistic::initSimMode:

   RooAbsCategoryLValue &simCat = (RooAbsCategoryLValue &)sim_pdf->indexCat();

   TString simCatName(simCat.GetName());
   std::unique_ptr<TList> dsetList{data_->split(simCat, processEmptyDataSets())};
   if (!dsetList) {
      oocoutE((TObject *)nullptr, Fitting)
         << "RooSimultaneousL::RooSimultaneousL(" << GetName()
         << ") ERROR: index category of simultaneous pdf is missing in dataset, aborting" << std::endl;
      throw std::logic_error("RooAbsTestStatistic::initSimMode() ERROR, index category of simultaneous pdf is missing "
                             "in dataset, aborting");
   }

   // Count number of used states
   RooCatType *type;
   std::unique_ptr<TIterator> catIter{simCat.typeIterator()};
   while ((type = (RooCatType *)catIter->Next())) {
      // Retrieve the PDF for this simCat state
      RooAbsPdf *component_pdf = sim_pdf->getPdf(type->GetName());
      auto dset = (RooAbsData *)dsetList->FindObject(type->GetName());

      if (component_pdf && dset && (0. != dset->sumEntries() || processEmptyDataSets())) {
         ++N_components;
      }
   }

   // Allocate arrays
   components_.reserve(N_components);
   //   _gofSplitMode.resize(N_components);  // not used, Hybrid mode only, see below

   // Create array of regular fit contexts, containing subset of data and single fitCat PDF
   catIter->Reset();
   std::size_t n = 0;
   while ((type = (RooCatType *)catIter->Next())) {
      // Retrieve the PDF for this simCat state
      RooAbsPdf *component_pdf = sim_pdf->getPdf(type->GetName());
      auto dset = (RooAbsData *)dsetList->FindObject(type->GetName());

      if (component_pdf && dset && (0. != dset->sumEntries() || processEmptyDataSets())) {
         ooccoutI((TObject *)nullptr, Fitting)
            << "RooSimultaneousL: creating slave calculator #" << n << " for state " << type->GetName()
            << " (" << dset->numEntries() << " dataset entries)" << std::endl;

         // *** START HERE
         // WVE HACK determine if we have a RooRealSumPdf and then treat it like a binned likelihood
         RooAbsPdf *binnedPdf = 0;
         Bool_t binnedL = kFALSE;
         if (component_pdf->getAttribute("BinnedLikelihood") && component_pdf->IsA()->InheritsFrom(RooRealSumPdf::Class())) {
            // Simplest case: top-level of component is a RRSP
            binnedPdf = component_pdf;
            binnedL = kTRUE;
         } else if (component_pdf->IsA()->InheritsFrom(RooProdPdf::Class())) {
            // Default case: top-level pdf is a product of RRSP and other pdfs
            RooFIter iter = ((RooProdPdf *)component_pdf)->pdfList().fwdIterator();
            RooAbsArg *component;
            while ((component = iter.next())) {
               if (component->getAttribute("BinnedLikelihood") &&
                   component->IsA()->InheritsFrom(RooRealSumPdf::Class())) {
                  binnedPdf = (RooAbsPdf *)component;
                  binnedL = kTRUE;
               }
               if (component->getAttribute("MAIN_MEASUREMENT")) {
                  // not really a binned pdf, but this prevents a (potentially) long list of subsidiary measurements to
                  // be passed to the slave calculator
                  binnedPdf = (RooAbsPdf *)component;
               }
            }
         }
         // WVE END HACK
         // Below here directly pass binnedPdf instead of PROD(binnedPdf,constraints) as constraints are evaluated
         // elsewhere anyway and omitting them reduces model complexity and associated handling/cloning times
//         if (_splitRange && rangeName) {
//            _gofArray[n] =
//               create(type->GetName(), type->GetName(), (binnedPdf ? *binnedPdf : *component_pdf), *dset, *projDeps,
//                      Form("%s_%s", rangeName, type->GetName()), addCoefRangeName, _nCPU * (_mpinterl ? -1 : 1),
//                      _mpinterl, _CPUAffinity, _verbose, _splitRange, binnedL);
//         } else {
//            _gofArray[n] =
//               create(type->GetName(), type->GetName(), (binnedPdf ? *binnedPdf : *component_pdf), *dset, *projDeps, rangeName,
//                      addCoefRangeName, _nCPU, _mpinterl, _CPUAffinity, _verbose, _splitRange, binnedL);
         if (binnedL) {
            components_.push_back(std::make_unique<RooBinnedL>((binnedPdf ? binnedPdf : component_pdf), dset));
         } else {
            components_.push_back(std::make_unique<RooUnbinnedL>((binnedPdf ? binnedPdf : component_pdf), dset));
         }
//         }
         components_.back()->set_sim_count(N_components);
         // *** END HERE

         // TODO: left out Hybrid mode for now, evaluate later whether to reinclude (also then change
         // evaluate_partition)
         //         // Fill per-component split mode with Bulk Partition for now so that Auto will map to bulk-splitting
         //         of all components if (_mpinterl==RooFit::Hybrid) {
         //            if (dset->numEntries()<10) {
         //               //cout << "RAT::initSim("<< GetName() << ") MP mode is auto, setting split mode for component
         //               "<< n << " to SimComponents"<< endl ; _gofSplitMode[n] = RooFit::SimComponents;
         //               _gofArray[n]->_mpinterl = RooFit::SimComponents;
         //            } else {
         //               //cout << "RAT::initSim("<< GetName() << ") MP mode is auto, setting split mode for component
         //               "<< n << " to BulkPartition"<< endl ; _gofSplitMode[n] = RooFit::BulkPartition;
         //               _gofArray[n]->_mpinterl = RooFit::BulkPartition;
         //            }
         //         }
         //
         // Servers may have been redirected between instantiation and (deferred) initialization

         std::unique_ptr<RooArgSet> actualParams {binnedPdf ? binnedPdf->getParameters(dset) : component_pdf->getParameters(dset)};
         std::unique_ptr<RooArgSet> selTargetParams {(RooArgSet *)getParameters()->selectCommon(*actualParams)};

         // TODO: I don't think we have to redirect servers, because our classes make no use of those, but we should make sure. Do we need to reset the parameter set instead?
//         components_.back()->recursiveRedirectServers(*selTargetParams);

         ++n;
      } else {
         if ((!dset || (0. != dset->sumEntries() && !processEmptyDataSets())) && component_pdf) {
            ooccoutD((TObject *)nullptr, Fitting) << "RooSimultaneousL: state " << type->GetName()
                                                      << " has no data entries, no slave calculator created" << std::endl;
         }
      }
   }
   oocoutI((TObject *)nullptr, Fitting) << "RooSimultaneousL: created " << n << " slave calculators."
                                            << std::endl;

   // Delete datasets by hand as TList::Delete() doesn't see our datasets as 'on the heap'...
   std::unique_ptr<TIterator> iter {dsetList->MakeIterator()};
   TObject *ds;
   while ((ds = iter->Next())) {
      delete ds;
   }
}

bool RooSimultaneousL::processEmptyDataSets() const
{
   // TODO: check whether this is correct! This is copied the implementation of the RooNLLVar override; the implementation in RooAbsTestStatistic always returns true
   return extended_;
}

double RooSimultaneousL::evaluate_partition(std::size_t events_begin, std::size_t events_end,
                                            std::size_t components_begin, std::size_t components_end)
{
   // Evaluate specified range of owned GOF objects
   double ret = 0;

   // from RooAbsOptTestStatistic::combinedValue (which is virtual, so could be different for non-RooNLLVar!):
   eval_carry_ = 0;
   for (std::size_t ix = components_begin; ix < components_end; ++ix) {
      double y = components_[ix]->evaluate_partition(events_begin, events_end, 0, 0);
      eval_carry_ += components_[ix]->get_carry();
      y -= eval_carry_;
      double t = ret + y;
      eval_carry_ = (t - ret) - y;
      ret = t;
   }

   // Note: compared to the RooAbsTestStatistic implementation that this was taken from, we leave out Hybrid and
   // SimComponents interleaving support here, this should be implemented by calculator, if desired.

   return ret;
}

} // namespace TestStatistics
} // namespace RooFit
