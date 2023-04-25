/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/BatchModeHelpers.h"
#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooBatchCompute.h>
#include <RooBinSamplingPdf.h>
#include <RooCategory.h>
#include <RooConstraintSum.h>
#include <RooDataSet.h>
#include "RooNLLVarNew.h"
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include <string>

using ROOT::Experimental::RooNLLVarNew;

namespace {

std::unique_ptr<RooAbsArg> createSimultaneousNLL(RooSimultaneous const &simPdf, bool isExtended,
                                                 std::string const &rangeName, RooFit::OffsetMode offset)
{
   RooAbsCategoryLValue const &simCat = simPdf.indexCat();

   // Prepare the NLL terms for each component
   RooArgList nllTerms;
   for (auto const &catState : simCat) {
      std::string const &catName = catState.first;
      RooAbsCategory::value_type catIndex = catState.second;

      // If the channel is not in the selected range of the category variable, we
      // won't create an NLL this channel.
      if (!rangeName.empty()) {
         // Only the RooCategory supports ranges, not the other
         // RooAbsCategoryLValue-derived classes.
         auto simCatAsRooCategory = dynamic_cast<RooCategory const *>(&simCat);
         if (simCatAsRooCategory && !simCatAsRooCategory->isStateInRange(rangeName.c_str(), catIndex)) {
            continue;
         }
      }

      if (RooAbsPdf *pdf = simPdf.getPdf(catName.c_str())) {
         auto name = std::string("nll_") + pdf->GetName();
         std::unique_ptr<RooArgSet> observables(
            static_cast<RooArgSet *>(std::unique_ptr<RooArgSet>(pdf->getVariables())->selectByAttrib("__obs__", true)));
         auto nll = std::make_unique<RooNLLVarNew>(name.c_str(), name.c_str(), *pdf, *observables, isExtended, offset);
         // Rename the special variables
         nll->setPrefix(std::string("_") + catName + "_");
         nllTerms.addOwned(std::move(nll));
      }
   }

   for (auto *nll : static_range_cast<RooNLLVarNew *>(nllTerms)) {
      nll->setSimCount(nllTerms.size());
   }

   // Time to sum the NLLs
   auto nll = std::make_unique<RooAddition>("mynll", "mynll", nllTerms);
   nll->addOwnedComponents(std::move(nllTerms));
   return nll;
}

} // namespace

std::unique_ptr<RooAbsReal>
RooFit::BatchModeHelpers::createNLL(RooAbsPdf &pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
                                    std::string const &rangeName, RooArgSet const &projDeps, bool isExtended,
                                    double integrateOverBinsPrecision, RooFit::OffsetMode offset)
{
   if (constraints) {
      // Redirect the global observables to the ones from the dataset if applicable.
      constraints->setData(data, false);

      // The computation graph for the constraints is very small, no need to do
      // the tracking of clean and dirty nodes here.
      constraints->setOperMode(RooAbsArg::ADirty);
   }

   RooArgSet observables;
   pdf.getObservables(data.get(), observables);
   observables.remove(projDeps, true, true);

   oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                            << ") fixing normalization set for coefficient determination to observables in data"
                            << "\n";
   pdf.fixAddCoefNormalization(observables, false);

   // Deal with the IntegrateBins argument
   RooArgList binSamplingPdfs;
   std::unique_ptr<RooAbsPdf> wrappedPdf = RooBinSamplingPdf::create(pdf, data, integrateOverBinsPrecision);
   RooAbsPdf &finalPdf = wrappedPdf ? *wrappedPdf : pdf;
   if (wrappedPdf) {
      binSamplingPdfs.addOwned(std::move(wrappedPdf));
   }
   // Done dealing with the IntegrateBins option

   RooArgList nllTerms;

   auto simPdf = dynamic_cast<RooSimultaneous *>(&finalPdf);
   if (simPdf) {
      simPdf->wrapPdfsInBinSamplingPdfs(data, integrateOverBinsPrecision);
      nllTerms.addOwned(createSimultaneousNLL(*simPdf, isExtended, rangeName, offset));
   } else {
      nllTerms.addOwned(
         std::make_unique<RooNLLVarNew>("RooNLLVarNew", "RooNLLVarNew", finalPdf, observables, isExtended, offset));
   }
   if (constraints) {
      nllTerms.addOwned(std::move(constraints));
   }

   std::string nllName = std::string("nll_") + pdf.GetName() + "_" + data.GetName();
   auto nll = std::make_unique<RooAddition>(nllName.c_str(), nllName.c_str(), nllTerms);
   nll->addOwnedComponents(std::move(binSamplingPdfs));
   nll->addOwnedComponents(std::move(nllTerms));

   return nll;
}
