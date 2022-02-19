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

#include <RooFit/BatchModeHelpers.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooBinSamplingPdf.h>
#include <RooConstraintSum.h>
#include <RooDataSet.h>
#include <RooFitDriver.h>
#include <RooNLLVarNew.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include <ROOT/StringUtils.hxx>

#include <string>

namespace {

std::unique_ptr<RooAbsArg> prepareSimultaneousModelForBatchMode(RooSimultaneous &simPdf, RooArgSet &observables,
                                                                RooAbsReal *weight, bool isExtended,
                                                                std::string const &rangeName)
{

   // Prepare the NLLTerms for each component
   RooArgList nllTerms;
   for (auto const &catItem : simPdf.indexCat()) {
      auto const &catName = catItem.first;
      auto *pdf = simPdf.getPdf(catName.c_str());
      auto nllName = std::string("nll_") + pdf->GetName();
      nllTerms.add(*new ROOT::Experimental::RooNLLVarNew(nllName.c_str(), nllName.c_str(), *pdf, observables, weight,
                                                         isExtended, rangeName));
   }

   RooArgSet newObservables;

   // Rename the observables and weights in each component
   std::size_t iNLL = 0;
   for (auto const &catItem : simPdf.indexCat()) {
      auto const &catName = catItem.first;
      auto &nll = nllTerms[iNLL];
      RooArgSet pdfObs;
      nll.getObservables(&observables, pdfObs);
      if (weight)
         pdfObs.add(*weight);
      RooArgSet obsClones;
      pdfObs.snapshot(obsClones);
      for (RooAbsArg *arg : obsClones) {
         auto newName = std::string("_") + catName + "_" + arg->GetName();
         arg->setAttribute((std::string("ORIGNAME:") + arg->GetName()).c_str());
         arg->SetName(newName.c_str());
      }
      nll.recursiveRedirectServers(obsClones, false, true);
      newObservables.add(obsClones);
      static_cast<ROOT::Experimental::RooNLLVarNew &>(nll).setObservables(obsClones);
      nll.addOwnedComponents(std::move(obsClones));
      ++iNLL;
   }

   observables.clear();
   observables.add(newObservables);

   // Time to sum the NLLs
   return std::make_unique<RooAddition>("mynll", "mynll", nllTerms, true);
}

} // namespace

std::unique_ptr<RooAbsReal>
RooFit::BatchModeHelpers::createNLL(RooAbsPdf &pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
                                    std::string const &rangeName, std::string const &addCoefRangeName,
                                    RooArgSet const &projDeps, bool isExtended, double integrateOverBinsPrecision,
                                    RooFit::BatchModeOption batchMode)
{
   std::unique_ptr<RooRealVar> weightVar;

   std::unique_ptr<ROOT::Experimental::RooFitDriver> driver;

   RooArgSet observables;
   pdf.getObservables(data.get(), observables);
   observables.remove(projDeps, true, true);

   oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                            << ") fixing normalization set for coefficient determination to observables in data"
                            << "\n";
   pdf.fixAddCoefNormalization(observables, false);
   if (!addCoefRangeName.empty()) {
      oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                               << ") fixing interpretation of coefficients of any component to range "
                               << addCoefRangeName << "\n";
      pdf.fixAddCoefRange(addCoefRangeName.c_str(), false);
   }

   if (data.isWeighted()) {
      std::string weightVarName = "_weight";
      if (auto *dataSet = dynamic_cast<RooDataSet const *>(&data)) {
         if (dataSet->weightVar())
            weightVarName = dataSet->weightVar()->GetName();
      }

      // make a clone of the weight variable (or an initial instance, if it doesn't exist)
      // the clone will hold the weight value (or values as a batch) and will participate
      // in the computation graph of the RooFit driver.
      weightVar = std::make_unique<RooRealVar>(weightVarName.c_str(), "Weight(s) of events", data.weight());
   }

   // Deal with the IntegrateBins argument
   RooArgList binSamplingPdfs;
   std::unique_ptr<RooAbsPdf> wrappedPdf;
   wrappedPdf = RooBinSamplingPdf::create(pdf, data, integrateOverBinsPrecision);
   RooAbsPdf &finalPdf = wrappedPdf ? *wrappedPdf : pdf;
   if (wrappedPdf) {
      binSamplingPdfs.addOwned(std::move(wrappedPdf));
   }
   // Done dealing with the IntegrateBins option

   RooArgList nllTerms;

   if (auto simPdf = dynamic_cast<RooSimultaneous *>(&finalPdf)) {
      auto *simPdfClone = static_cast<RooSimultaneous *>(simPdf->cloneTree());
      simPdfClone->wrapPdfsInBinSamplingPdfs(data, integrateOverBinsPrecision);
      // Warning! This mutates "observables"
      nllTerms.addOwned(
         prepareSimultaneousModelForBatchMode(*simPdfClone, observables, weightVar.get(), isExtended, rangeName));
   } else {
      nllTerms.addOwned(std::make_unique<ROOT::Experimental::RooNLLVarNew>(
         "RooNLLVarNew", "RooNLLVarNew", finalPdf, observables, weightVar.get(), isExtended, rangeName));
   }
   if (constraints) {
      nllTerms.addOwned(std::move(constraints));
   }

   std::string nllName = std::string("nll_") + pdf.GetName() + "_" + data.GetName();
   auto nll = std::make_unique<RooAddition>(nllName.c_str(), nllName.c_str(), nllTerms);
   nll->addOwnedComponents(std::move(binSamplingPdfs));
   nll->addOwnedComponents(std::move(nllTerms));

   if (auto simPdf = dynamic_cast<RooSimultaneous *>(&finalPdf)) {
      RooArgSet parameters;
      pdf.getParameters(data.get(), parameters);
      nll->recursiveRedirectServers(parameters);
      driver = std::make_unique<ROOT::Experimental::RooFitDriver>(data, *nll, observables, batchMode, rangeName,
                                                                  &simPdf->indexCat());
   } else {
      driver = std::make_unique<ROOT::Experimental::RooFitDriver>(data, *nll, observables, batchMode, rangeName);
   }

   // Set the fitrange attribute so that RooPlot can automatically plot the fitting range by default
   if (!rangeName.empty()) {

      std::string fitrangeValue;
      auto subranges = ROOT::Split(rangeName, ",");
      for (auto const &subrange : subranges) {
         if (subrange.empty())
            continue;
         std::string fitrangeValueSubrange = std::string("fit_") + nll->GetName();
         if (subranges.size() > 1) {
            fitrangeValueSubrange += "_" + subrange;
         }
         fitrangeValue += fitrangeValueSubrange + ",";
         for (auto *observable : static_range_cast<RooRealVar *>(observables)) {
            observable->setRange(fitrangeValueSubrange.c_str(), observable->getMin(subrange.c_str()),
                                 observable->getMax(subrange.c_str()));
         }
      }
      fitrangeValue = fitrangeValue.substr(0, fitrangeValue.size() - 1);
      pdf.setStringAttribute("fitrange", fitrangeValue.c_str());
   }

   auto driverWrapper = ROOT::Experimental::RooFitDriver::makeAbsRealWrapper(std::move(driver));
   driverWrapper->addOwnedComponents(std::move(nll));
   if (weightVar)
      driverWrapper->addOwnedComponents(std::move(weightVar));

   return driverWrapper;
}
