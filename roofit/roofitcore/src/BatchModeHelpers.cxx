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

using ROOT::Experimental::RooFitDriver;
using ROOT::Experimental::RooNLLVarNew;

namespace {

std::unique_ptr<RooAbsArg> prepareSimultaneousModelForBatchMode(RooSimultaneous &simPdf, RooArgSet &observables,
                                                                bool isExtended, std::string const &rangeName)
{

   // Prepare the NLLTerms for each component
   RooArgList nllTerms;
   for (auto const &catItem : simPdf.indexCat()) {
      auto const &catName = catItem.first;
      auto *pdf = simPdf.getPdf(catName.c_str());
      auto nllName = std::string("nll_") + pdf->GetName();
      nllTerms.add(*new RooNLLVarNew(nllName.c_str(), nllName.c_str(), *pdf, observables, isExtended, rangeName));
   }

   RooArgSet newObservables;

   // Rename the observables and weights in each component
   std::size_t iNLL = 0;
   for (auto const &catItem : simPdf.indexCat()) {
      auto const &catName = catItem.first;
      auto &nll = static_cast<RooNLLVarNew &>(nllTerms[iNLL]);
      newObservables.add(nll.prefixObservableAndWeightNames(std::string("_") + catName + "_"));
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
   std::unique_ptr<RooFitDriver> driver;

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
      nllTerms.addOwned(prepareSimultaneousModelForBatchMode(*simPdfClone, observables, isExtended, rangeName));
   } else {
      nllTerms.addOwned(
         std::make_unique<RooNLLVarNew>("RooNLLVarNew", "RooNLLVarNew", finalPdf, observables, isExtended, rangeName));
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
      driver = std::make_unique<RooFitDriver>(data, *nll, observables, batchMode, rangeName, &simPdf->indexCat());
   } else {
      driver = std::make_unique<RooFitDriver>(data, *nll, observables, batchMode, rangeName);
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

   auto driverWrapper = RooFitDriver::makeAbsRealWrapper(std::move(driver));
   driverWrapper->addOwnedComponents(std::move(nll));

   return driverWrapper;
}

void RooFit::BatchModeHelpers::logArchitectureInfo(RooFit::BatchModeOption batchMode)
{
   // We have to exit early if the message stream is not active. Otherwise it's
   // possible that this funciton skips logging because it thinks it has
   // already logged, but actually it didn't.
   if (!RooMsgService::instance().isActive(static_cast<RooAbsArg *>(nullptr), RooFit::Fitting, RooFit::INFO))
      return;

   // Don't repeat logging architecture info if the batchMode option didn't change
   {
      // Second element of pair tracks whether this function has already been called
      static std::pair<RooFit::BatchModeOption, bool> lastBatchMode;
      if (lastBatchMode.second && lastBatchMode.first == batchMode)
         return;
      lastBatchMode = {batchMode, true};
   }

   auto log = [](std::string_view message) {
      oocxcoutI(static_cast<RooAbsArg *>(nullptr), Fitting) << message << std::endl;
   };

   if (batchMode == RooFit::BatchModeOption::Cuda && !RooBatchCompute::dispatchCUDA) {
      throw std::runtime_error(std::string("In: ") + __func__ + "(), " + __FILE__ + ":" + __LINE__ +
                               ": Cuda implementation of the computing library is not available\n");
   }
   if (RooBatchCompute::dispatchCPU->architecture() == RooBatchCompute::Architecture::GENERIC) {
      log("using generic CPU library compiled with no vectorizations");
   } else {
      log(std::string("using CPU computation library compiled with -m") +
          RooBatchCompute::dispatchCPU->architectureName());
   }
   if (batchMode == RooFit::BatchModeOption::Cuda) {
      log("using CUDA computation library");
   }
}
