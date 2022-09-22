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
#include <RooBatchCompute.h>
#include <RooBinSamplingPdf.h>
#include <RooConstraintSum.h>
#include <RooDataSet.h>
#include <RooFitDriver.h>
#include <RooNLLVarNew.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooFitDriver.h>

#include <ROOT/StringUtils.hxx>

#include <string>

using ROOT::Experimental::RooFitDriver;
using ROOT::Experimental::RooNLLVarNew;

namespace {

std::unique_ptr<RooAbsArg> createSimultaneousNLL(RooSimultaneous const &simPdf, RooArgSet &observables, bool isExtended,
                                                 std::string const &rangeName, bool doOffset)
{
   // Prepare the NLLTerms for each component
   RooArgList nllTerms;
   RooArgSet newObservables;
   for (auto const &catItem : simPdf.indexCat()) {
      auto const &catName = catItem.first;
      if (RooAbsPdf *pdf = simPdf.getPdf(catName.c_str())) {
         auto name = std::string("nll_") + pdf->GetName();
         auto nll = std::make_unique<RooNLLVarNew>(name.c_str(), name.c_str(), *pdf, observables, isExtended, rangeName,
                                                   doOffset);
         // Rename the observables and weights
         newObservables.add(nll->prefixObservableAndWeightNames(std::string("_") + catName + "_"));
         nllTerms.addOwned(std::move(nll));
      }
   }

   observables.clear();
   observables.add(newObservables);

   // Time to sum the NLLs
   auto nll = std::make_unique<RooAddition>("mynll", "mynll", nllTerms);
   nll->addOwnedComponents(std::move(nllTerms));
   return nll;
}

class RooAbsRealWrapper final : public RooAbsReal {
public:
   RooAbsRealWrapper(std::unique_ptr<RooFitDriver> driver, std::string const &rangeName,
                     RooAbsCategory const *indexCatForSplitting, bool takeGlobalObservablesFromData)
      : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"}, _driver{std::move(driver)},
        _topNode("topNode", "top node", this, _driver->topNode()), _rangeName{rangeName},
        _indexCatForSplitting{indexCatForSplitting}, _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
   {
   }

   RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = nullptr)
      : RooAbsReal{other, name}, _driver{other._driver},
        _topNode("topNode", this, other._topNode), _data{other._data}, _parameters{other._parameters},
        _rangeName{other._rangeName}, _indexCatForSplitting{other._indexCatForSplitting},
        _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
   {
   }

   TObject *clone(const char *newname) const override { return new RooAbsRealWrapper(*this, newname); }

   double defaultErrorLevel() const override { return _driver->topNode().defaultErrorLevel(); }

   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool /*stripDisconnected*/) const override
   {
      outputSet.add(_parameters);
      if (observables) {
         outputSet.remove(*observables);
      }
      // If we take the global observables as data, we have to return these as
      // parameters instead of the parameters in the model. Otherwise, the
      // constant parameters in the fit result that are global observables will
      // not have the right values.
      if (_takeGlobalObservablesFromData && _data->getGlobalObservables()) {
         outputSet.replace(*_data->getGlobalObservables());
      }
      return false;
   }

   bool setData(RooAbsData &data, bool /*cloneData*/) override
   {
      _data = &data;
      _driver->topNode().getParameters(_data->get(), _parameters, true);
      _driver->setData(*_data, _rangeName, _indexCatForSplitting, /*skipZeroWeights=*/true,
                       _takeGlobalObservablesFromData);
      return true;
   }

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override
   {
      const_cast<RooAbsReal &>(_driver->topNode()).applyWeightSquared(flag);
   }

protected:
   double evaluate() const override { return _driver ? _driver->getVal() : 0.0; }

private:
   std::shared_ptr<RooFitDriver> _driver;
   RooRealProxy _topNode;
   RooAbsData *_data = nullptr;
   RooArgSet _parameters;
   std::string _rangeName;
   RooAbsCategory const *_indexCatForSplitting = nullptr;
   const bool _takeGlobalObservablesFromData;
};

} // namespace

std::unique_ptr<RooAbsReal>
RooFit::BatchModeHelpers::createNLL(RooAbsPdf &pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
                                    std::string const &rangeName, std::string const &addCoefRangeName,
                                    RooArgSet const &projDeps, bool isExtended, double integrateOverBinsPrecision,
                                    RooFit::BatchModeOption batchMode, bool doOffset,
                                    bool takeGlobalObservablesFromData)
{
   // Clone PDF and reattach to original parameters
   std::unique_ptr<RooAbsPdf> pdfClone{static_cast<RooAbsPdf *>(pdf.cloneTree())};
   {
      RooArgSet origParams;
      pdf.getParameters(data.get(), origParams);
      pdfClone->recursiveRedirectServers(origParams);
   }

   RooArgSet observables;
   RooArgSet obsClones;
   pdfClone->getObservables(data.get(), obsClones);
   pdf.getObservables(data.get(), observables);
   observables.remove(projDeps, true, true);
   obsClones.remove(projDeps, true, true);

   oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                            << ") fixing normalization set for coefficient determination to observables in data"
                            << "\n";
   pdfClone->fixAddCoefNormalization(obsClones, false);
   pdf.fixAddCoefNormalization(observables, false);
   if (!addCoefRangeName.empty()) {
      oocxcoutI(&pdf, Fitting) << "RooAbsPdf::fitTo(" << pdf.GetName()
                               << ") fixing interpretation of coefficients of any component to range "
                               << addCoefRangeName << "\n";
      pdfClone->fixAddCoefRange(addCoefRangeName.c_str(), false);
      pdf.fixAddCoefRange(addCoefRangeName.c_str(), false);
   }

   // Deal with the IntegrateBins argument
   RooArgList binSamplingPdfs;
   std::unique_ptr<RooAbsPdf> wrappedPdf = RooBinSamplingPdf::create(*pdfClone, data, integrateOverBinsPrecision);
   RooAbsPdf &finalPdf = wrappedPdf ? *wrappedPdf : *pdfClone;
   if (wrappedPdf) {
      binSamplingPdfs.addOwned(std::move(wrappedPdf));
   }
   // Done dealing with the IntegrateBins option

   RooArgList nllTerms;

   RooAbsCategory const *indexCatForSplitting = nullptr;
   if (auto simPdf = dynamic_cast<RooSimultaneous *>(&finalPdf)) {
      indexCatForSplitting = &simPdf->indexCat();
      simPdf->wrapPdfsInBinSamplingPdfs(data, integrateOverBinsPrecision);
      // Warning! This mutates "obsClones"
      nllTerms.addOwned(createSimultaneousNLL(*simPdf, obsClones, isExtended, rangeName, doOffset));
   } else {
      nllTerms.addOwned(std::make_unique<RooNLLVarNew>("RooNLLVarNew", "RooNLLVarNew", finalPdf, obsClones, isExtended,
                                                       rangeName, doOffset));
   }
   if (constraints) {
      nllTerms.addOwned(std::move(constraints));
   }

   std::string nllName = std::string("nll_") + pdfClone->GetName() + "_" + data.GetName();
   auto nll = std::make_unique<RooAddition>(nllName.c_str(), nllName.c_str(), nllTerms);
   nll->addOwnedComponents(std::move(binSamplingPdfs));
   nll->addOwnedComponents(std::move(nllTerms));

   auto driver = std::make_unique<RooFitDriver>(*nll, obsClones, batchMode);

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
         for (auto *observable : static_range_cast<RooRealVar *>(obsClones)) {
            observable->setRange(fitrangeValueSubrange.c_str(), observable->getMin(subrange.c_str()),
                                 observable->getMax(subrange.c_str()));
         }
      }
      pdf.setStringAttribute("fitrange", fitrangeValue.substr(0, fitrangeValue.size() - 1).c_str());
   }

   auto driverWrapper = std::make_unique<RooAbsRealWrapper>(std::move(driver), rangeName, indexCatForSplitting,
                                                            takeGlobalObservablesFromData);
   driverWrapper->setData(data, false);
   driverWrapper->addOwnedComponents(std::move(nll));
   driverWrapper->addOwnedComponents(std::move(pdfClone));

   return driverWrapper;
}

void RooFit::BatchModeHelpers::logArchitectureInfo(RooFit::BatchModeOption batchMode)
{
   // We have to exit early if the message stream is not active. Otherwise it's
   // possible that this funciton skips logging because it thinks it has
   // already logged, but actually it didn't.
   if (!RooMsgService::instance().isActive(static_cast<RooAbsArg *>(nullptr), RooFit::Fitting, RooFit::INFO)) {
      return;
   }

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
