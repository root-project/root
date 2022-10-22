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

#include <string>

using ROOT::Experimental::RooFitDriver;
using ROOT::Experimental::RooNLLVarNew;

namespace {

std::unique_ptr<RooAbsArg> createSimultaneousNLL(RooSimultaneous const &simPdf, RooArgSet &observables, bool isExtended,
                                                 std::string const &rangeName, bool doOffset, bool splitRange)
{
   // Prepare the NLL terms for each component
   RooArgList nllTerms;
   RooArgSet newObservables;
   for (auto const &catItem : simPdf.indexCat()) {
      std::string const &catName = catItem.first;
      if (RooAbsPdf *pdf = simPdf.getPdf(catName.c_str())) {
         auto name = std::string("nll_") + pdf->GetName();
         if (!rangeName.empty()) {
            pdf->setNormRange(RooHelpers::getRangeNameForSimComponent(rangeName, splitRange, catName).c_str());
         }
         auto nll = std::make_unique<RooNLLVarNew>(name.c_str(), name.c_str(), *pdf, observables, isExtended, doOffset);
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
                     RooAbsCategory const *indexCatForSplitting, bool splitRange, bool takeGlobalObservablesFromData)
      : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"}, _driver{std::move(driver)},
        _topNode("topNode", "top node", this, _driver->topNode()), _rangeName{rangeName},
        _indexCatForSplitting{indexCatForSplitting}, _splitRange{splitRange}, _takeGlobalObservablesFromData{
                                                                                 takeGlobalObservablesFromData}
   {
   }

   RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = nullptr)
      : RooAbsReal{other, name}, _driver{other._driver},
        _topNode("topNode", this, other._topNode), _data{other._data}, _parameters{other._parameters},
        _rangeName{other._rangeName}, _indexCatForSplitting{other._indexCatForSplitting},
        _splitRange{other._splitRange}, _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
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
      _driver->setData(*_data, _rangeName, _indexCatForSplitting, _splitRange, /*skipZeroWeights=*/true,
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
   bool _splitRange = false;
   const bool _takeGlobalObservablesFromData;
};

} // namespace

std::unique_ptr<RooAbsReal> RooFit::BatchModeHelpers::createNLL(std::unique_ptr<RooAbsPdf> &&pdf, RooAbsData &data,
                                                                std::unique_ptr<RooAbsReal> &&constraints,
                                                                std::string const &rangeName, RooArgSet const &projDeps,
                                                                bool isExtended, double integrateOverBinsPrecision,
                                                                RooFit::BatchModeOption batchMode, bool doOffset,
                                                                bool splitRange, bool takeGlobalObservablesFromData)
{
   RooArgSet observables;
   pdf->getObservables(data.get(), observables);
   observables.remove(projDeps, true, true);

   // Set the normalization range
   if (!rangeName.empty()) {
      pdf->setNormRange(rangeName.c_str());
   }

   oocxcoutI(pdf.get(), Fitting) << "RooAbsPdf::fitTo(" << pdf->GetName()
                                 << ") fixing normalization set for coefficient determination to observables in data"
                                 << "\n";
   pdf->fixAddCoefNormalization(observables, false);

   // Deal with the IntegrateBins argument
   RooArgList binSamplingPdfs;
   std::unique_ptr<RooAbsPdf> wrappedPdf = RooBinSamplingPdf::create(*pdf, data, integrateOverBinsPrecision);
   RooAbsPdf &finalPdf = wrappedPdf ? *wrappedPdf : *pdf;
   if (wrappedPdf) {
      binSamplingPdfs.addOwned(std::move(wrappedPdf));
   }
   // Done dealing with the IntegrateBins option

   RooArgList nllTerms;

   RooAbsCategory const *indexCatForSplitting = nullptr;
   if (auto simPdf = dynamic_cast<RooSimultaneous *>(&finalPdf)) {
      indexCatForSplitting = &simPdf->indexCat();
      simPdf->wrapPdfsInBinSamplingPdfs(data, integrateOverBinsPrecision);
      // Warning! This mutates "observables"
      nllTerms.addOwned(createSimultaneousNLL(*simPdf, observables, isExtended, rangeName, doOffset, splitRange));
   } else {
      nllTerms.addOwned(
         std::make_unique<RooNLLVarNew>("RooNLLVarNew", "RooNLLVarNew", finalPdf, observables, isExtended, doOffset));
   }
   if (constraints) {
      nllTerms.addOwned(std::move(constraints));
   }

   std::string nllName = std::string("nll_") + pdf->GetName() + "_" + data.GetName();
   auto nll = std::make_unique<RooAddition>(nllName.c_str(), nllName.c_str(), nllTerms);
   nll->addOwnedComponents(std::move(binSamplingPdfs));
   nll->addOwnedComponents(std::move(nllTerms));

   auto driver = std::make_unique<RooFitDriver>(*nll, observables, batchMode);

   auto driverWrapper = std::make_unique<RooAbsRealWrapper>(std::move(driver), rangeName, indexCatForSplitting,
                                                            splitRange, takeGlobalObservablesFromData);
   driverWrapper->setData(data, false);
   driverWrapper->addOwnedComponents(std::move(nll));
   driverWrapper->addOwnedComponents(std::move(pdf));

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
