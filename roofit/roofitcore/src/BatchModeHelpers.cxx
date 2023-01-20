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
#include "RooFitDriver.h"
#include "RooNLLVarNew.h"
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include <string>

using ROOT::Experimental::RooFitDriver;
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

class RooAbsRealWrapper final : public RooAbsReal {
public:
   RooAbsRealWrapper(std::unique_ptr<RooFitDriver> driver, std::string const &rangeName, RooSimultaneous const *simPdf,
                     bool takeGlobalObservablesFromData)
      : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"}, _driver{std::move(driver)},
        _topNode("topNode", "top node", this, _driver->topNode()), _rangeName{rangeName}, _simPdf{simPdf},
        _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
   {
   }

   RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = nullptr)
      : RooAbsReal{other, name}, _driver{other._driver},
        _topNode("topNode", this, other._topNode), _data{other._data}, _parameters{other._parameters},
        _rangeName{other._rangeName}, _simPdf{other._simPdf}, _takeGlobalObservablesFromData{
                                                                 other._takeGlobalObservablesFromData}
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

      // Figure out what are the parameters for the current dataset
      _parameters.clear();
      RooArgSet params;
      _driver->topNode().getParameters(_data->get(), params, true);
      for (RooAbsArg *param : params) {
         if (!param->getAttribute("__obs__")) {
            _parameters.add(*param);
         }
      }

      _driver->setData(*_data, _rangeName, _simPdf, /*skipZeroWeights=*/true, _takeGlobalObservablesFromData);
      return true;
   }

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override
   {
      const_cast<RooAbsReal &>(_driver->topNode()).applyWeightSquared(flag);
   }

   void printMultiline(std::ostream &os, Int_t /*contents*/, bool /*verbose*/ = false,
                       TString /*indent*/ = "") const override
   {
      _driver->print(os);
   }

protected:
   double evaluate() const override { return _driver ? _driver->getVal() : 0.0; }

private:
   std::shared_ptr<RooFitDriver> _driver;
   RooRealProxy _topNode;
   RooAbsData *_data = nullptr;
   RooArgSet _parameters;
   std::string _rangeName;
   RooSimultaneous const *_simPdf = nullptr;
   const bool _takeGlobalObservablesFromData;
};

} // namespace

std::unique_ptr<RooAbsReal>
RooFit::BatchModeHelpers::createNLL(std::unique_ptr<RooAbsPdf> &&pdf, RooAbsData &data,
                                    std::unique_ptr<RooAbsReal> &&constraints, std::string const &rangeName,
                                    RooArgSet const &projDeps, bool isExtended, double integrateOverBinsPrecision,
                                    RooFit::BatchModeOption batchMode, RooFit::OffsetMode offset,
                                    bool takeGlobalObservablesFromData)
{
   if (constraints) {
      // Redirect the global observables to the ones from the dataset if applicable.
      constraints->setData(data, false);

      // The computation graph for the constraints is very small, no need to do
      // the tracking of clean and dirty nodes here.
      constraints->setOperMode(RooAbsArg::ADirty);
   }

   RooArgSet observables;
   pdf->getObservables(data.get(), observables);
   observables.remove(projDeps, true, true);

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

   std::string nllName = std::string("nll_") + pdf->GetName() + "_" + data.GetName();
   auto nll = std::make_unique<RooAddition>(nllName.c_str(), nllName.c_str(), nllTerms);
   nll->addOwnedComponents(std::move(binSamplingPdfs));
   nll->addOwnedComponents(std::move(nllTerms));

   auto driver = std::make_unique<RooFitDriver>(*nll, batchMode);

   auto driverWrapper =
      std::make_unique<RooAbsRealWrapper>(std::move(driver), rangeName, simPdf, takeGlobalObservablesFromData);
   driverWrapper->setData(data, false);
   driverWrapper->addOwnedComponents(std::move(nll));
   driverWrapper->addOwnedComponents(std::move(pdf));

   return driverWrapper;
}

void RooFit::BatchModeHelpers::logArchitectureInfo(RooFit::BatchModeOption batchMode)
{
   // We have to exit early if the message stream is not active. Otherwise it's
   // possible that this function skips logging because it thinks it has
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
