/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\internal
\file RooFitDriverWrapper.cxx
\class RooFitDriverWrapper
\ingroup Roofitcore

Wraps a RooFitDriver that evaluates a RooAbsReal back into a RooAbsReal.
**/

#include "RooFitDriverWrapper.h"

#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include "RooFit/BatchModeDataHelpers.h"
#include <RooSimultaneous.h>

#include <TList.h>

RooFitDriverWrapper::RooFitDriverWrapper(RooAbsReal &topNode, std::unique_ptr<ROOT::Experimental::RooFitDriver> driver,
                                         std::string const &rangeName, RooSimultaneous const *simPdf,
                                         bool takeGlobalObservablesFromData)
   : RooAbsReal{"RooFitDriverWrapper", "RooFitDriverWrapper"},
     _driver{std::move(driver)},
     _topNode("topNode", "top node", this, topNode),
     _rangeName{rangeName},
     _simPdf{simPdf},
     _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
}

RooFitDriverWrapper::RooFitDriverWrapper(const RooFitDriverWrapper &other, const char *name)
   : RooAbsReal{other, name},
     _driver{other._driver},
     _topNode("topNode", this, other._topNode),
     _data{other._data},
     _rangeName{other._rangeName},
     _simPdf{other._simPdf},
     _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
{
}

bool RooFitDriverWrapper::getParameters(const RooArgSet *observables, RooArgSet &outputSet,
                                        bool /*stripDisconnected*/) const
{
   outputSet.add(_driver->getParameters());
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

bool RooFitDriverWrapper::setData(RooAbsData &data, bool /*cloneData*/)
{
   _data = &data;
   std::stack<std::vector<double>>{}.swap(_vectorBuffers);
   auto dataSpans = RooFit::BatchModeDataHelpers::getDataSpans(*_data, _rangeName, _simPdf, /*skipZeroWeights=*/true,
                                                               _takeGlobalObservablesFromData, _vectorBuffers);
   for (auto const &item : dataSpans) {
      _driver->setInput(item.first->GetName(), item.second, false);
   }
   return true;
}

/// \endcond
