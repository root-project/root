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
\file RooEvaluatorWrapper.cxx
\class RooEvaluatorWrapper
\ingroup Roofitcore

Wraps a RooFit::Evaluator that evaluates a RooAbsReal back into a RooAbsReal.
**/

#include "RooEvaluatorWrapper.h"

#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include "RooFit/BatchModeDataHelpers.h"
#include <RooSimultaneous.h>

#include <TList.h>

RooEvaluatorWrapper::RooEvaluatorWrapper(RooAbsReal &topNode, std::unique_ptr<RooFit::Evaluator> evaluator,
                                         std::string const &rangeName, RooSimultaneous const *simPdf,
                                         bool takeGlobalObservablesFromData)
   : RooAbsReal{"RooEvaluatorWrapper", "RooEvaluatorWrapper"},
     _evaluator{std::move(evaluator)},
     _topNode("topNode", "top node", this, topNode),
     _rangeName{rangeName},
     _simPdf{simPdf},
     _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
}

RooEvaluatorWrapper::RooEvaluatorWrapper(const RooEvaluatorWrapper &other, const char *name)
   : RooAbsReal{other, name},
     _evaluator{other._evaluator},
     _topNode("topNode", this, other._topNode),
     _data{other._data},
     _rangeName{other._rangeName},
     _simPdf{other._simPdf},
     _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
{
}

bool RooEvaluatorWrapper::getParameters(const RooArgSet *observables, RooArgSet &outputSet,
                                        bool /*stripDisconnected*/) const
{
   outputSet.add(_evaluator->getParameters());
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

bool RooEvaluatorWrapper::setData(RooAbsData &data, bool /*cloneData*/)
{
   _data = &data;
   std::stack<std::vector<double>>{}.swap(_vectorBuffers);
   auto dataSpans = RooFit::BatchModeDataHelpers::getDataSpans(*_data, _rangeName, _simPdf, /*skipZeroWeights=*/true,
                                                               _takeGlobalObservablesFromData, _vectorBuffers);
   for (auto const &item : dataSpans) {
      _evaluator->setInput(item.first->GetName(), item.second, false);
   }
   return true;
}

/// \endcond
