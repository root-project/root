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
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include <TList.h>

RooEvaluatorWrapper::RooEvaluatorWrapper(RooAbsReal &topNode, std::unique_ptr<RooFit::Evaluator> evaluator,
                                         std::string const &rangeName, RooAbsPdf const *pdf,
                                         bool takeGlobalObservablesFromData)
   : RooAbsReal{"RooEvaluatorWrapper", "RooEvaluatorWrapper"},
     _evaluator{std::move(evaluator)},
     _topNode("topNode", "top node", this, topNode),
     _rangeName{rangeName},
     _pdf{pdf},
     _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
}

RooEvaluatorWrapper::RooEvaluatorWrapper(const RooEvaluatorWrapper &other, const char *name)
   : RooAbsReal{other, name},
     _evaluator{other._evaluator},
     _topNode("topNode", this, other._topNode),
     _data{other._data},
     _rangeName{other._rangeName},
     _pdf{other._pdf},
     _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData},
     _dataSpans{other._dataSpans}
{
}

double RooEvaluatorWrapper::evaluate() const
{
   if (!_evaluator)
      return 0.0;

   _evaluator->setOffsetMode(hideOffset() ? RooFit::EvalContext::OffsetMode::WithoutOffset
                                          : RooFit::EvalContext::OffsetMode::WithOffset);

   return _evaluator->run()[0];
}

bool RooEvaluatorWrapper::getParameters(const RooArgSet *observables, RooArgSet &outputSet,
                                        bool /*stripDisconnected*/) const
{
   outputSet.add(_evaluator->getParameters());
   if (observables) {
      outputSet.remove(*observables, /*silent*/ false, /*matchByNameOnly*/ true);
   }
   // Exclude the data variables from the parameters which are not global observables
   for (auto const &item : _dataSpans) {
      if (_data->getGlobalObservables() && _data->getGlobalObservables()->find(item.first->GetName())) {
         continue;
      }
      RooAbsArg *found = outputSet.find(item.first->GetName());
      if (found) {
         outputSet.remove(*found);
      }
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
   bool skipZeroWeights = !_pdf || !_pdf->getAttribute("BinnedLikelihoodActive");
   _dataSpans = RooFit::Detail::BatchModeDataHelpers::getDataSpans(
      *_data, _rangeName, dynamic_cast<RooSimultaneous const *>(_pdf), skipZeroWeights, _takeGlobalObservablesFromData,
      _vectorBuffers);
   for (auto const &item : _dataSpans) {
      _evaluator->setInput(item.first->GetName(), item.second, false);
   }
   return true;
}

/// \endcond
