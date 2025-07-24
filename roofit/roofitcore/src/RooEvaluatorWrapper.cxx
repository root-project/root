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
#include <RooConstVar.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

RooEvaluatorWrapper::RooEvaluatorWrapper(RooAbsReal &topNode, RooAbsData *data, bool useGPU,
                                         std::string const &rangeName, RooAbsPdf const *pdf,
                                         bool takeGlobalObservablesFromData)
   : RooAbsReal{"RooEvaluatorWrapper", "RooEvaluatorWrapper"},
     _evaluator{std::make_unique<RooFit::Evaluator>(topNode, useGPU)},
     _topNode("topNode", "top node", this, topNode, false, false),
     _data{data},
     _paramSet("paramSet", "Set of parameters", this),
     _rangeName{rangeName},
     _pdf{pdf},
     _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
   if (data) {
      setData(*data, false);
   }
   _paramSet.add(_evaluator->getParameters());
   for (auto const &item : _dataSpans) {
      _paramSet.remove(*_paramSet.find(item.first->GetName()));
   }
}

RooEvaluatorWrapper::RooEvaluatorWrapper(const RooEvaluatorWrapper &other, const char *name)
   : RooAbsReal{other, name},
     _evaluator{other._evaluator},
     _topNode("topNode", this, other._topNode),
     _data{other._data},
     _paramSet("paramSet", "Set of parameters", this),
     _rangeName{other._rangeName},
     _pdf{other._pdf},
     _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData},
     _dataSpans{other._dataSpans}
{
   _paramSet.add(other._paramSet);
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
                                        bool stripDisconnected) const
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

   // The disconnected parameters are stripped away in
   // RooAbsArg::getParametersHook(), that is only called in the original
   // RooAbsArg::getParameters() implementation. So he have to call it to
   // identify disconnected parameters to remove.
   if (stripDisconnected) {
      RooArgSet paramsStripped;
      _topNode->getParameters(observables, paramsStripped, true);
      RooArgSet toRemove;
      for (RooAbsArg *param : outputSet) {
         if (!paramsStripped.find(param->GetName())) {
            toRemove.add(*param);
         }
      }
      outputSet.remove(toRemove, /*silent*/ false, /*matchByNameOnly*/ true);
   }

   return false;
}

bool RooEvaluatorWrapper::setData(RooAbsData &data, bool /*cloneData*/)
{
   // To make things easiear for RooFit, we only support resetting with
   // datasets that have the same structure, e.g. the same columns and global
   // observables. This is anyway the usecase: resetting same-structured data
   // when iterating over toys.
   constexpr auto errMsg = "Error in RooAbsReal::setData(): only resetting with same-structured data is supported.";

   _data = &data;
   bool isInitializing = _paramSet.empty();
   const std::size_t oldSize = _dataSpans.size();

   std::stack<std::vector<double>>{}.swap(_vectorBuffers);
   bool skipZeroWeights = !_pdf || !_pdf->getAttribute("BinnedLikelihoodActive");
   _dataSpans =
      RooFit::BatchModeDataHelpers::getDataSpans(*_data, _rangeName, dynamic_cast<RooSimultaneous const *>(_pdf),
                                                 skipZeroWeights, _takeGlobalObservablesFromData, _vectorBuffers);
   if (!isInitializing && _dataSpans.size() != oldSize) {
      coutE(DataHandling) << errMsg << std::endl;
      throw std::runtime_error(errMsg);
   }
   for (auto const &item : _dataSpans) {
      const char *name = item.first->GetName();
      _evaluator->setInput(name, item.second, false);
      if (_paramSet.find(name)) {
         coutE(DataHandling) << errMsg << std::endl;
         throw std::runtime_error(errMsg);
      }
   }
   return true;
}

/// \endcond
