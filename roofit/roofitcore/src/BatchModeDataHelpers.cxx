/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/BatchModeDataHelpers.h>

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooNLLVarNew.h>
#include <RunContext.h>

#include <ROOT/StringUtils.hxx>

#include <numeric>

namespace {

void splitByCategory(std::map<const TNamed *, RooSpan<const double>> &dataSpans, RooAbsCategory const &category,
                     std::stack<std::vector<double>> &buffers)
{
   std::stack<std::vector<double>> oldBuffers;
   std::swap(buffers, oldBuffers);

   auto catVals = dataSpans.at(category.namePtr());

   std::map<const TNamed *, RooSpan<const double>> dataMapSplit;

   for (auto const &dataMapItem : dataSpans) {

      auto const &varNamePtr = dataMapItem.first;
      auto const &xVals = dataMapItem.second;

      if (varNamePtr == category.namePtr())
         continue;

      std::map<RooAbsCategory::value_type, std::vector<double>> valuesMap;

      if (xVals.size() == 1) {
         // If the span is of size one, we will replicate it for each category
         // component instead of splitting is up by category value.
         for (auto const &catItem : category) {
            valuesMap[catItem.second].push_back(xVals[0]);
         }
      } else {
         for (std::size_t i = 0; i < xVals.size(); ++i) {
            valuesMap[catVals[i]].push_back(xVals[i]);
         }
      }

      for (auto const &item : valuesMap) {
         RooAbsCategory::value_type index = item.first;
         auto variableName = std::string("_") + category.lookupName(index) + "_" + varNamePtr->GetName();
         auto variableNamePtr = RooNameReg::instance().constPtr(variableName.c_str());

         buffers.emplace(std::move(item.second));
         auto const &values = buffers.top();
         dataMapSplit[variableNamePtr] = RooSpan<const double>(values.data(), values.size());
      }
   }

   dataSpans = std::move(dataMapSplit);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Extract all content from a RooFit datasets as a map of spans.
// Spans with the weights and squared weights will be also stored in the map,
// keyed with the names `_weight` and the `_weight_sumW2`. If the dataset is
// unweighted, these weight spans will only contain the single value `1.0`.
// Entries with zero weight will be skipped.
//
// \return A `std::map` with spans keyed to name pointers.
// \param[in] data The input dataset.
// \param[in] rangeName Select only entries from the data in a given range
//            (empty string for no range).
// \param[in] indexCat If not `nullptr`, each span is spit up by this category,
//            with the new names prefixed by the category component name
//            surrounded by underscores. For example, if you have a category
//            with `signal` and `control` samples, the span for a variable `x`
//            will be split in two spans `_signal_x` and `_control_x`.
// \param[in] buffers Pass here an empty stack of `double` vectors, which will
//            be used as memory for the data if the memory in the dataset
//            object can't be used directly (e.g. because you used the range
//            selection or the splitting by categories).
std::map<const TNamed *, RooSpan<const double>>
RooFit::BatchModeDataHelpers::getDataSpans(RooAbsData const &data, std::string_view rangeName,
                                           RooAbsCategory const *indexCat, std::stack<std::vector<double>> &buffers)
{
   std::map<const TNamed *, RooSpan<const double>> dataSpans; // output variable

   std::size_t nEvents = static_cast<size_t>(data.numEntries());

   // We also want to support empty datasets: in this case the
   // RooFitDriver::Dataset is not filled with anything.
   if (nEvents == 0)
      return dataSpans;

   if (!buffers.empty()) {
      throw std::invalid_argument("The buffers container must be empty when passed to getDataSpans()!");
   }

   auto &nameReg = RooNameReg::instance();

   auto weight = data.getWeightBatch(0, nEvents, /*sumW2=*/false);
   auto weightSumW2 = data.getWeightBatch(0, nEvents, /*sumW2=*/true);

   std::vector<bool> hasZeroWeight;
   hasZeroWeight.resize(nEvents);
   std::size_t nNonZeroWeight = 0;

   // Add weights to the datamap. They should have the names expected by the
   // RooNLLVarNew. We also add the sumW2 weights here under a different name,
   // so we can apply the sumW2 correction by easily swapping the spans.
   {
      buffers.emplace();
      auto &buffer = buffers.top();
      buffers.emplace();
      auto &bufferSumW2 = buffers.top();
      if (weight.empty()) {
         // If the dataset has no weight, we fill the data spans with a scalar
         // unity weight so we don't need to check for the existance of weights
         // later in the likelihood.
         buffer.push_back(1.0);
         bufferSumW2.push_back(1.0);
         weight = RooSpan<const double>(buffer.data(), 1);
         weightSumW2 = RooSpan<const double>(bufferSumW2.data(), 1);
         nNonZeroWeight = nEvents;
      } else {
         buffer.reserve(nEvents);
         bufferSumW2.reserve(nEvents);
         for (std::size_t i = 0; i < nEvents; ++i) {
            if (weight[i] != 0) {
               buffer.push_back(weight[i]);
               bufferSumW2.push_back(weightSumW2[i]);
               ++nNonZeroWeight;
            } else {
               hasZeroWeight[i] = true;
            }
         }
         weight = RooSpan<const double>(buffer.data(), nNonZeroWeight);
         weightSumW2 = RooSpan<const double>(bufferSumW2.data(), nNonZeroWeight);
      }
      using namespace ROOT::Experimental;
      dataSpans[nameReg.constPtr(RooNLLVarNew::weightVarName)] = weight;
      dataSpans[nameReg.constPtr(RooNLLVarNew::weightVarNameSumW2)] = weightSumW2;
   }

   // fill the RunContext with the observable data and map the observables
   // by namePtr in order to replace their memory addresses later, with
   // the ones from the variables that are actually in the computation graph.
   RooBatchCompute::RunContext evalData;
   data.getBatches(evalData, 0, nEvents);
   for (auto const &item : evalData.spans) {

      const TNamed *namePtr = nameReg.constPtr(item.first->GetName());
      RooSpan<const double> span{item.second};

      buffers.emplace();
      auto &buffer = buffers.top();
      buffer.reserve(nNonZeroWeight);

      for (std::size_t i = 0; i < nEvents; ++i) {
         if (!hasZeroWeight[i]) {
            buffer.push_back(span[i]);
         }
      }
      dataSpans[namePtr] = RooSpan<const double>(buffer.data(), buffer.size());
   }

   // Get the category batches and cast the also to double branches to put in
   // the data map
   for (auto const &item : data.getCategoryBatches(0, nEvents)) {

      const TNamed *namePtr = nameReg.constPtr(item.first.c_str());
      RooSpan<const RooAbsCategory::value_type> intSpan{item.second};

      buffers.emplace();
      auto &buffer = buffers.top();
      buffer.reserve(nNonZeroWeight);

      for (std::size_t i = 0; i < nEvents; ++i) {
         if (!hasZeroWeight[i]) {
            buffer.push_back(static_cast<double>(intSpan[i]));
         }
      }
      dataSpans[namePtr] = RooSpan<const double>(buffer.data(), buffer.size());
   }

   nEvents = nNonZeroWeight;

   // Now we have do do the range selection
   if (!rangeName.empty()) {
      // figure out which events are in the range
      std::vector<bool> isInRange(nEvents, false);
      for (auto const &range : ROOT::Split(rangeName, ",")) {
         std::vector<bool> isInSubRange(nEvents, true);
         for (auto *observable : dynamic_range_cast<RooAbsRealLValue *>(*data.get())) {
            // If the observables is not real-valued, it will not be considered for the range selection
            if (!observable)
               continue;
            observable->inRange({dataSpans.at(observable->namePtr()).data(), nEvents}, range, isInSubRange);
         }
         for (std::size_t i = 0; i < isInSubRange.size(); ++i) {
            isInRange[i] = isInRange[i] | isInSubRange[i];
         }
      }

      // reset the number of events
      nEvents = std::accumulate(isInRange.begin(), isInRange.end(), 0);

      // do the data reduction in the data map
      for (auto const &item : dataSpans) {
         auto const &allValues = item.second;
         if (allValues.size() == 1) {
            continue;
         }
         buffers.emplace(nEvents);
         double *buffer = buffers.top().data();
         std::size_t j = 0;
         for (std::size_t i = 0; i < isInRange.size(); ++i) {
            if (isInRange[i]) {
               buffer[j] = allValues[i];
               ++j;
            }
         }
         dataSpans[item.first] = RooSpan<const double>{buffer, nEvents};
      }
   }

   if (indexCat) {
      splitByCategory(dataSpans, *indexCat, buffers);
   }

   return dataSpans;
}
