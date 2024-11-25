/// \cond ROOFIT_INTERNAL

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

#include "RooFit/BatchModeDataHelpers.h"

#include <RooAbsData.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include "RooFitImplHelpers.h"
#include "RooFit/Detail/RooNLLVarNew.h"

#include <ROOT/StringUtils.hxx>

#include <numeric>

namespace {

// To avoid deleted move assignment.
template <class T>
void assignSpan(std::span<T> &to, std::span<T> const &from)
{
   to = from;
}

std::map<RooFit::Detail::DataKey, std::span<const double>>
getSingleDataSpans(RooAbsData const &data, std::string_view rangeName, std::string const &prefix,
                   std::stack<std::vector<double>> &buffers, bool skipZeroWeights)
{
   std::map<RooFit::Detail::DataKey, std::span<const double>> dataSpans; // output variable

   auto &nameReg = RooNameReg::instance();

   auto insert = [&](const char *key, std::span<const double> span) {
      const TNamed *namePtr = nameReg.constPtr((prefix + key).c_str());
      dataSpans[namePtr] = span;
   };

   auto retrieve = [&](const char *key) {
      const TNamed *namePtr = nameReg.constPtr((prefix + key).c_str());
      return dataSpans.at(namePtr);
   };

   std::size_t nEvents = static_cast<size_t>(data.numEntries());

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
         // unity weight so we don't need to check for the existence of weights
         // later in the likelihood.
         buffer.push_back(1.0);
         bufferSumW2.push_back(1.0);
         assignSpan(weight, {buffer.data(), 1});
         assignSpan(weightSumW2, {bufferSumW2.data(), 1});
         nNonZeroWeight = nEvents;
      } else {
         buffer.reserve(nEvents);
         bufferSumW2.reserve(nEvents);
         for (std::size_t i = 0; i < nEvents; ++i) {
            if (!skipZeroWeights || weight[i] != 0) {
               buffer.push_back(weight[i]);
               bufferSumW2.push_back(weightSumW2[i]);
               ++nNonZeroWeight;
            } else {
               hasZeroWeight[i] = true;
            }
         }
         assignSpan(weight, {buffer.data(), nNonZeroWeight});
         assignSpan(weightSumW2, {bufferSumW2.data(), nNonZeroWeight});
      }
      insert(RooFit::Detail::RooNLLVarNew::weightVarName, weight);
      insert(RooFit::Detail::RooNLLVarNew::weightVarNameSumW2, weightSumW2);
   }

   // Get the real-valued batches and cast the also to double branches to put in
   // the data map
   for (auto const &item : data.getBatches(0, nEvents)) {

      std::span<const double> span{item.second};

      buffers.emplace();
      auto &buffer = buffers.top();
      buffer.reserve(nNonZeroWeight);

      for (std::size_t i = 0; i < nEvents; ++i) {
         if (!hasZeroWeight[i]) {
            buffer.push_back(span[i]);
         }
      }
      insert(item.first->GetName(), {buffer.data(), buffer.size()});
   }

   // Get the category batches and cast the also to double branches to put in
   // the data map
   for (auto const &item : data.getCategoryBatches(0, nEvents)) {

      std::span<const RooAbsCategory::value_type> intSpan{item.second};

      buffers.emplace();
      auto &buffer = buffers.top();
      buffer.reserve(nNonZeroWeight);

      for (std::size_t i = 0; i < nEvents; ++i) {
         if (!hasZeroWeight[i]) {
            buffer.push_back(static_cast<double>(intSpan[i]));
         }
      }
      insert(item.first->GetName(), {buffer.data(), buffer.size()});
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
            if (observable) {
               observable->inRange({retrieve(observable->GetName()).data(), nEvents}, range, isInSubRange);
            }
         }
         for (std::size_t i = 0; i < isInSubRange.size(); ++i) {
            isInRange[i] = isInRange[i] || isInSubRange[i];
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
         assignSpan(dataSpans[item.first], {buffer, nEvents});
      }
   }

   return dataSpans;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Extract all content from a RooFit datasets as a map of spans.
/// Spans with the weights and squared weights will be also stored in the map,
/// keyed with the names `_weight` and the `_weight_sumW2`. If the dataset is
/// unweighted, these weight spans will only contain the single value `1.0`.
/// Entries with zero weight will be skipped.
///
/// \return A `std::map` with spans keyed to name pointers.
/// \param[in] data The input dataset.
/// \param[in] rangeName Select only entries from the data in a given range
///            (empty string for no range).
/// \param[in] simPdf A simultaneous pdf to use as a guide for splitting the
///            dataset. The spans from each channel data will be prefixed with
///            the channel name.
/// \param[in] skipZeroWeights Skip entries with zero weight when filling the
///            data spans. Be very careful with enabling it, because the user
///            might not expect that the batch results are not aligned with the
///            original dataset anymore!
/// \param[in] takeGlobalObservablesFromData Take also the global observables
///            stored in the dataset.
/// \param[in] buffers Pass here an empty stack of `double` vectors, which will
///            be used as memory for the data if the memory in the dataset
///            object can't be used directly (e.g. because you used the range
///            selection or the splitting by categories).
std::map<RooFit::Detail::DataKey, std::span<const double>>
RooFit::BatchModeDataHelpers::getDataSpans(RooAbsData const &data, std::string const &rangeName,
                                           RooSimultaneous const *simPdf, bool skipZeroWeights,
                                           bool takeGlobalObservablesFromData, std::stack<std::vector<double>> &buffers)
{
   std::vector<std::pair<std::string, RooAbsData const *>> datasets;
   std::vector<bool> isBinnedL;
   bool splitRange = false;
   std::vector<std::unique_ptr<RooAbsData>> splitDataSets;

   if (simPdf) {
      std::unique_ptr<TList> splits{data.split(*simPdf, true)};
      for (auto *d : static_range_cast<RooAbsData *>(*splits)) {
         RooAbsPdf *simComponent = simPdf->getPdf(d->GetName());
         // If there is no PDF for that component, we also don't need to fill the data
         if (!simComponent) {
            continue;
         }
         datasets.emplace_back(std::string("_") + d->GetName() + "_", d);
         isBinnedL.emplace_back(simComponent->getAttribute("BinnedLikelihoodActive"));
         // The dataset need to be kept alive because the datamap points to their content
         splitDataSets.emplace_back(d);
      }
      splitRange = simPdf->getAttribute("SplitRange");
   } else {
      datasets.emplace_back("", &data);
      isBinnedL.emplace_back(false);
   }

   std::map<RooFit::Detail::DataKey, std::span<const double>> dataSpans; // output variable

   for (std::size_t iData = 0; iData < datasets.size(); ++iData) {
      auto const &toAdd = datasets[iData];
      auto spans = getSingleDataSpans(
         *toAdd.second, RooHelpers::getRangeNameForSimComponent(rangeName, splitRange, toAdd.second->GetName()),
         toAdd.first, buffers, skipZeroWeights && !isBinnedL[iData]);
      for (auto const &item : spans) {
         dataSpans.insert(item);
      }
   }

   if (takeGlobalObservablesFromData && data.getGlobalObservables()) {
      buffers.emplace();
      auto &buffer = buffers.top();
      buffer.reserve(data.getGlobalObservables()->size());
      for (auto *arg : static_range_cast<RooRealVar const *>(*data.getGlobalObservables())) {
         buffer.push_back(arg->getVal());
         assignSpan(dataSpans[arg], {&buffer.back(), 1});
      }
   }

   return dataSpans;
}

////////////////////////////////////////////////////////////////////////////////
/// Figure out the output size for each node in the computation graph that
/// leads up to the top node, given some vector data as an input. The input
/// data spans are in general not of the same size, for example in the case of
/// a simultaneous fit.
///
/// \return A `std::map` with output sizes for each node in the computation graph.
/// \param[in] topNode The top node of the computation graph.
/// \param[in] inputSizeFunc A function to get the input sizes.
std::map<RooFit::Detail::DataKey, std::size_t> RooFit::BatchModeDataHelpers::determineOutputSizes(
   RooAbsArg const &topNode, std::function<int(RooFit::Detail::DataKey)> const &inputSizeFunc)
{
   std::map<RooFit::Detail::DataKey, std::size_t> output;

   RooArgSet serverSet;
   RooHelpers::getSortedComputationGraph(topNode, serverSet);

   for (RooAbsArg *arg : serverSet) {
      int inputSize = inputSizeFunc(arg);
      // The size == -1 encodes that the input doesn't come from an array
      // input.
      if (inputSize != -1) {
         output[arg] = inputSize;
      }
   }

   for (RooAbsArg *arg : serverSet) {
      std::size_t size = 1;
      if (output.find(arg) != output.end()) {
         continue;
      }
      if (!arg->isReducerNode()) {
         for (RooAbsArg *server : arg->servers()) {
            if (server->isValueServer(*arg)) {
               std::size_t inputSize = output.at(server);
               if (inputSize != 1) {
                  // If the input if from an external array, the output will
                  // adopt its size and we can stop the checking of other
                  // servers.
                  size = inputSize;
                  break;
               }
            }
         }
      }
      output[arg] = size;
   }

   return output;
}

/// \endcond
