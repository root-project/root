/*
 * Project: RooFit
 * Authors:
 *   Stephan Hageboeck, CERN 2021
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooAbsDataHelper.h>

#include <RooMsgService.h>
#include <RooDataSet.h>
#include <RooDataHist.h>

#include <TROOT.h>

#include <stdexcept>

namespace RooFit {
namespace Detail {

RooAbsDataFiller::RooAbsDataFiller()
{
   const auto nSlots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
   _events.resize(nSlots);
}

void RooAbsDataFiller::Initialize()
{
   RooAbsData &absData = GetAbsData();

   _eventSize = absData.get()->size();
   _isWeighted = absData.isWeighted();
   _isDataHist = std::string{absData.ClassName()} != "RooDataSet";
}

/// Append all `events` to the internal RooDataSet or increment the bins of a RooDataHist at the given locations.
///
/// \param events Events to fill into `data`. The layout is assumed to be `(x, y, z, ...) (x, y, z, ...), (...)`.
/// \note The order of the variables inside `events` must be consistent with the order given in the constructor.
/// No matching by name is performed.
/// \param eventSize Size of a single event.
void RooAbsDataFiller::FillAbsData(const std::vector<double> &events, unsigned int eventSize)
{
   if (events.empty())
      return;

   RooAbsData &absData = GetAbsData();
   const RooArgSet &argSet = *absData.get();

   // Relevant for weighted RooDataSet
   RooRealVar *weightVar = !_isDataHist && _isWeighted ? static_cast<RooDataSet &>(absData).weightVar() : nullptr;

   for (std::size_t i = 0; i < events.size(); i += eventSize) {

      // The RooDataHist has no dedicated RooRealVar for the weight. So we just
      // use a double.
      double weightVal = 1.0;

      // Creating a RooDataSet from an RDataFrame should be consistent with the
      // creation from a TTree. The construction from a TTree discards entries
      // outside the variable definition range, so we have to do that too (see
      // also RooTreeDataStore::loadValues).

      bool allOK = true;
      for (std::size_t j = 0; j < eventSize; ++j) {
         RooAbsRealLValue *destArg = nullptr;
         if (j < argSet.size()) {
            destArg = static_cast<RooAbsRealLValue *>(argSet[j]);
         } else {
            destArg = weightVar;
         }
         double sourceVal = events[i + j];

         if (destArg && !destArg->inRange(sourceVal, nullptr)) {
            _numInvalid++;
            allOK = false;
            const auto prefix = std::string(absData.ClassName()) + "Helper::FillAbsData(" + absData.GetName() + ") ";
            if (_numInvalid < 5) {
               // Unlike in the TreeVectorStore case, we don't log the event
               // number here because we don't know it anyway, because of
               // RDataFrame slots and multithreading.
               oocoutI(nullptr, DataHandling) << prefix << "Skipping event because " << destArg->GetName()
                                              << " cannot accommodate the value " << sourceVal << "\n";
            } else if (_numInvalid == 5) {
               oocoutI(nullptr, DataHandling) << prefix << "Skipping ...\n";
            }
            break;
         }
         if (destArg) {
            destArg->setVal(sourceVal);
         } else {
            weightVal = sourceVal;
         }
      }
      if (allOK) {
         absData.add(argSet, weightVar ? weightVar->getVal() : weightVal);
      }
   }
}

/// Empty all buffers into the dataset/hist to finish processing.
void RooAbsDataFiller::Finalize()
{
   RooAbsData &absData = GetAbsData();

   for (auto &vector : _events) {
      FillAbsData(vector, _nValues);
      vector.clear();
   }

   if (_numInvalid > 0) {
      const auto prefix = std::string(absData.ClassName()) + "Helper::Finalize(" + absData.GetName() + ") ";
      oocoutW(nullptr, DataHandling) << prefix << "Ignored " << _numInvalid << " out-of-range events\n";
   }
}

void RooAbsDataFiller::ExecImpl(std::size_t nValues, std::vector<double> &vector)
{
   if (nValues != _eventSize && !(_isWeighted && nValues == _eventSize + 1)) {
      throw std::invalid_argument(
         std::string("RooAbsData can hold ") + std::to_string(_eventSize) +
         " variables per event (plus an optional weight in case of weighted data), but RDataFrame passed " +
         std::to_string(nValues) + " columns.");
   }

   _nValues = nValues;

   if (vector.size() > 1024 && _mutexDataset.try_lock()) {
      const std::lock_guard<std::mutex> guard(_mutexDataset, std::adopt_lock_t());
      FillAbsData(vector, _nValues);
      vector.clear();
   }
}

} // namespace Detail
} // namespace RooFit
