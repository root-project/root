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

#ifndef RooFit_RooFitCore_RooAbsDataFiller_h
#define RooFit_RooFitCore_RooAbsDataFiller_h

#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooRealVar.h>

#include <vector>
#include <mutex>
#include <cstddef>
#include <string>

class TTreeReader;

namespace RooFit {
namespace Detail {

class RooAbsDataFiller {
public:
   RooAbsDataFiller();

   /// Move constructor. It transfers ownership of the internal RooAbsData object.
   RooAbsDataFiller(RooAbsDataFiller &&other) : _events{std::move(other._events)}, _eventSize{other._eventSize} {}

   /// Copy is discouraged.
   /// Use `rdataframe.Book<...>(std::move(absDataHelper), ...)` instead.
   RooAbsDataFiller(const RooAbsDataFiller &) = delete;
   /// RDataFrame interface method.
   void Initialize();
   /// RDataFrame interface method. No tasks.
   void InitTask(TTreeReader *, unsigned int) {}
   /// RDataFrame interface method.
   std::string GetActionName() { return "RooDataSetHelper"; }

   void ExecImpl(std::size_t nValues, std::vector<double>& vector);
   void Finalize();

   virtual RooAbsData &GetAbsData() = 0;

protected:
   void FillAbsData(const std::vector<double> &events, unsigned int eventSize);

   std::mutex _mutex_dataset;
   std::size_t _numInvalid = 0;

   std::vector<std::vector<double>> _events; // One vector of values per data-processing slot
   std::size_t _eventSize;                   // Number of variables in dataset
};

} // namespace Detail
} // namespace RooFit


#endif
