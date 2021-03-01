/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
/// Create RooDataSet/RooDataHist from RDataFrame.
/// \date Mar 2021
/// \author Stephan Hageboeck (CERN)
#ifndef ROOABSDATAHELPER
#define ROOABSDATAHELPER

#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooDataHist.h>

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/ActionHelpers.hxx>
#include <TROOT.h>

#include <vector>
#include <mutex>
#include <memory>
#include <cstddef>
#include <string>
#include <stdexcept>

class TTreeReader;

/// This is a helper for an RDataFrame action, which fills RooFit data classes.
///
/// \tparam DataSet_t Either RooDataSet or RooDataHist.
///
/// To construct RooDataSet / RooDataHist within RDataFrame
/// - Construct one of the two action helpers RooDataSetHelper or RooDataHistHelper. Pass constructor arguments
///   to RooAbsDataHelper::RooAbsDataHelper() as for the original classes.
///   The arguments are forwarded to the actual data classes without any changes.
/// - Book the helper as an RDataFrame action. Here, the RDataFrame column types have to be passed as template parameters.
/// - Pass the column names to the Book action. These are matched by position to the variables of the dataset.
///
/// All arguments passed to  are forwarded to RooDataSet::RooDataSet() / RooDataHist::RooDataHist().
///
/// #### Usage example:
/// ```
///    RooRealVar x("x", "x", -5.,   5.);
///    RooRealVar y("y", "y", -50., 50.);
///    auto myDataSet = rdataframe.Book<double, double>(
///      RooDataSetHelper{"dataset",          // Name   (directly forwarded to RooDataSet::RooDataSet())
///                      "Title of dataset",  // Title  (                   ~ " ~                      )
///                      RooArgSet(x, y) },   // Variables to create in dataset
///      {"x", "y"}                           // Column names from RDataFrame
///    );
///
/// ```
/// \warning Variables in the dataset and columns in RDataFrame are **matched by position, not by name**.
/// This enables the easy exchanging of columns that should be filled into the dataset.
template<class DataSet_t>
class RooAbsDataHelper : public ROOT::Detail::RDF::RActionImpl<RooAbsDataHelper<DataSet_t>> {
public:
  using Result_t = DataSet_t;

private:
  std::shared_ptr<DataSet_t> _dataset;
  std::mutex _mutex_dataset;

  std::vector<std::vector<double>> _events; // One vector of values per data-processing slot
  const std::size_t _eventSize; // Number of variables in dataset

public:

  /// Construct a helper to create RooDataSet/RooDataHist.
  /// \tparam Args_t Parameter pack of arguments.
  /// \param args Constructor arguments for RooDataSet::RooDataSet() or RooDataHist::RooDataHist().
  /// All arguments will be forwarded as they are.
  template<typename... Args_t>
  RooAbsDataHelper(Args_t&&... args) :
  _dataset{ new DataSet_t(std::forward<Args_t>(args)...) },
  _eventSize{ _dataset->get()->size() }
  {
    const auto nSlots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
    _events.resize(nSlots);
  }


  /// Move constructor. It transfers ownership of the internal RooAbsData object.
  RooAbsDataHelper(RooAbsDataHelper&& other) :
  _dataset{ std::move(other._dataset) },
  _mutex_dataset(),
  _events{ std::move(other._events) },
  _eventSize{ other._eventSize }
  {

  }

  /// Copy is discouraged.
  /// Use `rdataframe.Book<...>(std::move(absDataHelper), ...)` instead.
  RooAbsDataHelper(const RooAbsDataHelper&) = delete;
  /// Return internal dataset/hist.
  std::shared_ptr<DataSet_t> GetResultPtr() const { return _dataset; }
  /// RDataFrame interface method. Nothing has to be initialised.
  void Initialize() {}
  /// RDataFrame interface method. No tasks.
  void InitTask(TTreeReader *, unsigned int) {}
  /// RDataFrame interface method.
  std::string GetActionName() { return "RooDataSetHelper"; }

  /// Method that RDataFrame calls to pass a new event.
  ///
  /// \param slot When IMT is used, this is a number in the range [0, nSlots) to fill lock free.
  /// \param values x, y, z, ... coordinates of the event.
  template <typename... ColumnTypes>
  void Exec(unsigned int slot, ColumnTypes... values)
  {
    if (sizeof...(values) != _eventSize) {
      throw std::invalid_argument(std::string("RooDataSet can hold ")
      + std::to_string(_eventSize)
      + " variables per event, but RDataFrame passed "
      + std::to_string(sizeof...(values))
      + " columns.");
    }

    auto& vector = _events[slot];
    for (auto&& val : {values...}) {
      vector.push_back(val);
    }

    if (vector.size() > 1024 && _mutex_dataset.try_lock()) {
      const std::lock_guard<std::mutex> guard(_mutex_dataset, std::adopt_lock_t());
      FillDataSet(vector, _eventSize);
      vector.clear();
    }
  }

  /// Empty all buffers into the dataset/hist to finish processing.
  void Finalize() {
    for (auto& vector : _events) {
      FillDataSet(vector, _eventSize);
      vector.clear();
    }
  }


private:
  /// Append all `events` to the internal RooDataSet or increment the bins of a RooDataHist at the given locations.
  ///
  /// \param events Events to fill into `data`. The layout is assumed to be `(x, y, z, ...) (x, y, z, ...), (...)`.
  /// \note The order of the variables inside `events` must be consistent with the order given in the constructor.
  /// No matching by name is performed.
  /// \param eventSize Size of a single event.
  void FillDataSet(const std::vector<double>& events, unsigned int eventSize) {
    if (events.size() == 0)
      return;

    const RooArgSet& argSet = *_dataset->get();

    for (std::size_t i = 0; i < events.size(); i += eventSize) {
      for (std::size_t j=0; j < eventSize; ++j) {
        static_cast<RooAbsRealLValue*>(argSet[j])->setVal(events[i+j]);
      }
      _dataset->add(argSet);
    }
  }
};

/// Helper for creating a RooDataSet inside RDataFrame. \see RooAbsDataHelper
using RooDataSetHelper = RooAbsDataHelper<RooDataSet>;
/// Helper for creating a RooDataHist inside RDataFrame. \see RooAbsDataHelper
using RooDataHistHelper = RooAbsDataHelper<RooDataHist>;

#endif
