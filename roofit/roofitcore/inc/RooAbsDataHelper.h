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

#ifndef RooFit_RooFitCore_RooAbsDataHelper_h
#define RooFit_RooFitCore_RooAbsDataHelper_h

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
   RooAbsDataFiller &operator=(RooAbsDataFiller &&) = delete;

   /// Copy is discouraged.
   /// Use `rdataframe.Book<...>(std::move(absDataHelper), ...)` instead.
   RooAbsDataFiller(const RooAbsDataFiller &) = delete;
   RooAbsDataFiller &operator=(const RooAbsDataFiller &) = delete;

   ~RooAbsDataFiller() = default;

   /// RDataFrame interface method.
   void Initialize();
   /// RDataFrame interface method. No tasks.
   void InitTask(TTreeReader *, unsigned int) {}
   /// RDataFrame interface method.
   std::string GetActionName() { return "RooDataSetHelper"; }

   void ExecImpl(std::size_t nValues, std::vector<double> &vector);
   void Finalize();

   virtual RooAbsData &GetAbsData() = 0;

   std::vector<double> &events(std::size_t slot) { return _events[slot]; }

protected:
   void FillAbsData(const std::vector<double> &events, unsigned int eventSize);

   std::mutex _mutexDataset;
   std::size_t _numInvalid = 0;

   std::vector<std::vector<double>> _events; // One vector of values per data-processing slot
   std::size_t _eventSize;                   // Number of variables in dataset
   std::size_t _nValues;                     // Number of variables in dataframe

   bool _isWeighted = false;
   bool _isDataHist = false;
};

} // namespace Detail
} // namespace RooFit

// Forward declarations from RDF
namespace ROOT {
namespace Detail {
namespace RDF {
template <typename Helper>
class RActionImpl;
}
} // namespace Detail
} // namespace ROOT

/// This is a helper for an RDataFrame action, which fills RooFit data classes.
///
/// \tparam DataSet_t Either RooDataSet or RooDataHist.
///
/// To construct RooDataSet / RooDataHist within RDataFrame
/// - Construct one of the two action helpers RooDataSetHelper or RooDataHistHelper. Pass constructor arguments
///   to RooAbsDataHelper::RooAbsDataHelper() as for the original classes.
///   The arguments are forwarded to the actual data classes without any changes.
/// - Book the helper as an RDataFrame action. Here, the RDataFrame column types have to be passed as template
/// parameters.
/// - Pass the column names to the Book action. These are matched by position to the variables of the dataset.
///   If there is one more column name than variables in the dataset, the last columns values will be used as weights.
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
///      {"x", "y", "weight"}                 // Column names from RDataFrame
///                                           // (this example uses an additional column for the weight)
///    );
///
/// ```
/// \warning Variables in the dataset and columns in RDataFrame are **matched by position, not by name**.
/// This enables the easy exchanging of columns that should be filled into the dataset.
template <class DataSet_t>
class RooAbsDataHelper : public RooFit::Detail::RooAbsDataFiller,
                         public ROOT::Detail::RDF::RActionImpl<RooAbsDataHelper<DataSet_t>> {
public:
   using Result_t = DataSet_t;

   /// Construct a helper to create RooDataSet/RooDataHist.
   /// \tparam Args_t Parameter pack of arguments.
   /// \param args Constructor arguments for RooDataSet::RooDataSet() or RooDataHist::RooDataHist().
   /// All arguments will be forwarded as they are.
   template <typename... Args_t>
   RooAbsDataHelper(Args_t &&...args) : _dataset{new DataSet_t(std::forward<Args_t>(args)...)}
   {
   }

   /// Return internal dataset/hist.
   std::shared_ptr<DataSet_t> GetResultPtr() const { return _dataset; }

   /// Method that RDataFrame calls to pass a new event.
   ///
   /// \param slot When IMT is used, this is a number in the range [0, nSlots) to fill lock free.
   /// \param values x, y, z, ... coordinates of the event.
   template <typename... ColumnTypes>
   void Exec(unsigned int slot, ColumnTypes... values)
   {
      std::vector<double> &vector = _events[slot];
      for (auto &&val : {static_cast<double>(values)...}) {
         vector.push_back(val);
      }

      ExecImpl(sizeof...(values), vector);
   }

   DataSet_t &GetAbsData() override { return *_dataset; }

private:
   std::shared_ptr<DataSet_t> _dataset;
};

/// Helper for creating a RooDataSet inside RDataFrame. \see RooAbsDataHelper
using RooDataSetHelper = RooAbsDataHelper<RooDataSet>;
/// Helper for creating a RooDataHist inside RDataFrame. \see RooAbsDataHelper
using RooDataHistHelper = RooAbsDataHelper<RooDataHist>;

#endif
