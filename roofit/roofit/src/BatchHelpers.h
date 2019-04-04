// Author: Stephan Hageboeck, CERN  25 Feb 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOFIT_ROOFITCORE_INC_BATCHHELPERS_H_
#define ROOFIT_ROOFITCORE_INC_BATCHHELPERS_H_

#include "RooRealProxy.h"

#include <vector>
#include <RooSpan.h>

class RooArgSet;

#define USE_VDT

namespace BatchHelpers {


///\class LookupBatchData
/// Helper class to find RooRealProxy data in a vector of RooSpans.
/// A lookup instance is constructed with the data batches, a RooArgSet of the input
/// variables that the data batch contains, and a std::vector of proxy references.
/// The proxies will be searched for in the batch data, and later the lookup object can
/// be queried whether a proxy is in the batch data.
/// The class further provides a helper to test if batches overlap, a condition that
/// prohibits SIMD computations. Therefore, it should be asserted that the OUTPUT spans
/// do not overlap with the input data using testOverlap().
class LookupBatchData {
  public:
    ///Look up the index of all variables given in the proxies in the `inputVars`.
    ///\param[in] dataBatches Batch data.
    ///\param[in] inputVars Set of all input variables associated with the `dataBatches`
    /// that are eligible for batch computations.
    ///\param[in] proxies List of RooRealProxies to be checked. The proxies will be
    /// queried for the name of the object they represent, and this name is looked up in the `inputVars`.
    LookupBatchData(const std::vector<RooSpan<const double>>& dataBatches,
        const RooArgSet& inputVars,
        std::vector<std::reference_wrapper<const RooRealProxy>> proxies);

    ///Look up the index of all variables given in the `queryVars` in the `inputVars`.
    ///\param[in] dataBatches Batch data.
    ///\param[in] inputVars Set of all input variables associated with the `dataBatches`
    /// that are eligible for batch computations.
    ///\param[in] queryVars List of variables to be checked. The variables will be
    /// looked up by name in the `inputVars`.
    LookupBatchData(const std::vector<RooSpan<const double>>& dataBatches,
        const RooArgSet& inputVars,
        const RooArgSet& queryVars);

    ///Check if the variable associated with this proxy is in the batch data.
    bool isBatch(const RooRealProxy& proxy) const {
      auto result = _lookupMap.find(&proxy.arg());
      return result != _lookupMap.end() && result->second != -1;
    }

    ///Get batch data associated with this proxy.
    RooSpan<const double> data(const RooRealProxy& proxy) const {
      return data(proxy.arg());
    }

    ///Get batch data associated with this RooAbsReal.
    RooSpan<const double> data(const RooAbsReal& real) const {
      auto result = _lookupMap.find(&real);
      assert(result != _lookupMap.end() && 0 <= result->second && result->second < (int)_batchData.size());
      return _batchData[result->second];
    }

    ///Test if span overlaps with input data.
    ///\param[in] testee A span that should be checked for possible overlap with the input data.
    ///\return True in case of overlap.
    template <class Span_t>
    bool testOverlap(const Span_t& testee) const {
      auto theTest = [this,testee](typename decltype(_lookupMap)::value_type it) {
        if (it.second == -1)
          return false;

        const auto& theBatch = _batchData[it.second];
        return testee.overlaps(theBatch);
      };
      return std::any_of(_lookupMap.begin(), _lookupMap.end(), theTest);
    }

  private:
    std::map<const RooAbsReal*, int> _lookupMap;
    const std::vector<RooSpan<const double>>& _batchData;
};



///Little adapter that gives a bracket operator to types that don't
///have one. This is helpful for template programming, where one might
///encounter variables that sometimes have one changing value have constants that don't change inside a loop
template <class T>
class BracketAdapter {
  public:
    BracketAdapter(T& payload) :
    _payload{payload} { }

    double operator[](std::size_t i) const {
      return _payload[i];
    }

  private:
    T& _payload;
};

template <>
class BracketAdapter<double> {
  public:
    constexpr BracketAdapter(double payload) :
    _payload{payload} { }

    constexpr double operator[](std::size_t) const {
      return _payload;
    }

  private:
    const double _payload;
};

template <>
class BracketAdapter<RooRealProxy> {
  public:
    BracketAdapter(const RooRealProxy& payload) :
    _payload{payload} { }

    constexpr double operator[](std::size_t) const {
      return _payload;
    }

  private:
    const double _payload;
};

}

#endif /* ROOFIT_ROOFITCORE_INC_BATCHHELPERS_H_ */
