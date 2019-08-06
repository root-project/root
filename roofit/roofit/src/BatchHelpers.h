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

namespace BatchHelpers {

///Little adapter that gives a bracket operator to types that don't
///have one. This is helpful for template programming, where one might
///encounter variables that sometimes have one changing value have constants that don't change inside a loop
template <class T>
class BracketAdapter {
  public:
    constexpr BracketAdapter(T& payload) :
    _payload{payload} { }

    constexpr double operator[](std::size_t i) const {
      return _payload[i];
    }

  private:
    const T& _payload;
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
    constexpr BracketAdapter(const RooRealProxy& payload) :
    _payload{payload} { }

    constexpr double operator[](std::size_t) const {
      return _payload;
    }

  private:
    const double _payload;
};


class BracketAdapterWithBranch {
  public:
    constexpr explicit BracketAdapterWithBranch(double payload, const RooSpan<const double>& batch) :
    _payload(payload),
    _span(batch),
    _batchEmpty(batch.empty())
    {
    }

    constexpr double operator[](std::size_t i) const {
      return _batchEmpty ? _payload : _span[i];
    }

  private:
    const double _payload;
    const RooSpan<const double>& _span;
    const bool _batchEmpty;
};

}

#endif /* ROOFIT_ROOFITCORE_INC_BATCHHELPERS_H_ */
