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
  
constexpr size_t block = 1024;

struct ArrayWrapper {
    const double * __restrict__ ptr;
    bool _batch;
    
    constexpr double operator[](std::size_t i) const {
      return ptr[i];
    }
    constexpr bool batch() const {
      return _batch;
    }
};

struct EvaluateInfo {
  size_t size, nBatches;
};
  
size_t findSize(std::vector< RooSpan<const double> > parameters);
EvaluateInfo getInfo(std::vector<const RooRealProxy*> parameters, size_t begin, size_t batchSize);
EvaluateInfo init(std::vector< RooRealProxy > parameters, 
                  std::vector<  ArrayWrapper* > wrappers,
                  std::vector< double*> arrays,
                  size_t begin, size_t batchSize );

///Little adapter that gives a bracket operator to types that don't
///have one. It completely ignores the index and returns a constant.
template <class T = double>
class BracketAdapter {
  public:

    constexpr BracketAdapter(T payload) noexcept :
    _payload{payload} { }

    constexpr double operator[](std::size_t) const {
      return _payload;
    }

  private:
    const T _payload;
};


class BracketAdapterWithBranch {
  public:
    explicit BracketAdapterWithBranch(double payload, const RooSpan<const double>& batch) noexcept :
    _payload(payload),
    _pointer(batch.data()),
    _batchEmpty(batch.empty())
    {
    }

    inline double operator[](std::size_t i) const noexcept {
      return _batchEmpty ? _payload : _pointer[i];
    }

  private:
    const double _payload;
    const double* __restrict__ const _pointer;
    const bool _batchEmpty;
};

}

#endif /* ROOFIT_ROOFITCORE_INC_BATCHHELPERS_H_ */
