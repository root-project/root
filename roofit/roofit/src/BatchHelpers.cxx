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

#include "BatchHelpers.h"

#include "RooArgSet.h"

namespace BatchHelpers {

LookupBatchData::LookupBatchData(const std::vector<RooSpan<const double>>& dataBatches,
    const RooArgSet& inputVars,
    std::vector<std::reference_wrapper<const RooRealProxy>> proxies) :
  _lookupMap{ },
  _batchData{dataBatches} {

  for (auto proxy : proxies) {
    _lookupMap[&(proxy.get().arg())] = inputVars.index(proxy.get().arg().GetName());
  }
}

LookupBatchData::LookupBatchData(const std::vector<RooSpan<const double>>& dataBatches,
    const RooArgSet& inputVars,
    const RooArgSet& queryVars) :
  _lookupMap{ },
  _batchData{dataBatches} {

  for (const auto arg : queryVars) {
    assert(dynamic_cast<const RooAbsReal*>(arg));
    auto var = static_cast<const RooAbsReal*>(arg);
    _lookupMap[var] = inputVars.index(var);
  }
}

}
