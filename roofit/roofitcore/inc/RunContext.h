// Author: Stephan Hageboeck, CERN  Jul 2020

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2020, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOFIT_ROOFITCORE_INC_BATCHRUNCONTEXT_H_
#define ROOFIT_ROOFITCORE_INC_BATCHRUNCONTEXT_H_

#include "RooSpan.h"

#include <unordered_map>
#include <vector>

class RooArgSet;
class RooAbsReal;
class RooArgProxy;

namespace BatchHelpers {

/// Data that has to be passed around when evaluating functions / PDFs.
struct RunContext {
  /// Check if there is a span of data corresponding to the variable in this proxy.
  RooSpan<const double> getBatch(const RooArgProxy& proxy) const;
  RooSpan<double> makeBatch(const RooAbsReal* owner, std::size_t size);
  /// Clear computes values. Memory owned will stay intact.
  void clear() { spans.clear(); rangeName = nullptr; }

  /// Once an object has computed its value(s), the span pointing to the results is registered here.
  std::unordered_map<const RooAbsReal*, RooSpan<const double>> spans;
  /// This is the object that owns the memory where those data are stored.
  std::unordered_map<const RooAbsReal*, std::vector<double>> ownedMemory;
  const char* rangeName{nullptr}; /// If evaluation should only occur in a range, the range name can be passed here.
};

}

#endif
