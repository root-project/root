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

#ifndef ROOFIT_ROOFITCORE_RUNCONTEXT_H
#define ROOFIT_ROOFITCORE_RUNCONTEXT_H

#include "RooSpan.h"
#include "RooFit/Detail/DataMap.h"

#include <map>
#include <vector>

class RooArgSet;
class RooAbsArg;
class RooArgProxy;

namespace RooBatchCompute {

struct RunContext {
  /// Create an empty RunContext that doesn't have access to any computation results.
  RunContext() { }
  /// Deleted because copying the owned memory is expensive.
  /// If needed, it can be implemented, though.
  /// \warning Remember to relocate all spans in `spans` to new location
  /// in `ownedMemory` after data have been copied!
  RunContext(const RunContext&) = delete;
  /// Move a RunContext. All spans pointing to data retrieved from the original remain valid.
  RunContext(RunContext&&) = default;
  RooSpan<const double> getBatch(const RooArgProxy& proxy) const;
  RooSpan<const double> getBatch(const RooAbsArg* owner) const;
  /// Retrieve a batch of data corresponding to the element passed as `owner`.
  RooSpan<const double> operator[](const RooAbsArg* owner) const { return getBatch(owner); }
  RooSpan<double> getWritableBatch(const RooAbsArg* owner);
  RooSpan<double> makeBatch(const RooAbsArg* owner, std::size_t size);

  /// Clear all computation results without freeing memory.
  void clear();

  /// Once an object has computed its value(s), the span pointing to the results is registered here.
  std::map<RooFit::Detail::DataKey, RooSpan<const double>> spans;
  std::map<RooFit::Detail::DataKey, const double*> spansCuda;

  /// Memory owned by this struct. It is associated to nodes in the computation graph using their pointers.
  std::map<RooFit::Detail::DataKey, std::vector<double>> ownedMemory;
  std::map<RooFit::Detail::DataKey, double*> ownedMemoryCuda;

  const char* rangeName{nullptr};       ///< If evaluation should only occur in a range, the range name can be passed here.
  std::vector<double> logProbabilities; ///< Possibility to register log probabilities.
};

}

#endif
