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

#include "RunContext.h"

#include "RooArgProxy.h"
#include "RooAbsReal.h"

namespace BatchHelpers {

RooSpan<const double> RunContext::getBatch(const RooArgProxy& proxy) const {
  auto item = spans.find(static_cast<const RooAbsReal*>(proxy.absArg()));
  if (item != spans.end())
    return item->second;

  return {};
}

/// Create a writable batch. If the RunContext already owns memory for the object
/// `owner`, just resize the memory. If it doesn't exist yet, allocate it.
/// \param owner RooFit object whose value should be written into the memory.
/// \param size Requested size of the span.
/// \return A writeable RooSpan of the requested size, whose memory is owned by
/// the RunContext.
RooSpan<double> RunContext::makeBatch(const RooAbsReal* owner, std::size_t size) {
  auto item = ownedMemory.find(owner);
  if (item == ownedMemory.end() || item->second.size() != size) {
    std::vector<double>& data = ownedMemory[owner];
    data.resize(size);
    spans[owner] = RooSpan<const double>(data);
    return {data};
  }

  return {item->second};
}

}
