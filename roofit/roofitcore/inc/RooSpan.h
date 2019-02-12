// Author: Stephan Hageboeck, CERN  7 Feb 2019

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

#ifndef ROOFIT_ROOFITCORE_INC_DATABATCH_H_
#define ROOFIT_ROOFITCORE_INC_DATABATCH_H_

#include "ROOT/span.hxx"

////////////////////////////////////////////////////////////////////////////
/// A simple container to hold a batch of data values.
/// It can operate in two modes:
/// * Span: It holds only references to the storage held by another object
/// like a std::span does.
/// * Temp data: It holds its own data, and exposes the span.
/// This mode is necessary to ship data that are not available in
/// a contiguous storage like e.g. data from a TTree. This means, however, that
/// data have to be copied, and follow the DataBatch.
template<class T>
class RooSpan : public std::span<T> {
public:
  /// Construct from a range. Data held by foreign object.
  RooSpan(const_iterator begin, const_iterator end) :
  std::span(begin, end),
  _tmpStorage{}
  { }

  /// Construct with data to be held by the DataBatch
  RooSpan(std::vector<double>&& payload) :
  std::span(payload.data(), payload.size()),
  _tmpStorage{payload}
  { }

private:
  /// Auxiliary storage if a class does not support exposing the data directly.
  /// In this case, a temporary storage has to be created, and shipped with the batch.
  std::vector<T> _tmpStorage;
};



#endif /* ROOFIT_ROOFITCORE_INC_DATABATCH_H_ */
