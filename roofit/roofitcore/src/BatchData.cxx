// Author: Stephan Hageboeck, CERN  12 Apr 2019

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

#include "BatchData.h"

#include <ostream>
#include <iomanip>
#include <iostream>

namespace BatchHelpers {

void BatchData::resizeAndClear(std::size_t size, std::size_t batchSize) {
  if (_foreignData) {
    assert(_foreignData->size() == size);
    _batchStartPoints = {0};
    return;
  }

  _ownedData.resize(size);
  _batchSize = batchSize;
  const size_t nBatch = size / batchSize + (std::size_t)(size % batchSize > 0);

  _batchStatus.assign(nBatch, kDirtyOrUnknown);

  _batchStartPoints.clear();
  for (std::size_t i=0; i < nBatch; ++i) {
    _batchStartPoints.push_back(batchSize*i);
  }
}

RooSpan<const double> BatchData::makeBatch(std::size_t begin, std::size_t end) const {
#ifndef NDEBUG
  assert(isInit());
  assert(validRange(begin, end));

  const auto batchIndex = calcBatchIndex(begin);

  if (_batchStartPoints[batchIndex] != begin) {
    std::cerr << __FILE__ << ":" << __LINE__
        << " Starting to read inside batch #" << batchIndex
        << " from " << begin << " to " << end
        << " although batch starts at " << _batchStartPoints[batchIndex] << std::endl;
  }
  assert(calcBatchIndex(end-1) == batchIndex);
  assert(_batchStatus[batchIndex] >= kReady);
#endif

  return RooSpan<const double>(&data()[begin], end-begin);
}


////////////////////////////////////////////////////////////////////////////////
/// Make a batch and return a span pointing to the pdf-local memory.
/// The batch status is switched to `kWriting`, but the batch is not initialised.
///
/// \param[in] begin Begin of the batch.
/// \param[in] end   End of the batch (not included)
/// \return An uninitialised RooSpan starting at event `begin`.

RooSpan<double> BatchData::makeWritableBatchUnInit(std::size_t begin, std::size_t end) {
  assert(isInit());
  assert(validRange(begin, end));
  assert(!_ownedData.empty());

  const auto batchIndex = calcBatchIndex(begin);
#ifndef NDEBUG
  if (_batchStartPoints[batchIndex] != begin) {
    std::cerr << __FILE__ << ":" << __LINE__
        << " Starting to write inside batch #" << batchIndex
        << " from " << begin << " to " << end
        << " although batch starts at " << _batchStartPoints[batchIndex] << std::endl;
  }
  assert(calcBatchIndex(end-1) == batchIndex);
  assert(_batchStatus[batchIndex] != kReadyAndConstant);
#endif

  _batchStatus[batchIndex] = kWriting;
  return RooSpan<double>(&_ownedData[begin], end-begin);
}


////////////////////////////////////////////////////////////////////////////////
/// Make a batch and return a span pointing to the pdf-local memory.
/// Calls makeWritableBatchUnInit() and initialises the memory.
///
/// \param[in] begin Begin of the batch.
/// \param[in] end   End of the batch (not included)
/// \param[in] value Value to initialise with (defaults to 0.).
/// \return An initialised RooSpan starting at event `begin`.
RooSpan<double> BatchData::makeWritableBatchInit(std::size_t begin, std::size_t end, double value) {
  auto batch = makeWritableBatchUnInit(begin, end);
  for (auto& elm : batch) {
    elm = value;
  }

  return batch;
}

void BatchData::attachForeignStorage(const std::vector<double>& vec) {
  reset();

  _foreignData = &vec;
  _ownedData.clear();
  _batchStatus = {kReadyAndConstant};
  _batchStartPoints = {0};
}

void BatchData::print(std::ostream& os, const std::string& indent) const {
  os << indent << "Batch data access";
  if (!isInit()) {
    os << " not initialised." << std::endl;
    return;
  }

  using std::setw;

  os << " with " << data().size() << " items "
      << (_foreignData ? "(foreign)" : "(owned)") << ":";
  os << "\n" << indent << std::right << std::setw(8) << "Batch #" << std::setw(8) << "Start"
      << std::setw(7) << "Status";
  for (unsigned int i=0; i < _batchStatus.size(); ++i) {
    auto startPoint = _batchStartPoints[i];
    os << "\n" << indent
        << std::setw(8) << i << std::setw(8) << startPoint
        << std::setw(7) << _batchStatus[i] << ": {";
    for (unsigned int j=0; j < 5 && startPoint + j < data().size(); ++j) {
        os << data()[startPoint+j] << ", ";
    }
    os << "...}";
  }
  os << std::resetiosflags(std::ios::adjustfield) << std::endl;
}


std::size_t BatchData::calcBatchIndex(std::size_t begin) const {
  assert(isInit());
  if (_foreignData)
    return 0;

  return begin / _batchSize;
}


void BatchData::reset() {
  _ownedData.clear();
  _foreignData = nullptr;
  _batchStatus.clear();
  _batchStartPoints.clear();
  _batchSize = 0;
}

}
