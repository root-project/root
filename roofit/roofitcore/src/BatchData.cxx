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

namespace BatchHelpers {

void BatchData::allocateAndBatch(std::size_t size, std::size_t batchSize) {
  if (_foreignData) {
    assert(_foreignData->size() == size);
    _batchStartPoints = {0};
    return;
  }

  _ownedData.resize(size);
  _batchSize = batchSize;
  size_t nBatch = size / batchSize;
  if (size % batchSize > 0)
    ++nBatch;

  _batchStatus.assign(nBatch, kDirtyOrUnknown);

  _batchStartPoints.clear();
  for (unsigned int i=0; i < nBatch; ++i) {
    _batchStartPoints.push_back(batchSize*i);
  }
}

RooSpan<const double> BatchData::makeBatch(std::size_t begin, std::size_t end) const {
  assert(isInit());
  assert(validRange(begin, end));

  const auto batchIndex = calcBatchIndex(begin, end);

  assert(_batchStartPoints[batchIndex] == begin);
  assert(_batchStatus[batchIndex] >= kReady);

  return RooSpan<const double>(&data()[begin], end-begin);
}


RooSpan<double> BatchData::makeWritableBatch(std::size_t begin, std::size_t end) {
  assert(isInit());
  assert(validRange(begin, end));
  assert(!_ownedData.empty());

  const auto batchIndex = calcBatchIndex(begin, end);
  assert(_batchStartPoints[batchIndex] == begin);
  assert(_batchStatus[batchIndex] != kReadyAndConstant);

  _batchStatus[batchIndex] = kWriting;

  return RooSpan<double>(&_ownedData[begin], end-begin);
}

void BatchData::attachForeignStorage(const std::vector<double>& vec) {
  reset();

  _foreignData = &vec;
  _ownedData.clear();
  _batchStatus = {kReadyAndConstant};
  _batchStartPoints = {0};
}

void BatchData::print(std::ostream& os, const std::string& indent) const {
  os << indent << "Batch data";
  if (!isInit()) {
    os << " not initialised." << std::endl;
    return;
  }

  os << " with " << data().size() << " items:";
  for (unsigned int i=0; i < _batchStatus.size(); ++i) {
    os << "\n" << indent << "\t" << _batchStartPoints[i] << ":\t" << _batchStatus[i];
  }
  os << std::endl;
}


std::size_t BatchData::calcBatchIndex(std::size_t begin, std::size_t end) const {
  assert(isInit());
  if (_foreignData)
    return 0;

  auto batchIndex = begin / _batchSize;
  assert(batchIndex + 1 < data().size() / _batchSize || (end - begin) == _batchSize);

  return batchIndex;
}


void BatchData::reset() {
  _ownedData.clear();
  _foreignData = nullptr;
  _batchStatus.clear();
  _batchStartPoints.clear();
  _batchSize = 0;
}

}
