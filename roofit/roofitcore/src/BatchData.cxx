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

RooSpan<const double> BatchData::getBatch(std::size_t begin, std::size_t size) const {
  if (_foreignData) {
    if (begin >= _foreignData->size())
      return {};

    const double* dataBegin = &*(_foreignData->begin()+begin);
    const std::size_t maxSize = std::min(size, _foreignData->size() - begin);
    return RooSpan<const double>(dataBegin, maxSize);
  }

  const auto item = _ownedBatches.find(begin);
  if (item == _ownedBatches.end()) {
    // If requesting a batch inside another, a slower search algorithm must be used
    return findSpanInsideExistingBatch(begin, size);
  }

  const auto& batch = item->second;
  const std::size_t maxSize = std::min(size, batch.data.size() - (begin-batch.begin));

  return RooSpan<const double>(batch.data.data(), maxSize);
}


////////////////////////////////////////////////////////////////////////////////
/// Make a batch and return a span pointing to the pdf-local memory.
/// The batch status is switched to `kWriting`, but the batch is not initialised.
/// If a batch at this start point exists, the storage will be resized to fit the required
/// size.
///
/// \param[in] begin Begin of the batch.
/// \param[in] batchSize  Size of the batch.
/// \return An uninitialised RooSpan starting at event `begin`.

RooSpan<double> BatchData::makeWritableBatchUnInit(std::size_t begin, std::size_t batchSize) {
  auto item = _ownedBatches.find(begin);
  if (item == _ownedBatches.end()) {
    auto inserted = _ownedBatches.insert(std::make_pair(begin, Batch{begin, std::vector<double>(batchSize), kWriting}));
    return RooSpan<double>(inserted.first->second.data);
  }

  Batch& batch = item->second;
  batch.status = kWriting;
  if (batch.data.size() != batchSize) {
    batch.data.resize(batchSize);
  }

  return RooSpan<double>(batch.data);
}


////////////////////////////////////////////////////////////////////////////////
/// Make a batch and return a span pointing to the pdf-local memory.
/// Calls makeWritableBatchUnInit() and initialises the memory.
///
/// \param[in] begin Begin of the batch.
/// \param[in] batchSize   End of the batch (not included)
/// \param[in] value Value to initialise with (defaults to 0.).
/// \return An initialised RooSpan starting at event `begin`.
RooSpan<double> BatchData::makeWritableBatchInit(std::size_t begin, std::size_t batchSize, double value) {
  auto batch = makeWritableBatchUnInit(begin, batchSize);
  for (auto& elm : batch) {
    elm = value;
  }

  return batch;
}

////////////////////////////////////////////////////////////////////////////////
/// Attach a foreign storage. Batches coming from this storage will be read only.
void BatchData::attachForeignStorage(const std::vector<double>& vec) {
  clear();

  _foreignData = &vec;
  _ownedBatches.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Print to given output stream.
void BatchData::print(std::ostream& os, const std::string& indent) const {
  os << indent << "Batch data access";
  if (_ownedBatches.empty() && !_foreignData) {
    os << " not initialised." << std::endl;
    return;
  }

  using std::setw;

  os << " with " << (_foreignData ? "(foreign)" : "(owned)") << " data:";
  os << "\n" << indent << std::right << std::setw(8) << "Batch #" << std::setw(8) << "Start"
          << std::setw(7) << "Status";

  unsigned int i=0;
  for (auto item : _ownedBatches) {
    auto startPoint = item.first;
    const Batch& batch = item.second;

    os << "\n" << indent
        << std::setw(8) << i << std::setw(8) << startPoint
        << std::setw(7) << batch.status << ": {";
    for (unsigned int j=0; j < 5 && j < batch.data.size(); ++j) {
      os << batch.data[j] << ", ";
    }
    os << "...}";
  }
  os << std::resetiosflags(std::ios::adjustfield) << std::endl;
}



}
