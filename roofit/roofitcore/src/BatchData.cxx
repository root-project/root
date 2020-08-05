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

namespace BatchHelpers {

///////////////////////////////////////////////////////////////////////////
/// Return the status of the batch starting at `begin`.
/// \param[in] begin Start of the batch.
/// \param[in] normSet Optional normSet pointer to distinguish differently normalised computations.
/// \param[in] ownerTag Optional owner tag. This avoids reusing batch memory for e.g. getVal() and getLogVal().
/// \return One state of the Status_t enum.
BatchData::Status_t BatchData::status(std::size_t begin, const RooArgSet* const normSet, Tag_t ownerTag) const {
  if (_foreignData)
    return begin < _foreignData->size() ? kReadyAndConstant : kNoBatch;
  else if (_ownedBatches.empty())
    return kNoBatch;

  auto item = _ownedBatches.find(std::make_tuple(begin, normSet, ownerTag));
  if (item != _ownedBatches.end()) {
    return item->second.status;
  } else if ( (item = findEnclosingBatch(begin, normSet, ownerTag)) != _ownedBatches.end() ){
    // We didn't find a batch that starts with `begin`, but `begin` might be
    // inside of an existing batch. This search is slower.
    return item->second.status;
  }

  return kNoBatch;
}


///////////////////////////////////////////////////////////////////////////
/// Set the status of a batch with the given properties.
///
/// The status of foreign read-only data will never change.
/// \param[in] begin Begin index of the batch.
/// \param[in] size  Size of the batch for checking that enough data is available.
/// \param[in] normSet Optional normSet pointer to destinguish differently normalised computations.
/// \param[in] ownerTag Optional owner tag. This avoids reusing batch memory for e.g. getVal() and getLogVal().
/// \return True if status successfully set, false if no such batch / not writable.
bool BatchData::setStatus(std::size_t begin, std::size_t size, Status_t stat,
    const RooArgSet* const normSet, Tag_t ownerTag) {
  if (_foreignData)
    return false;

  auto item = _ownedBatches.find(std::make_tuple(begin, normSet, ownerTag));
  if (item == _ownedBatches.end() || size != item->second.data.size())
    return false;

  item->second.status = stat;
  return true;
}


///////////////////////////////////////////////////////////////////////////
/// Retrieve an existing batch.
///
/// \param[in] begin Begin index of the batch.
/// \param[in] maxSize  Requested size. Batch may come out smaller than this.
/// \param[in] normSet Optional normSet pointer to distinguish differently normalised computations.
/// \param[in] ownerTag Optional owner tag. This avoids reusing batch memory for e.g. getVal() and getLogVal().
/// \return Non-mutable contiguous batch data.
RooSpan<const double> BatchData::getBatch(std::size_t begin, std::size_t maxSize,
    const RooArgSet* const normSet, Tag_t ownerTag) const {
  if (_foreignData) {
    if (begin >= _foreignData->size())
      return {};

    const double* dataBegin = &*(_foreignData->begin()+begin);
    maxSize = std::min(maxSize, _foreignData->size() - begin);
    return RooSpan<const double>(dataBegin, maxSize);
  }

  if (_ownedBatches.empty())
    return {};

  const auto item = _ownedBatches.find(std::make_tuple(begin, normSet, ownerTag));
  if (item == _ownedBatches.end()) {
    // If requesting a batch inside another, a slower search algorithm must be used
    return createSpanInsideExistingBatch(begin, maxSize, normSet, ownerTag);
  }

  const auto& batch = item->second;
  maxSize = std::min(maxSize, batch.data.size() - (begin-batch.begin));

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
/// \param[in] normSet Optional normSet pointer to distinguish differently normalised computations.
/// \param[in] ownerTag Optional owner tag. This avoids reusing batch memory for e.g. getVal() and getLogVal().
/// \return An uninitialised RooSpan starting at event `begin`.
RooSpan<double> BatchData::makeWritableBatchUnInit(std::size_t begin, std::size_t batchSize,
    const RooArgSet* const normSet, Tag_t ownerTag) {
  auto item = _ownedBatches.find(std::make_tuple(begin, normSet, ownerTag));
  if (item == _ownedBatches.end()) {
    auto inserted = _ownedBatches.emplace(std::piecewise_construct,
        std::forward_as_tuple(begin, normSet, ownerTag),
        std::forward_as_tuple(Batch{begin, std::vector<double>(batchSize), kWriting}));
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
/// \param[in] normSet Optional normSet pointer to distinguish differently normalised computations.
/// \param[in] ownerTag Optional owner tag. This avoids reusing batch memory for e.g. getVal() and getLogVal().
/// \return An initialised RooSpan starting at event `begin`.
RooSpan<double> BatchData::makeWritableBatchInit(std::size_t begin, std::size_t batchSize, double value,
    const RooArgSet* const normSet, Tag_t ownerTag) {
  auto batch = makeWritableBatchUnInit(begin, batchSize, normSet, ownerTag);
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
    auto key = item.first;
    const Batch& batch = item.second;

    os << "\n" << indent
        << std::setw(8) << i << std::setw(8) << std::get<0>(key) << std::setw(8) << std::get<2>(key)
        << std::setw(7) << batch.status << ": {";
    for (unsigned int j=0; j < 5 && j < batch.data.size(); ++j) {
      os << batch.data[j] << ", ";
    }
    os << "...}";
  }
  os << std::resetiosflags(std::ios::adjustfield) << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the batch that contains the event with number `evt`.
/// \param[in] evt Index of the event to find.
/// \param[in] normSet Optional normalisation set defining what this batch was normalised to.
/// \param[in] ownerTag Optional owner tag to prevent sharing of memory between e.g. getVal() and getLogVal().
BatchData::Map_t::const_iterator BatchData::findEnclosingBatch(std::size_t evt,
    const RooArgSet* const normSet, Tag_t ownerTag) const {
  for (auto it = _ownedBatches.cbegin(); it != _ownedBatches.cend(); ++it) {
    if (normSet == std::get<1>(it->first)
        && ownerTag == std::get<2>(it->first)
        && it->second.inBatch(evt))
      return it;
  }

  return _ownedBatches.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Create a span pointing to existing batch memory.
/// \param[in] begin Index of the event to find.
/// \param[in] batchSize Requested size of the span. May come out smaller if no more data exists.
/// \param[in] normSet Optional normalisation set defining what this batch was normalised to.
/// \param[in] ownerTag Optional owner tag to prevent sharing of memory between e.g. getVal() and getLogVal().
/// \return RooSpan pointing inside an existing batch or an empty span if no such batch.
RooSpan<const double> BatchData::createSpanInsideExistingBatch(std::size_t begin, std::size_t batchSize,
    const RooArgSet* const normSet, Tag_t ownerTag) const {
  for (auto it = _ownedBatches.cbegin(); it != _ownedBatches.cend(); ++it) {
    if (normSet == std::get<1>(it->first)
        && ownerTag == std::get<2>(it->first)
        && it->second.inBatch(begin))
      return it->second.makeSpan(begin, batchSize);
  }

  return {};
}

}
