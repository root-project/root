// Author: Stephan Hageboeck, CERN  10 Apr 2019

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

#ifndef ROOFIT_ROOFITCORE_INC_BATCHDATA_H_
#define ROOFIT_ROOFITCORE_INC_BATCHDATA_H_

#include "RooSpan.h"
#include <unordered_map>
#include <assert.h>

namespace BatchHelpers {

class BatchData {
  public:
    /// Status of the batch. Make sure that everything that is readable has
    /// a status >= kReady.
    enum Status_t {kNoBatch, kDirty, kWriting, kReady, kReadyAndConstant};

    BatchData() :
      _ownedBatches(),
      _foreignData(nullptr)
    {

    }

    /// Discard all storage.
    void clear() {
      _ownedBatches.clear();
    }

    /// Return the status of the batch starting at `begin`.
    /// \param[in] begin Start of the batch.
    /// \param[in] size  Size of the batch. This is used to check if the size of an existing batch
    /// matches the requested size. Asking for a too large size will be signalled by kNoBatch.
    Status_t status(std::size_t begin, std::size_t size) const {
      if (_foreignData && begin+size <= _foreignData->size())
        return kReadyAndConstant;
      else if (_foreignData)
        return kNoBatch;

      if (_ownedBatches.empty())
        return kNoBatch;

      auto item = _ownedBatches.find(begin);

      if (item == _ownedBatches.end()) {
        // We didn't find a batch that starts with `begin`. Check if there's
        // a batch that's enclosing the requested range.
        // This can be slow, but a subset of a batch is asked for:
        item = findEnclosingBatch(begin);
        if (item == _ownedBatches.end())
          return kNoBatch;

        auto item2 = findEnclosingBatch(begin+size-1);

        return item == item2 ? item->second.status : kNoBatch;
      }

      const Batch& batch = item->second;
      if (size <= batch.data.size())
        return batch.status;
      else
        return kNoBatch;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Set the status of a batch with the given start point and size.
    ///
    /// The status of foreign read-only data will never change.
    /// \return True if status successfully set, false if no such batch / not writable.
    bool setStatus(std::size_t begin, std::size_t size, Status_t stat) {
      if (_foreignData)
        return false;

      auto item = _ownedBatches.find(begin);
      if (item == _ownedBatches.end() || size != item->second.data.size())
        return false;

      item->second.status = stat;
      return true;
    }

    /// Mark all batches dirty. This will trigger recomputations.
    void markDirty() {
      for (auto& elm : _ownedBatches) {
        if (elm.second.status != kReadyAndConstant)
          elm.second.status = kDirty;
      }
    }


    RooSpan<const double> getBatch(std::size_t begin, std::size_t batchSize) const;

    RooSpan<double> makeWritableBatchUnInit(std::size_t begin, std::size_t batchSize);

    RooSpan<double> makeWritableBatchInit(std::size_t begin, std::size_t batchSize, double value);

    void attachForeignStorage(const std::vector<double>& vec);

    void print(std::ostream& os, const std::string& indent) const;



  private:

    struct Batch {
      std::size_t begin;
      std::vector<double> data;
      Status_t status;

      bool inBatch(std::size_t evt) const {
        return begin <= evt && evt < begin + data.size();
      }

      RooSpan<const double> makeSpan(std::size_t evt, std::size_t batchSize) const {
        return RooSpan<const double>(&data[evt-begin], batchSize);
      }
    };


    bool validRange(std::size_t begin, std::size_t size) const {
      if (_foreignData) {
        return begin < _foreignData->size() && begin+size <= _foreignData->size();
      }

      auto batch = findSpanInsideExistingBatch(begin, size);
      return !batch.empty();
    }


    using Map_t = std::unordered_map<std::size_t, Batch>;
    Map_t::const_iterator findEnclosingBatch(std::size_t evt) const {
      for (auto it = _ownedBatches.cbegin(); it != _ownedBatches.cend(); ++it) {
        if (it->second.inBatch(evt))
          return it;
      }

      return _ownedBatches.end();
    }

    RooSpan<const double> findSpanInsideExistingBatch(std::size_t begin, std::size_t batchSize) const {
      for (auto it = _ownedBatches.cbegin(); it != _ownedBatches.cend(); ++it) {
        if (it->second.inBatch(begin) && it->second.inBatch(begin+batchSize-1))
          return it->second.makeSpan(begin, batchSize);
      }

      return RooSpan<const double>();
    }

    Map_t _ownedBatches;
    const std::vector<double>* _foreignData;
};

}

#endif /* ROOFIT_ROOFITCORE_INC_BATCHDATA_H_ */
