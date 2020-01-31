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
#include <map>
#include <tuple>
#include <assert.h>

class RooArgSet;

namespace BatchHelpers {

/// A class to store batches of data points that can be accessed via RooSpan.
class BatchData {
  public:
    /// Status of the batch. Make sure that everything that is readable has
    /// a status >= kReady.
    enum Status_t {kNoBatch, kDirty, kWriting, kReady, kReadyAndConstant};
    enum Tag_t {kUnspecified, kgetVal, kgetLogVal};

    BatchData() :
      _ownedBatches(),
      _foreignData(nullptr)
    {

    }

    /// Discard all storage.
    void clear() {
      _ownedBatches.clear();
    }

    Status_t status(std::size_t begin, const RooArgSet* const normSet = nullptr, Tag_t ownerTag = kUnspecified) const;

    bool setStatus(std::size_t begin, std::size_t size, Status_t stat,
        const RooArgSet* const normSet = nullptr, Tag_t ownerTag = kUnspecified);

    /// Mark all batches dirty. This will trigger recomputations.
    void markDirty() {
      for (auto& elm : _ownedBatches) {
        if (elm.second.status != kReadyAndConstant)
          elm.second.status = kDirty;
      }
    }


    RooSpan<const double> getBatch(std::size_t begin, std::size_t maxSize,
        const RooArgSet* const normSet = nullptr, Tag_t ownerTag = kUnspecified) const;

    RooSpan<double> makeWritableBatchUnInit(std::size_t begin, std::size_t batchSize,
        const RooArgSet* const normSet = nullptr, Tag_t ownerTag = kUnspecified);

    RooSpan<double> makeWritableBatchInit(std::size_t begin, std::size_t batchSize, double value,
        const RooArgSet* const normSet = nullptr, Tag_t ownerTag = kUnspecified);

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
        assert(inBatch(evt));
        return RooSpan<const double>(data.data() + (evt-begin), std::min(batchSize, data.size()));
      }
    };

    /// Key type of map that holds the batch storage.
    using Key_t = std::tuple<std::size_t, const RooArgSet* const, Tag_t>;

    //  A small benchmark of map vs. unordered map showed that a map finds elements faster up to 1000 elements.
    //  The usual size should be <~= 10 elements.
    /// Storage for batch data.
    using Map_t = std::map<Key_t, Batch>;


    Map_t::const_iterator findEnclosingBatch(std::size_t evt,
        const RooArgSet* const normSet, Tag_t ownerTag) const;
    RooSpan<const double> createSpanInsideExistingBatch(std::size_t begin, std::size_t batchSize,
        const RooArgSet* const normSet, Tag_t ownerTag) const;


    Map_t _ownedBatches;
    const std::vector<double>* _foreignData;
};

}

#endif /* ROOFIT_ROOFITCORE_INC_BATCHDATA_H_ */
