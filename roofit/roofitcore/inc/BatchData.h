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
#include <assert.h>

namespace BatchHelpers {

class BatchData {
  public:
    /// Status of the batch. Make sure that everything that is readable has
    /// a status >= kReady.
    enum Status_t {kDirtyOrUnknown, kWriting, kReady, kReadyAndConstant};

    BatchData() :
    _ownedData(),
    _foreignData(nullptr),
    _batchStatus(),
    _batchSize(0)
    {

    }

    /// Check if batch size is known.
    bool isInit() const {
      return !_batchStatus.empty();
    }

    void resizeAndClear(std::size_t size, std::size_t batchSize);

    Status_t status(std::size_t begin, std::size_t end) const {
      assert(isInit());
      assert(calcBatchIndex(begin) == calcBatchIndex(end-1));
      return _batchStatus[calcBatchIndex(begin)];
    }

    void setStatus(std::size_t begin, std::size_t end, Status_t status) {
      assert(isInit());
      assert(calcBatchIndex(begin) == calcBatchIndex(end-1));
      if (_foreignData)
        return;

      _batchStatus[calcBatchIndex(begin)] = status;
    }

    void reset();

    void markStale() {
      for (auto& elm : _batchStatus) {
        if (elm != kReadyAndConstant)
          elm = kDirtyOrUnknown;
      }
    }


    RooSpan<const double> makeBatch(std::size_t begin, std::size_t end) const;

    RooSpan<double> makeWritableBatchUnInit(std::size_t begin, std::size_t end);

    RooSpan<double> makeWritableBatchInit(std::size_t begin, std::size_t end, double value);

    void attachForeignStorage(const std::vector<double>& vec);

    void print(std::ostream& os, const std::string& indent) const;

    double operator[](std::size_t evt) const {
      assert(status(evt, evt+1) >= kReady);
      return data()[evt];
    }

  private:
    bool validRange(std::size_t begin, std::size_t end) const {
      return begin < data().size() && end <= data().size();
    }

    std::size_t calcBatchIndex(std::size_t begin) const;

    const std::vector<double>& data() const {
      return _foreignData ? *_foreignData : _ownedData;
    }

    std::vector<double>  _ownedData;
    const std::vector<double>* _foreignData;
    std::vector<Status_t> _batchStatus;
    std::vector<std::size_t> _batchStartPoints;
    std::size_t _batchSize;
};

}

#endif /* ROOFIT_ROOFITCORE_INC_BATCHDATA_H_ */
