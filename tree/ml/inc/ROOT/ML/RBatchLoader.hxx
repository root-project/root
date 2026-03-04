// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025
// Author: Silia Taider, CERN 02/2026

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RBATCHLOADER
#define ROOT_INTERNAL_ML_RBATCHLOADER

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "ROOT/ML/RFlat2DMatrix.hxx"

namespace ROOT::Experimental::Internal::ML {
/**
 \class ROOT::Experimental::Internal::ML::RBatchLoader

\brief Building and loading the batches from loaded chunks in RChunkLoader

In this class the chunks that are loaded into memory (see RChunkLoader) are split into batches used in the ML training
which are loaded into a queue. This is done for both the training and validation chunks separately.
*/

class RBatchLoader {
private:
   std::size_t fBatchSize;
   // needed for calculating the total number of batch columns when vectors columns are present
   std::vector<std::string> fCols;
   std::mutex &fLock;
   std::condition_variable &fCV;
   std::vector<std::size_t> fVecSizes;
   std::size_t fSumVecSizes;
   std::size_t fNumColumns;
   std::size_t fNumEntries;
   bool fDropRemainder;

   std::size_t fNumFullBatches;
   std::size_t fNumBatches;
   std::size_t fLeftoverBatchSize;

   bool fIsActive = false;
   bool fProducerDone = true;

   // queues of flattened tensors (rows * cols)
   std::queue<std::unique_ptr<RFlat2DMatrix>> fBatchQueue;

   // current batch that is loaded into memory
   std::unique_ptr<RFlat2DMatrix> fCurrentBatch;

   // primary and secondary leftover batches used to create batches from a chunk
   std::unique_ptr<RFlat2DMatrix> fPrimaryLeftoverBatch;
   std::unique_ptr<RFlat2DMatrix> fSecondaryLeftoverBatch;

public:
   RBatchLoader(std::size_t batchSize, const std::vector<std::string> &cols, std::mutex &sharedMutex,
                std::condition_variable &sharedCV, const std::vector<std::size_t> &vecSizes = {},
                std::size_t numEntries = 0, bool dropRemainder = false);

   void Activate();
   void DeActivate();
   void Reset();
   void MarkProducerDone();

   std::unique_ptr<RFlat2DMatrix> CreateBatch(RFlat2DMatrix &chunkTensor, std::size_t idxs);
   RFlat2DMatrix GetBatch();
   void CreateBatches(RFlat2DMatrix &chunkTensor, bool isLastBatch);

   bool isProducerDone() { return fProducerDone; }
   std::size_t GetNumBatches() { return fNumBatches; }
   std::size_t GetNumEntries() { return fNumEntries; }
   std::size_t GetNumRemainderRows() { return fLeftoverBatchSize; }
   std::size_t GetNumBatchQueue() { return fBatchQueue.size(); }
};

} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RBATCHLOADER
