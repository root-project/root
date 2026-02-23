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

#include <vector>
#include <memory>
#include <numeric>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

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
                std::size_t numEntries = 0, bool dropRemainder = false)
      : fBatchSize(batchSize),
        fCols(cols),
        fLock(sharedMutex),
        fCV(sharedCV),
        fVecSizes(vecSizes),
        fNumEntries(numEntries),
        fDropRemainder(dropRemainder)
   {
      fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);
      fNumColumns = fCols.size() + fSumVecSizes - fVecSizes.size();

      if (fBatchSize == 0) {
         fBatchSize = fNumEntries;
      }

      fLeftoverBatchSize = fNumEntries % fBatchSize;
      fNumFullBatches = fNumEntries / fBatchSize;

      std::size_t numLeftoverBatches = fLeftoverBatchSize == 0 ? 0 : 1;

      if (fDropRemainder) {
         fNumBatches = fNumFullBatches;
      } else {
         fNumBatches = fNumFullBatches + numLeftoverBatches;
      }

      fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
   }

   /// \brief Activate the batchloader. This means that batches can be created and loaded.
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fLock);
         if (fIsActive)
            return;
         fIsActive = true;
         fProducerDone = false;
      }

      fCV.notify_all();
   }

   /// \brief DeActivate the batchloader. This means that no more batches are created.
   /// Batches can still be returned if they are already loaded.
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fLock);
         if (!fIsActive)
            return;
         fIsActive = false;
      }

      fCV.notify_all();
   }

   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be destroyed.
   /// \param[in] chunkTensor Tensor with the data from the chunk
   /// \param[in] idxs Index of batch in the chunk
   /// \return Batch
   std::unique_ptr<RFlat2DMatrix> CreateBatch(RFlat2DMatrix &chunkTensor, std::size_t idxs)
   {
      auto batch = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
      std::copy(chunkTensor.GetData() + (idxs * fBatchSize * fNumColumns),
                chunkTensor.GetData() + ((idxs + 1) * fBatchSize * fNumColumns), batch->GetData());

      return batch;
   }

   /// \brief Loading the batch from the queue.
   /// \return Batch
   RFlat2DMatrix GetBatch()
   {
      std::unique_lock<std::mutex> lock(fLock);

      // Wait until:
      //  - there is data in the queue
      //  - or producer declares "done"
      //  - or we are deactivated
      fCV.wait(lock, [&] { return !fBatchQueue.empty() || fProducerDone || !fIsActive; });

      if (fBatchQueue.empty()) {
         // producer done and no queued data -> end-of-epoch signal
         fCurrentBatch = std::make_unique<RFlat2DMatrix>();
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fBatchQueue.front());
      fBatchQueue.pop();
      // Notify the loading thread that the queue has drained
      fCV.notify_all();

      return *fCurrentBatch;
   }

   /// \brief Creating the batches from a chunk and add them to the queue.
   /// \param[in] chunkTensor Tensor with the data from the chunk
   /// \param[in] isLastBatch Check if the batch in the chunk is the last one
   void CreateBatches(RFlat2DMatrix &chunkTensor, bool isLastBatch)
   {
      std::size_t chunkSize = chunkTensor.GetRows();
      std::size_t numCols = chunkTensor.GetCols();
      std::size_t numBatches = chunkSize / fBatchSize;
      std::size_t leftoverBatchSize = chunkSize % fBatchSize;

      // create a vector of batches
      std::vector<std::unique_ptr<RFlat2DMatrix>> batches;

      // fill the full batches from the chunk into a vector
      for (std::size_t i = 0; i < numBatches; i++) {
         batches.emplace_back(CreateBatch(chunkTensor, i));
      }

      // copy the remaining entries from the chunk into a leftover batch
      RFlat2DMatrix LeftoverBatch(leftoverBatchSize, numCols);
      std::copy(chunkTensor.GetData() + (numBatches * fBatchSize * numCols),
                chunkTensor.GetData() + (numBatches * fBatchSize * numCols + leftoverBatchSize * numCols),
                LeftoverBatch.GetData());

      // calculate how many empty slots are left in fPrimaryLeftoverBatch
      std::size_t PrimaryLeftoverSize = fPrimaryLeftoverBatch->GetRows();
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      // copy LeftoverBatch to end of fPrimaryLeftoverBatch
      if (emptySlots >= leftoverBatchSize) {
         fPrimaryLeftoverBatch->Resize(PrimaryLeftoverSize + leftoverBatchSize, numCols);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (leftoverBatchSize * fNumColumns),
                   fPrimaryLeftoverBatch->GetData() + (PrimaryLeftoverSize * numCols));

         // copy LeftoverBatch to end of fPrimaryLeftoverBatch and add it to the batch
         if (emptySlots == leftoverBatchSize) {
            auto copy = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverBatch->GetData(), fPrimaryLeftoverBatch->GetData() + (fBatchSize * fNumColumns),
                      copy->GetData());
            batches.emplace_back(std::move(copy));

            // reset fPrimaryLeftoverBatch and fSecondaryLeftoverBatch
            *fPrimaryLeftoverBatch = *fSecondaryLeftoverBatch;
            fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
         }
      }

      // copy LeftoverBatch to both fPrimaryLeftoverBatch and fSecondaryLeftoverBatch
      else if (emptySlots < leftoverBatchSize) {
         // copy the first part of LeftoverBatch to end of fPrimaryLeftoverTrainingBatch
         fPrimaryLeftoverBatch->Resize(fBatchSize, numCols);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * numCols),
                   fPrimaryLeftoverBatch->GetData() + (PrimaryLeftoverSize * numCols));

         // copy the last part of LeftoverBatch to the end of fSecondaryLeftoverBatch
         fSecondaryLeftoverBatch->Resize(leftoverBatchSize - emptySlots, numCols);
         std::copy(LeftoverBatch.GetData() + (emptySlots * numCols),
                   LeftoverBatch.GetData() + (leftoverBatchSize * numCols), fSecondaryLeftoverBatch->GetData());

         // add fPrimaryLeftoverBatch to the batch vector
         auto copy = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
         std::copy(fPrimaryLeftoverBatch->GetData(), fPrimaryLeftoverBatch->GetData() + (fBatchSize * fNumColumns),
                   copy->GetData());
         batches.emplace_back(std::move(copy));

         // exchange fPrimaryLeftoverBatch and fSecondaryLeftoverBatch
         *fPrimaryLeftoverBatch = *fSecondaryLeftoverBatch;
         // reset fSecondaryLeftoverTrainingBatch
         fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      }

      // copy the content of fPrimaryLeftoverBatch to the leftover batch from the chunk
      if (isLastBatch) {
         if (!fDropRemainder && fLeftoverBatchSize > 0) {
            auto copy = std::make_unique<RFlat2DMatrix>(fLeftoverBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverBatch->GetData(),
                      fPrimaryLeftoverBatch->GetData() + (fLeftoverBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
         }

         fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
         fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      }

      {
         std::lock_guard<std::mutex> lock(fLock);
         for (auto &batch : batches) {
            fBatchQueue.push(std::move(batch));
         }
      }

      fCV.notify_all();
   }

   /// \brief Reset the batchloader state.
   void Reset()
   {
      {
         std::lock_guard<std::mutex> lock(fLock);

         while (!fBatchQueue.empty()) {
            fBatchQueue.pop();
         }

         fCurrentBatch.reset();
         fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
         fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      }

      fCV.notify_all();
   }

   /// \brief Signal that the producer has finished pushing all batches for this epoch.
   void MarkProducerDone()
   {
      fProducerDone = true;
      fCV.notify_all();
   }

   bool isProducerDone() { return fProducerDone; }
   std::size_t GetNumBatches() { return fNumBatches; }
   std::size_t GetNumEntries() { return fNumEntries; }
   std::size_t GetNumRemainderRows() { return fLeftoverBatchSize; }
   std::size_t GetNumBatchQueue() { return fBatchQueue.size(); }
};

} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RBATCHLOADER
