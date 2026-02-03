// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025

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
   std::vector<std::size_t> fVecSizes;
   std::size_t fSumVecSizes;
   std::size_t fNumColumns;
   std::size_t fNumEntries;
   bool fDropRemainder;

   std::size_t fNumFullBatches;
   std::size_t fNumLeftoverBatches;
   std::size_t fNumBatches;
   std::size_t fLeftoverBatchSize;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   // queues of flattened tensors (rows * cols)
   std::queue<std::unique_ptr<RFlat2DMatrix>> fBatchQueue;

   // current batch that is loaded into memory
   std::unique_ptr<RFlat2DMatrix> fCurrentBatch;

   // primary and secondary leftover batches used to create batches from a chunk
   std::unique_ptr<RFlat2DMatrix> fPrimaryLeftoverBatch;
   std::unique_ptr<RFlat2DMatrix> fSecondaryLeftoverBatch;

public:
   RBatchLoader(std::size_t batchSize, const std::vector<std::string> &cols,
                const std::vector<std::size_t> &vecSizes = {}, std::size_t numEntries = 0, bool dropRemainder = false)
      : fBatchSize(batchSize), fCols(cols), fVecSizes(vecSizes), fNumEntries(numEntries), fDropRemainder(dropRemainder)
   {

      fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);
      fNumColumns = fCols.size() + fSumVecSizes - fVecSizes.size();

      if (fBatchSize == 0) {
         fBatchSize = fNumEntries;
      }

      fLeftoverBatchSize = fNumEntries % fBatchSize;
      fNumFullBatches = fNumEntries / fBatchSize;

      fNumLeftoverBatches = fLeftoverBatchSize == 0 ? 0 : 1;

      if (fDropRemainder) {
         fNumBatches = fNumFullBatches;
      }

      else {
         fNumBatches = fNumFullBatches + fNumLeftoverBatches;
      }

      fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
   }

public:
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = true;
      }
      fBatchCondition.notify_all();
   }

   /// \brief DeActivate the batchloader. This means that no more batches are created.
   /// Batches can still be returned if they are already loaded
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = false;
      }
      fBatchCondition.notify_all();
   }

   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be destroyed.
   /// \param[in] chunkTensor Tensor with the data from the chunk
   /// \param[in] idxs Index of batch in the chunk
   /// \return Batch
   std::unique_ptr<RFlat2DMatrix> CreateBatch(RFlat2DMatrix &chunTensor, std::size_t idxs)
   {
      auto batch = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
      std::copy(chunTensor.GetData() + (idxs * fBatchSize * fNumColumns),
                chunTensor.GetData() + ((idxs + 1) * fBatchSize * fNumColumns), batch->GetData());

      return batch;
   }

   /// \brief Loading the batch from the queue
   /// \return Batch
   RFlat2DMatrix GetBatch()
   {

      if (fBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<RFlat2DMatrix>();
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fBatchQueue.front());
      fBatchQueue.pop();

      return *fCurrentBatch;
   }

   /// \brief Creating the batches from a chunk and add them to the queue.
   /// \param[in] chunkTensor Tensor with the data from the chunk
   /// \param[in] lastbatch Check if the batch in the chunk is the last one
   void CreateBatches(RFlat2DMatrix &chunkTensor, std::size_t lastbatch)
   {
      std::size_t ChunkSize = chunkTensor.GetRows();
      std::size_t NumCols = chunkTensor.GetCols();
      std::size_t Batches = ChunkSize / fBatchSize;
      std::size_t LeftoverBatchSize = ChunkSize % fBatchSize;

      // create a vector of batches
      std::vector<std::unique_ptr<RFlat2DMatrix>> batches;

      // fill the full batches from the chunk into a vector
      for (std::size_t i = 0; i < Batches; i++) {
         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, i));
      }

      // copy the remaining entries from the chunk into a leftover batch
      RFlat2DMatrix LeftoverBatch(LeftoverBatchSize, NumCols);
      std::copy(chunkTensor.GetData() + (Batches * fBatchSize * NumCols),
                chunkTensor.GetData() + (Batches * fBatchSize * NumCols + LeftoverBatchSize * NumCols),
                LeftoverBatch.GetData());

      // calculate how many empty slots are left in fPrimaryLeftoverBatch
      std::size_t PrimaryLeftoverSize = fPrimaryLeftoverBatch->GetRows();
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      // copy LeftoverBatch to end of fPrimaryLeftoverBatch
      if (emptySlots >= LeftoverBatchSize) {
         fPrimaryLeftoverBatch->Resize(PrimaryLeftoverSize + LeftoverBatchSize, NumCols);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fPrimaryLeftoverBatch->GetData() + (PrimaryLeftoverSize * NumCols));

         // copy LeftoverBatch to end of fPrimaryLeftoverBatch and add it to the batch vector
         if (emptySlots == LeftoverBatchSize) {
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
      else if (emptySlots < LeftoverBatchSize) {
         // copy the first part of LeftoverBatch to end of fPrimaryLeftoverTrainingBatch
         fPrimaryLeftoverBatch->Resize(fBatchSize, NumCols);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * NumCols),
                   fPrimaryLeftoverBatch->GetData() + (PrimaryLeftoverSize * NumCols));

         // copy the last part of LeftoverBatch to the end of fSecondaryLeftoverBatch
         fSecondaryLeftoverBatch->Resize(LeftoverBatchSize - emptySlots, NumCols);
         std::copy(LeftoverBatch.GetData() + (emptySlots * NumCols),
                   LeftoverBatch.GetData() + (LeftoverBatchSize * NumCols), fSecondaryLeftoverBatch->GetData());

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
      if (lastbatch == 1) {

         if (fDropRemainder == false && fLeftoverBatchSize > 0) {
            auto copy = std::make_unique<RFlat2DMatrix>(fLeftoverBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverBatch->GetData(),
                      fPrimaryLeftoverBatch->GetData() + (fLeftoverBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
         }

         fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
         fSecondaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      }

      // append the batches from the batch vector from the chunk to the training batch queue
      for (std::size_t i = 0; i < batches.size(); i++) {
         fBatchQueue.push(std::move(batches[i]));
      }
   }

   std::size_t GetNumBatches() { return fNumBatches; }
   std::size_t GetNumEntries() { return fNumEntries; }
   std::size_t GetNumRemainderRows() { return fLeftoverBatchSize; }
   std::size_t GetNumBatchQueue() { return fBatchQueue.size(); }
};

} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RBATCHLOADER
