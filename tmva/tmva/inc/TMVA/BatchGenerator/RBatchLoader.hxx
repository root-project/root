// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RBATCHLOADER
#define TMVA_RBATCHLOADER

#include <vector>
#include <memory>
#include <numeric>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"
#include "TMVA/Tools.h"

namespace TMVA {
namespace Experimental {
namespace Internal {

class RBatchLoader {
private:
   const TMVA::Experimental::RTensor<float> &fChunkTensor;
   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fMaxBatches;
   std::size_t fTrainingRemainderRow = 0;
   std::size_t fValidationRemainderRow = 0;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainingRemainder;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainder;

public:
   RBatchLoader(const TMVA::Experimental::RTensor<float> &chunkTensor, const std::size_t batchSize,
                const std::size_t numColumns, const std::size_t maxBatches)
      : fChunkTensor(chunkTensor), fBatchSize(batchSize), fNumColumns(numColumns), fMaxBatches(maxBatches)
   {
      // Create remainders tensors
      fTrainingRemainder =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
      fValidationRemainder =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
   }

   ~RBatchLoader() { DeActivate(); }

public:
   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be destroyed.
   /// \return Training batch
   const TMVA::Experimental::RTensor<float> &GetTrainBatch()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() { return !fTrainingBatchQueue.empty() || !fIsActive; });

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      fBatchCondition.notify_all();

      return *fCurrentBatch;
   }

   /// \brief Returns a batch of data for validation
   /// The owner of this batch has to be with the RBatchLoader.
   /// This is because the same validation batches should be used in all epochs.
   /// \return Validation batch
   const TMVA::Experimental::RTensor<float> &GetValidationBatch()
   {
      if (fValidationBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fValidationBatchQueue.front());
      fValidationBatchQueue.pop();

      return *fCurrentBatch;
   }

   /// \brief Activate the batchloader so it will accept chunks to batch
   void Activate()
   {
      fTrainingRemainderRow = 0;
      fValidationRemainderRow = 0;

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

   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(const TMVA::Experimental::RTensor<float> &chunkTensor, std::span<const std::size_t> idxs,
               std::size_t batchSize)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({batchSize, fNumColumns}));

      for (std::size_t i = 0; i < batchSize; i++) {
         std::copy(chunkTensor.GetData() + (idxs[i] * fNumColumns),
                   chunkTensor.GetData() + ((idxs[i] + 1) * fNumColumns), batch->GetData() + i * fNumColumns);
      }

      return batch;
   }

   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateFirstBatch(const TMVA::Experimental::RTensor<float> &remainderTensor, std::size_t remainderTensorRow,
                    std::span<const std::size_t> eventIndices)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));

      for (size_t i = 0; i < remainderTensorRow; i++) {
         std::copy(remainderTensor.GetData() + i * fNumColumns, remainderTensor.GetData() + (i + 1) * fNumColumns,
                   batch->GetData() + i * fNumColumns);
      }

      for (std::size_t i = 0; i < (fBatchSize - remainderTensorRow); i++) {
         std::copy(fChunkTensor.GetData() + eventIndices[i] * fNumColumns,
                   fChunkTensor.GetData() + (eventIndices[i] + 1) * fNumColumns,
                   batch->GetData() + (i + remainderTensorRow) * fNumColumns);
      }

      return batch;
   }

   /// @brief save to remaining data when the whole chunk has to be saved
   /// @param chunkTensor
   /// @param remainderTensor
   /// @param remainderTensorRow
   /// @param eventIndices
   void SaveRemainingData(TMVA::Experimental::RTensor<float> &remainderTensor, const std::size_t remainderTensorRow,
                          const std::vector<std::size_t> eventIndices, const std::size_t start = 0)
   {
      for (std::size_t i = start; i < eventIndices.size(); i++) {
         std::copy(fChunkTensor.GetData() + eventIndices[i] * fNumColumns,
                   fChunkTensor.GetData() + (eventIndices[i] + 1) * fNumColumns,
                   remainderTensor.GetData() + (i - start + remainderTensorRow) * fNumColumns);
      }
   }

   /// \brief Create training batches from the given chunk of data based on the given event indices
   /// Batches are added to the training queue of batches
   /// \param chunkTensor
   /// \param eventIndices
   void CreateTrainingBatches(const std::vector<std::size_t> &eventIndices)
   {
      // Wait until less than a full chunk of batches are in the queue before splitting the next chunk into
      // batches
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         fBatchCondition.wait(lock, [this]() { return (fTrainingBatchQueue.size() < fMaxBatches) || !fIsActive; });
         if (!fIsActive)
            return;
      }

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      if (eventIndices.size() + fTrainingRemainderRow >= fBatchSize) {
         batches.emplace_back(CreateFirstBatch(*fTrainingRemainder, fTrainingRemainderRow, eventIndices));
      } else {
         SaveRemainingData(*fTrainingRemainder, fTrainingRemainderRow, eventIndices);
         fTrainingRemainderRow += eventIndices.size();
         return;
      }

      // Create tasks of fBatchSize until all idx are used
      std::size_t start = fBatchSize - fTrainingRemainderRow;
      for (; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {
         // Grab the first fBatchSize indices
         std::span<const std::size_t> idxs{eventIndices.data() + start, eventIndices.data() + start + fBatchSize};

         // Fill a batch
         batches.emplace_back(CreateBatch(fChunkTensor, idxs, fBatchSize));
      }

      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         for (std::size_t i = 0; i < batches.size(); i++) {
            fTrainingBatchQueue.push(std::move(batches[i]));
         }
      }

      fBatchCondition.notify_all();

      fTrainingRemainderRow = eventIndices.size() - start;
      SaveRemainingData(*fTrainingRemainder, 0, eventIndices, start);
   }

   /// \brief Create validation batches from the given chunk based on the given event indices
   /// Batches are added to the vector of validation batches
   /// \param chunkTensor
   /// \param eventIndices
   void CreateValidationBatches(const std::vector<std::size_t> &eventIndices)
   {
      if (eventIndices.size() + fValidationRemainderRow >= fBatchSize) {
         fValidationBatchQueue.push(CreateFirstBatch(*fValidationRemainder, fValidationRemainderRow, eventIndices));
      } else {
         SaveRemainingData(*fValidationRemainder, fValidationRemainderRow, eventIndices);
         fValidationRemainderRow += eventIndices.size();
         return;
      }

      // Create tasks of fBatchSize untill all idx are used
      std::size_t start = fBatchSize - fValidationRemainderRow;
      for (; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {

         std::vector<std::size_t> idx;

         for (std::size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         fValidationBatchQueue.push(CreateBatch(fChunkTensor, idx, fBatchSize));
      }

      fValidationRemainderRow = eventIndices.size() - start;
      SaveRemainingData(*fValidationRemainder, 0, eventIndices, start);
   }

   void LastBatches()
   {
      {
         if (fTrainingRemainderRow) {
            std::vector<std::size_t> idx = std::vector<std::size_t>(fTrainingRemainderRow);
            std::iota(idx.begin(), idx.end(), 0);

            std::unique_ptr<TMVA::Experimental::RTensor<float>> batch =
               CreateBatch(*fTrainingRemainder, idx, fTrainingRemainderRow);

            std::unique_lock<std::mutex> lock(fBatchLock);
            fTrainingBatchQueue.push(std::move(batch));
         }
      }

      if (fValidationRemainderRow) {
         std::vector<std::size_t> idx = std::vector<std::size_t>(fValidationRemainderRow);
         std::iota(idx.begin(), idx.end(), 0);

         fValidationBatchQueue.push(CreateBatch(*fValidationRemainder, idx, fValidationRemainderRow));
      }
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBATCHLOADER
