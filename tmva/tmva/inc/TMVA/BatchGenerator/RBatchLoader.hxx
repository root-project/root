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

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RBatchLoader
\ingroup tmva
\brief Building and loading the batches from loaded chunks in RChunkLoader

In this class the chunks that are loaded into memory (see RChunkLoader) are split into batches used in the ML training which are loaded into a queue. This is done for both the training and validation chunks separatly.
*/

class RBatchLoader {
private:
   // clang-format on      
   std::size_t fChunkSize;
   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fMaxBatches;
   std::size_t fTrainingRemainderRow = 0;
   std::size_t fValidationRemainderRow = 0;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   // queuse of tensors of the training and validation batches
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;

   // number of training and validation batches in the queue
   std::size_t fNumTrainingBatchQueue;
   std::size_t fNumValidationBatchQueue;

   // current batch that is loaded into memeory
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   // primary and secondary batches used to create batches from a chunk
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fPrimaryLeftoverTrainingBatch;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fSecondaryLeftoverTrainingBatch;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fPrimaryLeftoverValidationBatch;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fSecondaryLeftoverValidationBatch;

public:
   RBatchLoader(std::size_t chunkSize, std::size_t batchSize, std::size_t numColumns)
      : fChunkSize(chunkSize), fBatchSize(batchSize), fNumColumns(numColumns)
   {

      fPrimaryLeftoverTrainingBatch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      fSecondaryLeftoverTrainingBatch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});

      fPrimaryLeftoverValidationBatch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      fSecondaryLeftoverValidationBatch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});

      fNumTrainingBatchQueue = fTrainingBatchQueue.size();
      fNumValidationBatchQueue = fValidationBatchQueue.size();
   }

public:
   void Activate()
   {
      // fTrainingRemainderRow = 0;
      // fValidationRemainderRow = 0;

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
   /// \param[in] chunkTensor RTensor with the data from the chunk
   /// \param[in] idxs Index of batch in the chunk
   /// \return Training batch
   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(TMVA::Experimental::RTensor<float> &chunkTensor, std::size_t idxs)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));
      std::copy(chunkTensor.GetData() + (idxs * fBatchSize * fNumColumns),
                chunkTensor.GetData() + ((idxs + 1) * fBatchSize * fNumColumns), batch->GetData());

      return batch;
   }

   
   /// \brief Loading the training batch from the queue
   /// \return Training batch
   TMVA::Experimental::RTensor<float> GetTrainBatch()
   {

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      return *fCurrentBatch;
   }

   /// \brief Loading the validation batch from the queue
   /// \return Training batch
   TMVA::Experimental::RTensor<float> GetValidationBatch()
   {

      if (fValidationBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fValidationBatchQueue.front());
      fValidationBatchQueue.pop();

      return *fCurrentBatch;
   }

   /// \brief Creating the training batches from a chunk and add them to the queue.
   /// \param[in] chunkTensor RTensor with the data from the chunk
   /// \param[in] lastbatch Check if the batch in the chunk is the last one
   /// \param[in] leftoverBatchSize Size of the leftover batch in the training dataset
   /// \param[in] dromRemainder Bool to drop the remainder batch or not
   void CreateTrainingBatches(TMVA::Experimental::RTensor<float> &chunkTensor, int lastbatch,
                              std::size_t leftoverBatchSize, bool dropRemainder)
   {
      std::size_t ChunkSize = chunkTensor.GetShape()[0];
      std::size_t Batches = ChunkSize / fBatchSize;
      std::size_t LeftoverBatchSize = ChunkSize % fBatchSize;

      // create a vector of batches
      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      // fill the full batches from the chunk into a vector
      for (std::size_t i = 0; i < Batches; i++) {
         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, i));
      }

      // copy the remaining entries from the chunk into a leftover batch
      TMVA::Experimental::RTensor<float> LeftoverBatch({LeftoverBatchSize, fNumColumns});
      std::copy(chunkTensor.GetData() + (Batches * fBatchSize * fNumColumns),
                chunkTensor.GetData() + (Batches * fBatchSize * fNumColumns + LeftoverBatchSize * fNumColumns),
                LeftoverBatch.GetData());

      // calculate how many empty slots are left in fPrimaryLeftoverTrainingBatch
      std::size_t PrimaryLeftoverSize = (*fPrimaryLeftoverTrainingBatch).GetShape()[0];
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      // copy LeftoverBatch to end of fPrimaryLeftoverTrainingBatch
      if (emptySlots >= LeftoverBatchSize) {
         (*fPrimaryLeftoverTrainingBatch) =
            (*fPrimaryLeftoverTrainingBatch).Resize({PrimaryLeftoverSize + LeftoverBatchSize, fNumColumns});
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fPrimaryLeftoverTrainingBatch->GetData() + (PrimaryLeftoverSize * fNumColumns));

         // copy LeftoverBatch to end of fPrimaryLeftoverTrainingBatch and add it to the batch vector
         if (emptySlots == LeftoverBatchSize) {
            auto copy =
               std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
            std::copy(fPrimaryLeftoverTrainingBatch->GetData(),
                      fPrimaryLeftoverTrainingBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));

            // reset fPrimaryLeftoverTrainingBatch and fSecondaryLeftoverTrainingBatch
            *fPrimaryLeftoverTrainingBatch = *fSecondaryLeftoverTrainingBatch;
            fSecondaryLeftoverValidationBatch =
               std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
         }
      }

      // copy LeftoverBatch to both fPrimaryLeftoverTrainingBatch and fSecondaryLeftoverTrainingBatch
      else if (emptySlots < LeftoverBatchSize) {
         // copy the first part of LeftoverBatch to end of fPrimaryLeftoverTrainingBatch 
         (*fPrimaryLeftoverTrainingBatch) = (*fPrimaryLeftoverTrainingBatch).Resize({fBatchSize, fNumColumns});
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * fNumColumns),
                   fPrimaryLeftoverTrainingBatch->GetData() + (PrimaryLeftoverSize * fNumColumns));

         // copy the last part of LeftoverBatch to the end of fSecondaryLeftoverTrainingBatch
         (*fSecondaryLeftoverTrainingBatch) =
            (*fSecondaryLeftoverTrainingBatch).Resize({LeftoverBatchSize - emptySlots, fNumColumns});
         std::copy(LeftoverBatch.GetData() + (emptySlots * fNumColumns),
                   LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fSecondaryLeftoverTrainingBatch->GetData());
         
         // add fPrimaryLeftoverTrainingBatch to the batch vector
         auto copy =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
         std::copy(fPrimaryLeftoverTrainingBatch->GetData(),
                   fPrimaryLeftoverTrainingBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
         batches.emplace_back(std::move(copy));
         
         // exchange fPrimaryLeftoverTrainingBatch and fSecondaryLeftoverValidationBatch
         *fPrimaryLeftoverTrainingBatch = *fSecondaryLeftoverTrainingBatch;
         
         // restet fSecondaryLeftoverValidationBatch
         fSecondaryLeftoverValidationBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      }

      // copy the content of fPrimaryLeftoverTrainingBatch to the leftover batch from the chunk
      if (lastbatch == 1) {

         if (dropRemainder == false && leftoverBatchSize > 0) {
            auto copy = std::make_unique<TMVA::Experimental::RTensor<float>>(
               std::vector<std::size_t>{leftoverBatchSize, fNumColumns});
            std::copy((*fPrimaryLeftoverTrainingBatch).GetData(),
                      (*fPrimaryLeftoverTrainingBatch).GetData() + (leftoverBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
         }

         fPrimaryLeftoverTrainingBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
         fSecondaryLeftoverTrainingBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      }

      // append the batches from the batch vector from the chunk to the training batch queue
      for (std::size_t i = 0; i < batches.size(); i++) {
         fTrainingBatchQueue.push(std::move(batches[i]));
      }
   }
   
   /// \brief Creating the validation batches from a chunk and adding them to the queue
   /// \param[in] chunkTensor RTensor with the data from the chunk
   /// \param[in] lastbatch Check if the batch in the chunk is the last one
   /// \param[in] leftoverBatchSize Size of the leftover batch in the validation dataset
   /// \param[in] dromRemainder Bool to drop the remainder batch or not
   void CreateValidationBatches(TMVA::Experimental::RTensor<float> &chunkTensor, std::size_t lastbatch,
                                std::size_t leftoverBatchSize, bool dropRemainder)
   {
      std::size_t ChunkSize = chunkTensor.GetShape()[0];
      std::size_t NumCols = chunkTensor.GetShape()[1];
      std::size_t Batches = ChunkSize / fBatchSize;
      std::size_t LeftoverBatchSize = ChunkSize % fBatchSize;

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      for (std::size_t i = 0; i < Batches; i++) {
         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, i));
      }

      TMVA::Experimental::RTensor<float> LeftoverBatch({LeftoverBatchSize, NumCols});
      std::copy(chunkTensor.GetData() + (Batches * fBatchSize * NumCols),
                chunkTensor.GetData() + (Batches * fBatchSize * NumCols + LeftoverBatchSize * NumCols),
                LeftoverBatch.GetData());

      std::size_t PrimaryLeftoverSize = (*fPrimaryLeftoverValidationBatch).GetShape()[0];
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      if (emptySlots >= LeftoverBatchSize) {
         (*fPrimaryLeftoverValidationBatch) =
            (*fPrimaryLeftoverValidationBatch).Resize({PrimaryLeftoverSize + LeftoverBatchSize, NumCols});
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (LeftoverBatchSize * NumCols),
                   fPrimaryLeftoverValidationBatch->GetData() + (PrimaryLeftoverSize * NumCols));

         if (emptySlots == LeftoverBatchSize) {
            auto copy =
               std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
            std::copy(fPrimaryLeftoverValidationBatch->GetData(),
                      fPrimaryLeftoverValidationBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
            *fPrimaryLeftoverValidationBatch = *fSecondaryLeftoverValidationBatch;
            fSecondaryLeftoverValidationBatch =
               std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
         }
      }

      else if (emptySlots < LeftoverBatchSize) {
         (*fPrimaryLeftoverValidationBatch) = (*fPrimaryLeftoverValidationBatch).Resize({fBatchSize, NumCols});
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * NumCols),
                   fPrimaryLeftoverValidationBatch->GetData() + (PrimaryLeftoverSize * NumCols));
         (*fSecondaryLeftoverValidationBatch) =
            (*fSecondaryLeftoverValidationBatch).Resize({LeftoverBatchSize - emptySlots, NumCols});
         std::copy(LeftoverBatch.GetData() + (emptySlots * NumCols),
                   LeftoverBatch.GetData() + (LeftoverBatchSize * NumCols),
                   fSecondaryLeftoverValidationBatch->GetData());
         auto copy =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
         std::copy(fPrimaryLeftoverValidationBatch->GetData(),
                   fPrimaryLeftoverValidationBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
         batches.emplace_back(std::move(copy));
         *fPrimaryLeftoverValidationBatch = *fSecondaryLeftoverValidationBatch;
         fSecondaryLeftoverValidationBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      }

      if (lastbatch == 1) {

         if (dropRemainder == false && leftoverBatchSize > 0) {
            auto copy = std::make_unique<TMVA::Experimental::RTensor<float>>(
               std::vector<std::size_t>{leftoverBatchSize, fNumColumns});
            std::copy((*fPrimaryLeftoverValidationBatch).GetData(),
                      (*fPrimaryLeftoverValidationBatch).GetData() + (leftoverBatchSize * fNumColumns),
                      copy->GetData());
            batches.emplace_back(std::move(copy));
         }
         fPrimaryLeftoverValidationBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
         fSecondaryLeftoverValidationBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      }

      for (std::size_t i = 0; i < batches.size(); i++) {
         fValidationBatchQueue.push(std::move(batches[i]));
      }
   }
   std::size_t GetNumTrainingBatchQueue() { return fTrainingBatchQueue.size(); }
   std::size_t GetNumValidationBatchQueue() { return fValidationBatchQueue.size(); }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBATCHLOADER
