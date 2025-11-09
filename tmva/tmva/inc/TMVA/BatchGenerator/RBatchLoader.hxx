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

#include "TMVA/BatchGenerator/RFlat2DMatrix.hxx"
#include "TMVA/Tools.h"

namespace TMVA::Experimental::Internal {

/**
\class ROOT::TMVA::Experimental::Internal::RBatchLoader
\ingroup tmva
\brief Building and loading the batches from loaded chunks in RChunkLoader

In this class the chunks that are loaded into memory (see RChunkLoader) are split into batches used in the ML training
which are loaded into a queue. This is done for both the training and validation chunks separately.
*/

class RBatchLoader {
private:
   std::size_t fBatchSize;
   std::size_t fNumColumns;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   // queues of flattened tensors (rows * cols)
   std::queue<std::unique_ptr<RFlat2DMatrix>> fTrainingBatchQueue;
   std::queue<std::unique_ptr<RFlat2DMatrix>> fValidationBatchQueue;

   // number of training and validation batches in the queue
   std::size_t fNumTrainingBatchQueue;
   std::size_t fNumValidationBatchQueue;

   // current batch that is loaded into memory
   std::unique_ptr<RFlat2DMatrix> fCurrentBatch;

   // primary and secondary leftover batches used to create batches from a chunk
   std::unique_ptr<RFlat2DMatrix> fPrimaryLeftoverTrainingBatch;
   std::unique_ptr<RFlat2DMatrix> fSecondaryLeftoverTrainingBatch;

   std::unique_ptr<RFlat2DMatrix> fPrimaryLeftoverValidationBatch;
   std::unique_ptr<RFlat2DMatrix> fSecondaryLeftoverValidationBatch;

public:
   RBatchLoader(std::size_t batchSize, std::size_t numColumns) : fBatchSize(batchSize), fNumColumns(numColumns)
   {

      fPrimaryLeftoverTrainingBatch = std::make_unique<RFlat2DMatrix>();
      fSecondaryLeftoverTrainingBatch = std::make_unique<RFlat2DMatrix>();

      fPrimaryLeftoverValidationBatch = std::make_unique<RFlat2DMatrix>();
      fSecondaryLeftoverValidationBatch = std::make_unique<RFlat2DMatrix>();

      fNumTrainingBatchQueue = fTrainingBatchQueue.size();
      fNumValidationBatchQueue = fValidationBatchQueue.size();
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
   /// \param[in] chunkTensor RTensor with the data from the chunk
   /// \param[in] idxs Index of batch in the chunk
   /// \return Training batch
   std::unique_ptr<RFlat2DMatrix> CreateBatch(RFlat2DMatrix &chunTensor, std::size_t idxs)
   {
      auto batch = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
      std::copy(chunTensor.GetData() + (idxs * fBatchSize * fNumColumns),
                chunTensor.GetData() + ((idxs + 1) * fBatchSize * fNumColumns), batch->GetData());

      return batch;
   }

   /// \brief Loading the training batch from the queue
   /// \return Training batch
   RFlat2DMatrix GetTrainBatch()
   {

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<RFlat2DMatrix>();
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      return *fCurrentBatch;
   }

   /// \brief Loading the validation batch from the queue
   /// \return Validation batch
   RFlat2DMatrix GetValidationBatch()
   {

      if (fValidationBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<RFlat2DMatrix>();
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
   void
   CreateTrainingBatches(RFlat2DMatrix &chunkTensor, int lastbatch, std::size_t leftoverBatchSize, bool dropRemainder)
   {
      std::size_t ChunkSize = chunkTensor.GetRows();
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
      RFlat2DMatrix LeftoverBatch(LeftoverBatchSize, fNumColumns);
      std::copy(chunkTensor.GetData() + (Batches * fBatchSize * fNumColumns),
                chunkTensor.GetData() + (Batches * fBatchSize * fNumColumns + LeftoverBatchSize * fNumColumns),
                LeftoverBatch.GetData());

      // calculate how many empty slots are left in fPrimaryLeftoverTrainingBatch
      std::size_t PrimaryLeftoverSize = fPrimaryLeftoverTrainingBatch->GetRows();
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      // copy LeftoverBatch to end of fPrimaryLeftoverTrainingBatch
      if (emptySlots >= LeftoverBatchSize) {
         fPrimaryLeftoverTrainingBatch->Resize(PrimaryLeftoverSize + LeftoverBatchSize, fNumColumns);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fPrimaryLeftoverTrainingBatch->GetData() + (PrimaryLeftoverSize * fNumColumns));

         // copy LeftoverBatch to end of fPrimaryLeftoverTrainingBatch and add it to the batch vector
         if (emptySlots == LeftoverBatchSize) {
            auto copy = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverTrainingBatch->GetData(),
                      fPrimaryLeftoverTrainingBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));

            // reset fPrimaryLeftoverTrainingBatch and fSecondaryLeftoverTrainingBatch
            *fPrimaryLeftoverTrainingBatch = *fSecondaryLeftoverTrainingBatch;
            fSecondaryLeftoverTrainingBatch = std::make_unique<RFlat2DMatrix>();
         }
      }

      // copy LeftoverBatch to both fPrimaryLeftoverTrainingBatch and fSecondaryLeftoverTrainingBatch
      else if (emptySlots < LeftoverBatchSize) {
         // copy the first part of LeftoverBatch to end of fPrimaryLeftoverTrainingBatch
         fPrimaryLeftoverTrainingBatch->Resize(fBatchSize, fNumColumns);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * fNumColumns),
                   fPrimaryLeftoverTrainingBatch->GetData() + (PrimaryLeftoverSize * fNumColumns));

         // copy the last part of LeftoverBatch to the end of fSecondaryLeftoverTrainingBatch
         fSecondaryLeftoverTrainingBatch->Resize(LeftoverBatchSize - emptySlots, fNumColumns);
         std::copy(LeftoverBatch.GetData() + (emptySlots * fNumColumns),
                   LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fSecondaryLeftoverTrainingBatch->GetData());

         // add fPrimaryLeftoverTrainingBatch to the batch vector
         auto copy = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
         std::copy(fPrimaryLeftoverTrainingBatch->GetData(),
                   fPrimaryLeftoverTrainingBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
         batches.emplace_back(std::move(copy));

         // exchange fPrimaryLeftoverTrainingBatch and fSecondaryLeftoverTrainingBatch
         *fPrimaryLeftoverTrainingBatch = *fSecondaryLeftoverTrainingBatch;

         // reset fSecondaryLeftoverTrainingBatch
         fSecondaryLeftoverTrainingBatch = std::make_unique<RFlat2DMatrix>();
      }

      // copy the content of fPrimaryLeftoverTrainingBatch to the leftover batch from the chunk
      if (lastbatch == 1) {

         if (dropRemainder == false && leftoverBatchSize > 0) {
            auto copy = std::make_unique<RFlat2DMatrix>(leftoverBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverTrainingBatch->GetData(),
                      fPrimaryLeftoverTrainingBatch->GetData() + (leftoverBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
         }

         fPrimaryLeftoverTrainingBatch = std::make_unique<RFlat2DMatrix>();
         fSecondaryLeftoverTrainingBatch = std::make_unique<RFlat2DMatrix>();
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
   /// \param[in] dropRemainder Bool to drop the remainder batch or not
   void CreateValidationBatches(RFlat2DMatrix &chunkTensor, std::size_t lastbatch, std::size_t leftoverBatchSize,
                                bool dropRemainder)
   {
      std::size_t ChunkSize = chunkTensor.GetRows();
      std::size_t Batches = ChunkSize / fBatchSize;
      std::size_t LeftoverBatchSize = ChunkSize % fBatchSize;

      std::vector<std::unique_ptr<RFlat2DMatrix>> batches;

      for (std::size_t i = 0; i < Batches; i++) {
         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, i));
      }

      RFlat2DMatrix LeftoverBatch(LeftoverBatchSize, fNumColumns);
      std::copy(chunkTensor.GetData() + (Batches * fBatchSize * fNumColumns),
                chunkTensor.GetData() + (Batches * fBatchSize * fNumColumns + LeftoverBatchSize * fNumColumns),
                LeftoverBatch.GetData());

      std::size_t PrimaryLeftoverSize = fPrimaryLeftoverValidationBatch->GetRows();
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      if (emptySlots >= LeftoverBatchSize) {
         fPrimaryLeftoverValidationBatch->Resize(PrimaryLeftoverSize + LeftoverBatchSize, fNumColumns);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fPrimaryLeftoverValidationBatch->GetData() + (PrimaryLeftoverSize * fNumColumns));

         if (emptySlots == LeftoverBatchSize) {
            auto copy = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverValidationBatch->GetData(),
                      fPrimaryLeftoverValidationBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
            *fPrimaryLeftoverValidationBatch = *fSecondaryLeftoverValidationBatch;
            fSecondaryLeftoverValidationBatch = std::make_unique<RFlat2DMatrix>();
         }
      }

      else if (emptySlots < LeftoverBatchSize) {
         fPrimaryLeftoverValidationBatch->Resize(fBatchSize, fNumColumns);
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * fNumColumns),
                   fPrimaryLeftoverValidationBatch->GetData() + (PrimaryLeftoverSize * fNumColumns));
         fSecondaryLeftoverValidationBatch->Resize((LeftoverBatchSize - emptySlots), fNumColumns);
         std::copy(LeftoverBatch.GetData() + (emptySlots * fNumColumns),
                   LeftoverBatch.GetData() + (LeftoverBatchSize * fNumColumns),
                   fSecondaryLeftoverValidationBatch->GetData());
         auto copy = std::make_unique<RFlat2DMatrix>(fBatchSize, fNumColumns);
         std::copy(fPrimaryLeftoverValidationBatch->GetData(),
                   fPrimaryLeftoverValidationBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
         batches.emplace_back(std::move(copy));
         *fPrimaryLeftoverValidationBatch = *fSecondaryLeftoverValidationBatch;
         fSecondaryLeftoverValidationBatch = std::make_unique<RFlat2DMatrix>();
      }

      if (lastbatch == 1) {

         if (dropRemainder == false && leftoverBatchSize > 0) {
            auto copy = std::make_unique<RFlat2DMatrix>(leftoverBatchSize, fNumColumns);
            std::copy(fPrimaryLeftoverValidationBatch->GetData(),
                      fPrimaryLeftoverValidationBatch->GetData() + (leftoverBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
         }
         fPrimaryLeftoverValidationBatch = std::make_unique<RFlat2DMatrix>();
         fSecondaryLeftoverValidationBatch = std::make_unique<RFlat2DMatrix>();
      }

      for (std::size_t i = 0; i < batches.size(); i++) {
         fValidationBatchQueue.push(std::move(batches[i]));
      }
   }

   std::size_t GetNumTrainingBatchQueue() { return fTrainingBatchQueue.size(); }
   std::size_t GetNumValidationBatchQueue() { return fValidationBatchQueue.size(); }
};

} // namespace TMVA::Experimental::Internal

#endif // TMVA_RBATCHLOADER
