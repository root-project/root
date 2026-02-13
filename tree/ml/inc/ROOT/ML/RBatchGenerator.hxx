// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026
// Author: Silia Taider, CERN 02/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RBATCHGENERATOR
#define ROOT_INTERNAL_ML_RBATCHGENERATOR

#include "ROOT/ML/RFlat2DMatrix.hxx"
#include "ROOT/ML/RFlat2DMatrixOperators.hxx"
#include "ROOT/ML/RSampler.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"

#include "ROOT/ML/RDatasetLoader.hxx"
#include "ROOT/ML/RChunkLoader.hxx"
#include "ROOT/ML/RBatchLoader.hxx"
#include "TROOT.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <variant>
#include <vector>

// Empty namespace to create a hook for the Pythonization
namespace ROOT::Experimental::ML {
}

namespace ROOT::Experimental::Internal::ML {
/**
\class ROOT::Experimental::Internal::ML::RBatchGenerator
\brief

In this class, the processes of loading chunks (see RChunkLoader) and creating batches from those chunks (see
RBatchLoader) are combined, allowing batches from the training and validation sets to be loaded directly from a dataset
in an RDataFrame.
*/

template <typename... Args>
class RBatchGenerator {
private:
   std::vector<std::string> fCols;
   std::vector<std::size_t> fVecSizes;
   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fBlockSize;
   std::size_t fSetSeed;

   float fValidationSplit;

   std::unique_ptr<RDatasetLoader<Args...>> fDatasetLoader;
   std::unique_ptr<RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<RBatchLoader> fTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fValidationBatchLoader;
   std::unique_ptr<RSampler> fTrainingSampler;
   std::unique_ptr<RSampler> fValidationSampler;

   std::unique_ptr<RFlat2DMatrixOperators> fTensorOperators;

   std::vector<ROOT::RDF::RNode> fRdfs;

   std::unique_ptr<std::thread> fLoadingThread;
   std::condition_variable fLoadingCondition;
   std::mutex fLoadingMutex;

   std::size_t fTrainingChunkNum{0};
   std::size_t fValidationChunkNum{0};

   bool fDropRemainder;
   bool fShuffle;
   bool fLoadEager;
   std::string fSampleType;
   float fSampleRatio;
   bool fReplacement;

   bool fIsActive{false}; // Whether the loading thread is active
   bool fUseWholeFile;

   bool fEpochActive{false};
   bool fTrainingEpochActive{false};
   bool fValidationEpochActive{false};

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;

   std::size_t fNumTrainingChunks;
   std::size_t fNumValidationChunks;

   // flattened buffers for chunks and temporary tensors (rows * cols)
   std::vector<RFlat2DMatrix> fTrainingDatasets;
   std::vector<RFlat2DMatrix> fValidationDatasets;

   RFlat2DMatrix fTrainingDataset;
   RFlat2DMatrix fValidationDataset;

   RFlat2DMatrix fSampledTrainingDataset;
   RFlat2DMatrix fSampledValidationDataset;

   RFlat2DMatrix fTrainChunkTensor;

   RFlat2DMatrix fValidationChunkTensor;

public:
   RBatchGenerator(const std::vector<ROOT::RDF::RNode> &rdfs, const std::size_t chunkSize, const std::size_t blockSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols,
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, bool shuffle = true,
                   bool dropRemainder = true, const std::size_t setSeed = 0, bool loadEager = false,
                   std::string sampleType = "", float sampleRatio = 1.0, bool replacement = false)

      : fRdfs(rdfs),
        fCols(cols),
        fVecSizes(vecSizes),
        fChunkSize(chunkSize),
        fBlockSize(blockSize),
        fBatchSize(batchSize),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fDropRemainder(dropRemainder),
        fSetSeed(setSeed),
        fShuffle(shuffle),
        fLoadEager(loadEager),
        fSampleType(sampleType),
        fSampleRatio(sampleRatio),
        fReplacement(replacement),
        fUseWholeFile(maxChunks == 0)
   {
      fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);

      if (fLoadEager) {
         fDatasetLoader = std::make_unique<RDatasetLoader<Args...>>(fRdfs, fValidationSplit, fCols, fVecSizes,
                                                                    vecPadding, fShuffle, fSetSeed);
         // split the datasets and extract the training and validation datasets
         fDatasetLoader->SplitDatasets();

         if (fSampleType == "") {
            fDatasetLoader->ConcatenateDatasets();

            fTrainingDataset = fDatasetLoader->GetTrainingDataset();
            fValidationDataset = fDatasetLoader->GetValidationDataset();

            fNumTrainingEntries = fDatasetLoader->GetNumTrainingEntries();
            fNumValidationEntries = fDatasetLoader->GetNumValidationEntries();
         }

         else {
            fTrainingDatasets = fDatasetLoader->GetTrainingDatasets();
            fValidationDatasets = fDatasetLoader->GetValidationDatasets();

            fTrainingSampler = std::make_unique<RSampler>(fTrainingDatasets, fSampleType, fSampleRatio, fReplacement,
                                                          fShuffle, fSetSeed);
            fValidationSampler = std::make_unique<RSampler>(fValidationDatasets, fSampleType, fSampleRatio,
                                                            fReplacement, fShuffle, fSetSeed);

            fNumTrainingEntries = fTrainingSampler->GetNumEntries();
            fNumValidationEntries = fValidationSampler->GetNumEntries();
         }
      }

      else {
         fChunkLoader = std::make_unique<RChunkLoader<Args...>>(fRdfs[0], fChunkSize, fBlockSize, fValidationSplit,
                                                                fCols, fVecSizes, vecPadding, fShuffle, fSetSeed);

         // split the dataset into training and validation sets
         fChunkLoader->SplitDataset();

         fNumTrainingEntries = fChunkLoader->GetNumTrainingEntries();
         fNumValidationEntries = fChunkLoader->GetNumValidationEntries();

         // number of training and validation chunks, calculated in RChunkConstructor
         fNumTrainingChunks = fChunkLoader->GetNumTrainingChunks();
         fNumValidationChunks = fChunkLoader->GetNumValidationChunks();
      }

      fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fCols, fLoadingMutex, fLoadingCondition,
                                                            fVecSizes, fNumTrainingEntries, fDropRemainder);
      fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fCols, fLoadingMutex, fLoadingCondition,
                                                              fVecSizes, fNumValidationEntries, fDropRemainder);
   }

   ~RBatchGenerator() { DeActivate(); }

   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         if (!fIsActive)
            return;
         fIsActive = false;
      }

      fLoadingCondition.notify_all();

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            fLoadingThread->join();
         }
      }

      fLoadingThread.reset();
   }

   /// \brief Activate the loading process by starting the batchloader, and
   /// spawning the loading thread.
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         if (fIsActive)
            return;

         fIsActive = true;
      }

      if (fLoadEager) {
         return;
      }

      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   void ActivateEpoch()
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      fEpochActive = true;
   }

   void DeActivateEpoch()
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      fEpochActive = false;
   }

   void ActivateTrainingEpoch()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         fTrainingEpochActive = true;
         fTrainingChunkNum = 0;
      }

      fTrainingBatchLoader->Activate();
      fLoadingCondition.notify_all();
   }

   void DeActivateTrainingEpoch()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         fTrainingEpochActive = false;
      }

      fTrainingBatchLoader->Reset();
      fTrainingBatchLoader->DeActivate();
      fLoadingCondition.notify_all();
   }

   void ActivateValidationEpoch()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         fValidationEpochActive = true;
         fValidationChunkNum = 0;
      }

      fValidationBatchLoader->Activate();
      fLoadingCondition.notify_all();
   }

   void DeActivateValidationEpoch()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         fValidationEpochActive = false;
      }

      fValidationBatchLoader->Reset();
      fValidationBatchLoader->DeActivate();
      fLoadingCondition.notify_all();
   }

   void LoadChunks()
   {
      // Set minimum number of batches to keep in the queue before producer goes to work.
      // This is to ensure that the producer will get a chance to work if the consumer is too fast and drains the queue
      // quickly. With this, the maximum queue size will be approximately fChunkSize*1.5.
      // TODO(staider): this is a heuristic that can be improved, by taking into consideration a "maximum number of
      // batches in memory" set by the user.
      const std::size_t kMinQueuedBatches = std::max<std::size_t>(1, (fChunkSize / fBatchSize) / 2);

      std::unique_lock<std::mutex> lock(fLoadingMutex);

      while (true) {
         // Wait until we have work or shutdown
         fLoadingCondition.wait(lock, [&] {
            return !fIsActive || (fTrainingEpochActive && fTrainingChunkNum < fNumTrainingChunks) ||
                   (fValidationEpochActive && fValidationChunkNum < fNumValidationChunks);
         });

         if (!fIsActive)
            break;

         // Validation queue below watermark and producer not done
         auto validationEmpty = [&] {
            if (!fValidationEpochActive || fValidationChunkNum >= fNumValidationChunks)
               return false;
            if (fValidationBatchLoader->isProducerDone())
               return false;
            return fValidationBatchLoader->GetNumBatchQueue() < kMinQueuedBatches;
         };

         // TRAINING
         if (fTrainingEpochActive) {
            while (true) {
               // Stop conditions (shutdown or epoch end)
               if (!fIsActive || !fTrainingEpochActive)
                  break;

               // End-of-epoch: signal consumers
               if (fTrainingChunkNum >= fNumTrainingChunks) {
                  fTrainingBatchLoader->MarkProducerDone();
                  break;
               }

               // Give validation a chance if it is hungry
               if (validationEmpty())
                  break;

               // If queue is not empty, wait until it drains below watermark, or validation needs data, or we are
               // deactivated The extra validation check is for the case of prefetching, where we request data for the
               // next training epoch while validation is active and might need data
               if (fTrainingBatchLoader->GetNumBatchQueue() >= kMinQueuedBatches) {
                  fLoadingCondition.wait(lock, [&] {
                     return !fIsActive || !fTrainingEpochActive ||
                            fTrainingBatchLoader->GetNumBatchQueue() < kMinQueuedBatches || validationEmpty();
                  });
                  continue;
               }

               // Claim chunk under lock
               const std::size_t chunkIdx = fTrainingChunkNum++;
               const bool isLastTrainChunk = (chunkIdx == fNumTrainingChunks - 1);

               // Release lock while working
               lock.unlock();
               fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, chunkIdx);
               fTrainingBatchLoader->CreateBatches(fTrainChunkTensor, isLastTrainChunk);
               lock.lock();
            }
         }

         // VALIDATION
         if (fValidationEpochActive) {
            while (true) {
               // Stop conditions (shutdown or epoch end)
               if (!fIsActive || !fValidationEpochActive)
                  break;

               // End-of-epoch: signal consumers
               if (fValidationChunkNum >= fNumValidationChunks) {
                  fValidationBatchLoader->MarkProducerDone();
                  break;
               }

               // If queue is not hungry, wait until it drains below watermark, or we are deactivated
               if (fValidationBatchLoader->GetNumBatchQueue() >= kMinQueuedBatches) {
                  fLoadingCondition.wait(lock, [&] {
                     return !fIsActive || !fValidationEpochActive ||
                            fValidationBatchLoader->GetNumBatchQueue() < kMinQueuedBatches;
                  });
                  continue;
               }

               // Claim chunk under lock
               const std::size_t chunkIdx = fValidationChunkNum++;
               const bool isLastValidationChunk = (chunkIdx == fNumValidationChunks - 1);

               // Release lock while working
               lock.unlock();
               fChunkLoader->LoadValidationChunk(fValidationChunkTensor, chunkIdx);
               fValidationBatchLoader->CreateBatches(fValidationChunkTensor, isLastValidationChunk);
               lock.lock();
            }
         }
      }
   }

   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see
   /// RBatchLoader)
   void CreateTrainBatches()
   {
      fTrainingBatchLoader->Activate();

      if (fLoadEager) {
         if (fSampleType == "") {
            fTensorOperators->ShuffleTensor(fSampledTrainingDataset, fTrainingDataset);
         }

         else {
            fTrainingSampler->Sampler(fSampledTrainingDataset);
         }

         fTrainingBatchLoader->CreateBatches(fSampledTrainingDataset, true);
         fTrainingBatchLoader->MarkProducerDone();
      } else {
         fChunkLoader->CreateTrainingChunksIntervals();
      }
   }

   /// \brief Creates validation batches by first loading a chunk (see RChunkLoader), and then split it into batches
   /// (see RBatchLoader)
   void CreateValidationBatches()
   {
      fValidationBatchLoader->Activate();

      if (fLoadEager) {
         if (fSampleType == "") {
            fTensorOperators->ShuffleTensor(fSampledValidationDataset, fValidationDataset);
         }

         else {
            fValidationSampler->Sampler(fSampledValidationDataset);
         }

         fValidationBatchLoader->CreateBatches(fSampledValidationDataset, true);
         fValidationBatchLoader->MarkProducerDone();
      }

      else {
         fChunkLoader->CreateValidationChunksIntervals();
      }
   }

   /// \brief Loads a training batch from the queue
   RFlat2DMatrix GetTrainBatch()
   {
      // Get next batch if available
      return fTrainingBatchLoader->GetBatch();
   }

   /// \brief Loads a validation batch from the queue
   RFlat2DMatrix GetValidationBatch()
   {
      // Get next batch if available
      return fValidationBatchLoader->GetBatch();
   }

   std::size_t NumberOfTrainingBatches() { return fTrainingBatchLoader->GetNumBatches(); }
   std::size_t NumberOfValidationBatches() { return fValidationBatchLoader->GetNumBatches(); }

   std::size_t TrainRemainderRows() { return fTrainingBatchLoader->GetNumRemainderRows(); }
   std::size_t ValidationRemainderRows() { return fValidationBatchLoader->GetNumRemainderRows(); }

   bool IsActive()
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      return fIsActive;
   }

   bool IsTrainingActive()
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      return fTrainingEpochActive;
   }

   bool IsValidationActive()
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      return fValidationEpochActive;
   }
};

} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RBATCHGENERATOR
