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

#ifndef ROOT_INTERNAL_ML_RDATALOADERENGINE
#define ROOT_INTERNAL_ML_RDATALOADERENGINE

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ROOT/ML/RBatchLoader.hxx"
#include "ROOT/ML/RClusterLoader.hxx"
#include "ROOT/ML/RDatasetLoader.hxx"
#include "ROOT/ML/RFlat2DMatrix.hxx"
#include "ROOT/ML/RFlat2DMatrixOperators.hxx"
#include "ROOT/ML/RSampler.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"

// Empty namespace to create a hook for the Pythonization
namespace ROOT::Experimental::ML {
}

namespace ROOT::Experimental::Internal::ML {
/**
 \class ROOT::Experimental::Internal::ML::RDataLoaderEngine
\brief

In this class, the processes of loading clusters (see RClusterLoader) and creating batches from those clusters (see
RBatchLoader) are combined, allowing batches from the training and validation sets to be loaded directly from a dataset
in an RDataFrame.
*/

template <typename... Args>
class RDataLoaderEngine {
private:
   std::vector<std::string> fCols;
   std::vector<std::size_t> fVecSizes;
   std::size_t fBatchSize;
   std::size_t fSetSeed;

   // buffer quantities
   std::size_t fBatchesInMemory;
   std::size_t fBufferCapacity;
   std::size_t fLowWatermark;
   std::size_t fHighWatermark;

   std::size_t fTrainingClusterIdx{0};
   std::size_t fValidationClusterIdx{0};

   float fTestSize;

   std::unique_ptr<RDatasetLoader<Args...>> fDatasetLoader;
   std::unique_ptr<RClusterLoader<Args...>> fClusterLoader;
   std::unique_ptr<RBatchLoader> fTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fValidationBatchLoader;
   std::unique_ptr<RSampler> fTrainingSampler;
   std::unique_ptr<RSampler> fValidationSampler;

   std::unique_ptr<RFlat2DMatrixOperators> fTensorOperators;

   std::vector<ROOT::RDF::RNode> fRdfs;

   std::unique_ptr<std::thread> fLoadingThread;
   std::condition_variable fLoadingCondition;
   std::mutex fLoadingMutex;

   bool fDropRemainder;
   bool fShuffle;
   bool fLoadEager;
   std::string fSampleType;
   float fSampleRatio;
   bool fReplacement;

   bool fIsActive{false}; // Whether the loading thread is active

   bool fEpochActive{false};
   bool fTrainingEpochActive{false};
   bool fValidationEpochActive{false};

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;

   // flattened buffers for chunks and temporary tensors (rows * cols)
   std::vector<RFlat2DMatrix> fTrainingDatasets;
   std::vector<RFlat2DMatrix> fValidationDatasets;

   RFlat2DMatrix fTrainingDataset;
   RFlat2DMatrix fValidationDataset;

   RFlat2DMatrix fSampledTrainingDataset;
   RFlat2DMatrix fSampledValidationDataset;

   std::size_t fTrainingEpochCount{0};
   std::size_t fValidationEpochCount{0};

public:
   RDataLoaderEngine(const std::vector<ROOT::RDF::RNode> &rdfs, const std::size_t batchSize,
                     const std::size_t batchesInMemory, const std::vector<std::string> &cols,
                     const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                     const float testSize = 0.0, bool shuffle = true, bool dropRemainder = true,
                     const std::size_t setSeed = 0, bool loadEager = false, std::string sampleType = "",
                     float sampleRatio = 1.0, bool replacement = false)
      : fRdfs(rdfs),
        fCols(cols),
        fVecSizes(vecSizes),
        fBatchSize(batchSize),
        fBatchesInMemory(batchesInMemory),
        fTestSize(testSize),
        fDropRemainder(dropRemainder),
        fSetSeed(setSeed),
        fShuffle(shuffle),
        fLoadEager(loadEager),
        fSampleType(sampleType),
        fSampleRatio(sampleRatio),
        fReplacement(replacement)
   {
      fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);

      if (fLoadEager) {
         fDatasetLoader = std::make_unique<RDatasetLoader<Args...>>(fRdfs, fTestSize, fCols, fVecSizes, vecPadding,
                                                                    fShuffle, fSetSeed);
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
         // scan cluster boundaries
         fClusterLoader = std::make_unique<RClusterLoader<Args...>>(fRdfs, fCols, fVecSizes, vecPadding, fTestSize,
                                                                    fShuffle, fSetSeed);

         // derive buffer quantities
         fBufferCapacity = fBatchSize * fBatchesInMemory;
         fLowWatermark = fBufferCapacity / 2;
         fHighWatermark = fBufferCapacity;

         // split cluster list into training and validation
         fClusterLoader->SplitDataset();
         fNumTrainingEntries = fClusterLoader->GetNumTrainingEntries();
         fNumValidationEntries = fClusterLoader->GetNumValidationEntries();
      }

      fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fCols, fLoadingMutex, fLoadingCondition,
                                                            fVecSizes, fNumTrainingEntries, fDropRemainder);
      fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fCols, fLoadingMutex, fLoadingCondition,
                                                              fVecSizes, fNumValidationEntries, fDropRemainder);
   }

   ~RDataLoaderEngine() { DeActivate(); }

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

   /// \brief Activate the loading process by spawning the loading thread.
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

      fLoadingThread = std::make_unique<std::thread>(&RDataLoaderEngine::LoadData, this);
   }

   /// \brief Activate the training epoch by starting the batchloader.
   void ActivateTrainingEpoch()
   {
      {
         std::lock_guard<std::mutex> lock(fLoadingMutex);
         fTrainingEpochActive = true;
         fTrainingClusterIdx = 0;
         if (!fLoadEager) {
            // Shuffle the cluster indices at the beginning of each epoch
            fClusterLoader->ShuffleTrainingClusters(fTrainingEpochCount++);
         }
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
         fValidationClusterIdx = 0;
         if (!fLoadEager) {
            fClusterLoader->ShuffleValidationClusters(fValidationEpochCount++);
         }
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

   /// \brief Main loop for loading clusters and creating batches.
   /// The producer (loading thread) will keep loading clusters and creating batches until the end of the epoch is
   /// reached, or the generator is deactivated.
   void LoadData()
   {
      std::unique_lock<std::mutex> lock(fLoadingMutex);

      while (true) {
         // Wait until we have work or shutdown
         fLoadingCondition.wait(lock, [&] {
            return !fIsActive ||
                   (fTrainingEpochActive && fTrainingClusterIdx < fClusterLoader->GetNumTrainingClusters()) ||
                   (fValidationEpochActive && fValidationClusterIdx < fClusterLoader->GetNumValidationClusters());
         });

         if (!fIsActive) {
            break;
         }

         const std::size_t numTrainingClusters = fClusterLoader->GetNumTrainingClusters();
         const std::size_t numValidationClusters = fClusterLoader->GetNumValidationClusters();

         // Helper: check if validation queue below watermark and needs the producer
         auto validationEmpty = [&] {
            if (!fValidationEpochActive || fValidationClusterIdx >= numValidationClusters)
               return false;
            if (fValidationBatchLoader->isProducerDone())
               return false;
            return fValidationBatchLoader->GetNumBatchQueue() < fLowWatermark / fBatchSize;
         };

         // -- TRAINING --
         if (fTrainingEpochActive) {
            while (true) {
               // Stop conditions (shutdown or epoch end)
               if (!fIsActive || !fTrainingEpochActive)
                  break;

               // No more chunks to load: signal consumers
               if (fTrainingClusterIdx >= numTrainingClusters) {
                  fTrainingBatchLoader->MarkProducerDone();
                  break;
               }

               // In the case of training prefetching, we could start requesting data for the next training loop while
               // validation is active and might need data. To avoid getting stuck in the training loop, we check if the
               // validation queue is below watermark and if so, we break out of the training loop.
               if (validationEmpty()) {
                  break;
               }

               // If queue is not empty, wait until it drains below watermark, or validation needs data, or we are
               // deactivated.
               if (fTrainingBatchLoader->GetNumBatchQueue() >= fLowWatermark / fBatchSize) {
                  fLoadingCondition.wait(lock, [&] {
                     return !fIsActive || !fTrainingEpochActive ||
                            fTrainingBatchLoader->GetNumBatchQueue() < (fLowWatermark / fBatchSize) ||
                            validationEmpty();
                  });
                  continue;
               }

               // Accumulate clusters to load, enough to fill the buffer, or until we run out of clusters
               std::vector<RClusterRange> trainClustersToLoad;
               auto accumulatedEntries = 0;
               const bool discovering = !fClusterLoader->IsSplitDiscovered();
               while (fTrainingClusterIdx < numTrainingClusters && accumulatedEntries < fBufferCapacity &&
                      (!discovering || trainClustersToLoad.empty())) {
                  const auto &cluster = fClusterLoader->GetTrainingClusters()[fTrainingClusterIdx++];
                  trainClustersToLoad.push_back(cluster);
                  accumulatedEntries += cluster.GetNumEntries();
               }

               const bool isLastBuffer = (fTrainingClusterIdx >= numTrainingClusters);

               // Release lock while reading and loading data to allow the consumer to access the queue freely in
               // parallel. The loading thread re-acquires the lock in CreateBatches when it needs to push batches to
               // the queue.
               lock.unlock();
               RFlat2DMatrix stagingBuffer(accumulatedEntries, fClusterLoader->GetNumChunkCols());
               std::size_t rowOffset = 0;

               for (auto &cluster : trainClustersToLoad) {
                  auto loadedEntries = fClusterLoader->LoadTrainingClusterInto(stagingBuffer, cluster.rdfIdx,
                                                                               cluster.start, cluster.end, rowOffset);
                  if (discovering) {
                     // For the first epoch, we might discover that the cluster has fewer entries than expected because
                     // of filters
                     cluster.SetNumEntries(loadedEntries);
                  }
                  rowOffset += cluster.GetNumEntries();
               }

               if (discovering && fNumTrainingEntries == 0 && fClusterLoader->GetNumTrainingEntries() > 0) {
                  fNumTrainingEntries = fClusterLoader->GetNumTrainingEntries();
                  fNumValidationEntries = fClusterLoader->GetNumValidationEntries();
                  fTrainingBatchLoader->RecalculateBatchCounts(fNumTrainingEntries);
                  fValidationBatchLoader->RecalculateBatchCounts(fNumValidationEntries);
               }

               if (rowOffset < static_cast<std::size_t>(accumulatedEntries)) {
                  stagingBuffer.Resize(rowOffset, stagingBuffer.GetCols());
               }

               RFlat2DMatrix shuffledStagingBuffer;
               fTensorOperators->ShuffleTensor(shuffledStagingBuffer, stagingBuffer);
               fTrainingBatchLoader->CreateBatches(shuffledStagingBuffer, isLastBuffer);

               // Re-acquire the lock before the next iteration to check conditions and update indices
               lock.lock();

               if (isLastBuffer && discovering) {
                  fClusterLoader->FinaliseSplitDiscovery();
               }
            }
         }

         // -- VALIDATION --
         if (fValidationEpochActive) {
            while (true) {
               // Stop conditions (shutdown or epoch end)
               if (!fIsActive || !fValidationEpochActive)
                  break;

               // No more chunks to load: signal consumers
               if (fValidationClusterIdx >= numValidationClusters) {
                  fValidationBatchLoader->MarkProducerDone();
                  break;
               }

               // If queue is not hungry, wait until it drains below watermark, or we are deactivated
               if (fValidationBatchLoader->GetNumBatchQueue() >= (fLowWatermark / fBatchSize)) {
                  fLoadingCondition.wait(lock, [&] {
                     return !fIsActive || !fValidationEpochActive ||
                            fValidationBatchLoader->GetNumBatchQueue() < (fLowWatermark / fBatchSize);
                  });
                  continue;
               }

               // Accumulate clusters to load, enough to fill the buffer, or until we run out of clusters
               std::vector<RClusterRange> valClustersToLoad;
               auto accumulatedEntries = 0;
               while (fValidationClusterIdx < numValidationClusters && accumulatedEntries < fBufferCapacity) {
                  const auto &cluster = fClusterLoader->GetValidationClusters()[fValidationClusterIdx++];
                  valClustersToLoad.push_back(cluster);
                  accumulatedEntries += cluster.GetNumEntries();
               }

               const bool isLastBuffer = (fValidationClusterIdx >= numValidationClusters);

               lock.unlock();

               RFlat2DMatrix stagingBuffer(accumulatedEntries, fClusterLoader->GetNumChunkCols());
               std::size_t rowOffset = 0;

               for (const auto &cluster : valClustersToLoad) {
                  fClusterLoader->LoadValidationClusterInto(stagingBuffer, cluster.rdfIdx, cluster.start, cluster.end,
                                                            rowOffset);
                  rowOffset += cluster.GetNumEntries();
               }

               RFlat2DMatrix shuffledStagingBuffer;
               fTensorOperators->ShuffleTensor(shuffledStagingBuffer, stagingBuffer);
               fValidationBatchLoader->CreateBatches(shuffledStagingBuffer, isLastBuffer);

               lock.lock();
            }
         }
      }
   }

   /// \brief Create training batches by first loading a chunk (see RClusterLoader) and split it into batches (see
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
      }
   }

   /// \brief Creates validation batches by first loading a chunk (see RClusterLoader), and then split it into batches
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

#endif // ROOT_INTERNAL_ML_RDATALOADERENGINE