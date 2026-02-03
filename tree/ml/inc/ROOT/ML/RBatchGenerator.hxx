// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026

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

   std::vector<ROOT::RDF::RNode> f_rdfs;

   std::unique_ptr<std::thread> fLoadingThread;

   std::size_t fTrainingChunkNum;
   std::size_t fValidationChunkNum;

   std::mutex fIsActiveMutex;

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

   RFlat2DMatrix fTrainTensor;
   RFlat2DMatrix fTrainChunkTensor;

   RFlat2DMatrix fValidationTensor;
   RFlat2DMatrix fValidationChunkTensor;

public:
   RBatchGenerator(const std::vector<ROOT::RDF::RNode> &rdfs, const std::size_t chunkSize, const std::size_t blockSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols,
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, bool shuffle = true,
                   bool dropRemainder = true, const std::size_t setSeed = 0, bool loadEager = false,
                   std::string sampleType = "", float sampleRatio = 1.0, bool replacement = false)

      : f_rdfs(rdfs),
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
         fDatasetLoader = std::make_unique<RDatasetLoader<Args...>>(f_rdfs, fValidationSplit, fCols, fVecSizes,
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
         fChunkLoader = std::make_unique<RChunkLoader<Args...>>(f_rdfs[0], fChunkSize, fBlockSize, fValidationSplit,
                                                                fCols, fVecSizes, vecPadding, fShuffle, fSetSeed);

         // split the dataset into training and validation sets
         fChunkLoader->SplitDataset();

         fNumTrainingEntries = fChunkLoader->GetNumTrainingEntries();
         fNumValidationEntries = fChunkLoader->GetNumValidationEntries();

         // number of training and validation chunks, calculated in RChunkConstructor
         fNumTrainingChunks = fChunkLoader->GetNumTrainingChunks();
         fNumValidationChunks = fChunkLoader->GetNumValidationChunks();
      }

      fTrainingBatchLoader =
         std::make_unique<RBatchLoader>(fBatchSize, fCols, fVecSizes, fNumTrainingEntries, fDropRemainder);
      fValidationBatchLoader =
         std::make_unique<RBatchLoader>(fBatchSize, fCols, fVecSizes, fNumValidationEntries, fDropRemainder);
   }

   ~RBatchGenerator() { DeActivate(); }

   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = false;
      }

      fTrainingBatchLoader->DeActivate();
      fValidationBatchLoader->DeActivate();

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            fLoadingThread->join();
         }
      }
   }

   /// \brief Activate the loading process by starting the batchloader, and
   /// spawning the loading thread.
   void Activate()
   {
      if (fIsActive)
         return;

      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = true;
      }

      fTrainingBatchLoader->Activate();
      fValidationBatchLoader->Activate();
      // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   void ActivateEpoch() { fEpochActive = true; }

   void DeActivateEpoch() { fEpochActive = false; }

   void ActivateTrainingEpoch() { fTrainingEpochActive = true; }

   void DeActivateTrainingEpoch() { fTrainingEpochActive = false; }

   void ActivateValidationEpoch() { fValidationEpochActive = true; }

   void DeActivateValidationEpoch() { fValidationEpochActive = false; }

   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see
   /// RBatchLoader)
   void CreateTrainBatches()
   {
      fTrainingEpochActive = true;
      if (fLoadEager) {
         if (fSampleType == "") {
            fTensorOperators->ShuffleTensor(fSampledTrainingDataset, fTrainingDataset);
         }

         else {
            fTrainingSampler->Sampler(fSampledTrainingDataset);
         }

         fTrainingBatchLoader->CreateBatches(fSampledTrainingDataset, 1);
      }

      else {
         fChunkLoader->CreateTrainingChunksIntervals();
         fTrainingChunkNum = 0;
         fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
         fTrainingBatchLoader->CreateBatches(fTrainChunkTensor, fNumTrainingChunks);
         fTrainingChunkNum++;
      }
   }

   /// \brief Creates validation batches by first loading a chunk (see RChunkLoader), and then split it into batches
   /// (see RBatchLoader)
   void CreateValidationBatches()
   {
      fValidationEpochActive = true;
      if (fLoadEager) {
         if (fSampleType == "") {
            fTensorOperators->ShuffleTensor(fSampledValidationDataset, fValidationDataset);
         }

         else {
            fValidationSampler->Sampler(fSampledValidationDataset);
         }

         fValidationBatchLoader->CreateBatches(fSampledValidationDataset, 1);
      }

      else {
         fChunkLoader->CreateValidationChunksIntervals();
         fValidationChunkNum = 0;
         fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
         fValidationBatchLoader->CreateBatches(fValidationChunkTensor, fNumValidationChunks);
         fValidationChunkNum++;
      }
   }

   /// \brief Loads a training batch from the queue
   RFlat2DMatrix GetTrainBatch()
   {
      if (!fLoadEager) {
         auto batchQueue = fTrainingBatchLoader->GetNumBatchQueue();

         // load the next chunk if the queue is empty
         if (batchQueue < 1 && fTrainingChunkNum < fNumTrainingChunks) {
            fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
            std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
            fTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch);
            fTrainingChunkNum++;
         }
      }
      // Get next batch if available
      return fTrainingBatchLoader->GetBatch();
   }

   /// \brief Loads a validation batch from the queue
   RFlat2DMatrix GetValidationBatch()
   {
      if (!fLoadEager) {
         auto batchQueue = fValidationBatchLoader->GetNumBatchQueue();

         // load the next chunk if the queue is empty
         if (batchQueue < 1 && fValidationChunkNum < fNumValidationChunks) {
            fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
            std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
            fValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch);
            fValidationChunkNum++;
         }
      }
      // Get next batch if available
      return fValidationBatchLoader->GetBatch();
   }

   std::size_t NumberOfTrainingBatches() { return fTrainingBatchLoader->GetNumBatches(); }
   std::size_t NumberOfValidationBatches() { return fValidationBatchLoader->GetNumBatches(); }

   std::size_t TrainRemainderRows() { return fTrainingBatchLoader->GetNumRemainderRows(); }
   std::size_t ValidationRemainderRows() { return fValidationBatchLoader->GetNumRemainderRows(); }

   bool IsActive() { return fIsActive; }
   bool TrainingIsActive() { return fTrainingEpochActive; }
   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
};

} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RBATCHGENERATOR
