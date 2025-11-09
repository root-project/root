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

#ifndef TMVA_RBATCHGENERATOR
#define TMVA_RBATCHGENERATOR

#include "TMVA/BatchGenerator/RFlat2DMatrix.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/BatchGenerator/RChunkLoader.hxx"
#include "TMVA/BatchGenerator/RBatchLoader.hxx"
#include "TROOT.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <variant>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RBatchGenerator
\ingroup tmva
\brief 

In this class, the processes of loading chunks (see RChunkLoader) and creating batches from those chunks (see RBatchLoader) are combined, allowing batches from the training and validation sets to be loaded directly from a dataset in an RDataFrame.
*/

template <typename... Args>
class RBatchGenerator {
private:
   std::vector<std::string> fCols;
   // clang-format on
   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fBlockSize;
   std::size_t fNumColumns;
   std::size_t fNumChunkCols;
   std::size_t fNumEntries;
   std::size_t fSetSeed;
   std::size_t fSumVecSizes;

   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> fEntries;
   float fValidationSplit;

   std::unique_ptr<RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   std::size_t fTrainingChunkNum;
   std::size_t fValidationChunkNum;

   ROOT::RDF::RNode &f_rdf;

   std::mutex fIsActiveMutex;

   bool fDropRemainder;
   bool fShuffle;
   bool fIsActive{false}; // Whether the loading thread is active
   bool fNotFiltered;
   bool fUseWholeFile;

   bool fEpochActive{false};
   bool fTrainingEpochActive{false};
   bool fValidationEpochActive{false};

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;

   std::size_t fNumTrainingChunks;
   std::size_t fNumValidationChunks;

   std::size_t fLeftoverTrainingBatchSize;
   std::size_t fLeftoverValidationBatchSize;

   std::size_t fNumFullTrainingBatches;
   std::size_t fNumFullValidationBatches;

   std::size_t fNumLeftoverTrainingBatches;
   std::size_t fNumLeftoverValidationBatches;

   std::size_t fNumTrainingBatches;
   std::size_t fNumValidationBatches;

   // flattened buffers for chunks and temporary tensors (rows * cols)
   RFlat2DMatrix fTrainTensor;
   RFlat2DMatrix fTrainChunkTensor;

   RFlat2DMatrix fValidationTensor;
   RFlat2DMatrix fValidationChunkTensor;

public:
   RBatchGenerator(ROOT::RDF::RNode &rdf, const std::size_t chunkSize, const std::size_t blockSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols,
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, bool shuffle = true,
                   bool dropRemainder = true, const std::size_t setSeed = 0)

      : f_rdf(rdf),
        fCols(cols),
        fChunkSize(chunkSize),
        fBlockSize(blockSize),
        fBatchSize(batchSize),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fDropRemainder(dropRemainder),
        fSetSeed(setSeed),
        fShuffle(shuffle),
        fNotFiltered(f_rdf.GetFilterNames().empty()),
        fUseWholeFile(maxChunks == 0),
        fNumColumns(cols.size())
   {

      fNumEntries = f_rdf.Count().GetValue();
      fEntries = f_rdf.Take<ULong64_t>("rdfentry_");

      fSumVecSizes = std::accumulate(vecSizes.begin(), vecSizes.end(), 0);
      fNumChunkCols = fNumColumns + fSumVecSizes - vecSizes.size();
      
      // add the last element in entries to not go out of range when filling chunks
      fEntries->push_back((*fEntries)[fNumEntries - 1] + 1);

      fChunkLoader =
         std::make_unique<RChunkLoader<Args...>>(f_rdf, fNumEntries, fEntries, fChunkSize, fBlockSize, fValidationSplit,
                                                 fCols, vecSizes, vecPadding, fShuffle, fSetSeed);
      fBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumChunkCols);

      // split the dataset into training and validation sets
      fChunkLoader->SplitDataset();

      // number of training and validation entries after the split
      fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
      fNumTrainingEntries = fNumEntries - fNumValidationEntries;
      
      fLeftoverTrainingBatchSize = fNumTrainingEntries % fBatchSize;
      fLeftoverValidationBatchSize = fNumValidationEntries % fBatchSize;

      fNumFullTrainingBatches = fNumTrainingEntries / fBatchSize;
      fNumFullValidationBatches = fNumValidationEntries / fBatchSize;

      fNumLeftoverTrainingBatches = fLeftoverTrainingBatchSize == 0 ? 0 : 1;
      fNumLeftoverValidationBatches = fLeftoverValidationBatchSize == 0 ? 0 : 1;

      if (dropRemainder) {
         fNumTrainingBatches = fNumFullTrainingBatches;
         fNumValidationBatches = fNumFullValidationBatches;
      }

      else {
         fNumTrainingBatches = fNumFullTrainingBatches + fNumLeftoverTrainingBatches;
         fNumValidationBatches = fNumFullValidationBatches + fNumLeftoverValidationBatches;
      }

      // number of training and validation chunks, calculated in RChunkConstructor
      fNumTrainingChunks = fChunkLoader->GetNumTrainingChunks();
      fNumValidationChunks = fChunkLoader->GetNumValidationChunks();

      fTrainingChunkNum = 0;
      fValidationChunkNum = 0;
   }

   ~RBatchGenerator() { DeActivate(); }

   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = false;
      }

      fBatchLoader->DeActivate();

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

      fBatchLoader->Activate();
      // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   void ActivateEpoch() { fEpochActive = true; }

   void DeActivateEpoch() { fEpochActive = false; }

   void ActivateTrainingEpoch() { fTrainingEpochActive = true; }

   void DeActivateTrainingEpoch() { fTrainingEpochActive = false; }

   void ActivateValidationEpoch() { fValidationEpochActive = true; }

   void DeActivateValidationEpoch() { fValidationEpochActive = false; }

   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see RBatchLoader)
   void CreateTrainBatches()
   {

      fChunkLoader->CreateTrainingChunksIntervals();
      fTrainingEpochActive = true;
      fTrainingChunkNum = 0;
      fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
      std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
      fBatchLoader->CreateTrainingBatches(fTrainChunkTensor, lastTrainingBatch, fLeftoverTrainingBatchSize,
                                          fDropRemainder);
      fTrainingChunkNum++;
   }

   /// \brief Creates validation batches by first loading a chunk (see RChunkLoader), and then split it into batches (see RBatchLoader)   
   void CreateValidationBatches()
   {

      fChunkLoader->CreateValidationChunksIntervals();
      fValidationEpochActive = true;
      fValidationChunkNum = 0;
      fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
      std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
      fBatchLoader->CreateValidationBatches(fValidationChunkTensor, lastValidationBatch, fLeftoverValidationBatchSize,
                                            fDropRemainder);
      fValidationChunkNum++;
   }

   /// \brief Loads a training batch from the queue
   RFlat2DMatrix GetTrainBatch()
   {
      auto batchQueue = fBatchLoader->GetNumTrainingBatchQueue();

      // load the next chunk if the queue is empty
      if (batchQueue < 1 && fTrainingChunkNum < fNumTrainingChunks) {
         fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
         std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
         fBatchLoader->CreateTrainingBatches(fTrainChunkTensor, lastTrainingBatch, fLeftoverTrainingBatchSize,
                                             fDropRemainder);
         fTrainingChunkNum++;
      }

      else {
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, 0, fNumEntries);
      }

      // Get next batch if available
      return fBatchLoader->GetTrainBatch();
   }

   /// \brief Loads a validation batch from the queue
   RFlat2DMatrix GetValidationBatch()
   {
      auto batchQueue = fBatchLoader->GetNumValidationBatchQueue();

      // load the next chunk if the queue is empty      
      if (batchQueue < 1 && fValidationChunkNum < fNumValidationChunks) {
         fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
         std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
         fBatchLoader->CreateValidationBatches(fValidationChunkTensor, lastValidationBatch,
                                               fLeftoverValidationBatchSize, fDropRemainder);
         fValidationChunkNum++;
      }

      else {
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, 0, fNumEntries);
      }

      // Get next batch if available
      return fBatchLoader->GetValidationBatch();
   }

   std::size_t NumberOfTrainingBatches() { return fNumTrainingBatches; }
   std::size_t NumberOfValidationBatches() { return fNumValidationBatches; }

   std::size_t TrainRemainderRows() { return fLeftoverTrainingBatchSize; }
   std::size_t ValidationRemainderRows() { return fLeftoverValidationBatchSize; }

   bool IsActive() { return fIsActive; }
   bool TrainingIsActive() { return fTrainingEpochActive; }
   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBATCHGENERATOR
