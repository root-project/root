#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/RChunkLoader.hxx"
#include "TMVA/RBatchLoader.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"
#include "TROOT.h"

namespace TMVA {
namespace Experimental {
namespace Internal {

template <typename... Args>
class RBatchGenerator {
private:
   TMVA::RandomGenerator<TRandom3> fRng = TMVA::RandomGenerator<TRandom3>(0);

   std::string fFileName;
   std::string fTreeName;

   std::vector<std::string> fCols;
   std::string fFilters;

   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fMaxBatches;
   std::size_t fNumColumns;
   std::size_t fNumEntries;
   std::size_t fCurrentRow = 0;

   float fValidationSplit;

   std::unique_ptr<TMVA::Experimental::Internal::RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<TMVA::Experimental::Internal::RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fUseWholeFile = true;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::vector<std::vector<std::size_t>> fTrainingIdxs;
   std::vector<std::vector<std::size_t>> fValidationIdxs;

   // filled batch elements
   std::mutex fIsActiveLock;

   bool fShuffle = true;
   bool fIsActive = false;

   std::vector<std::size_t> fVecSizes;
   float fVecPadding;

public:
   RBatchGenerator(const std::string &treeName, const std::string &fileName, const std::size_t chunkSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols, const std::string &filters = "",
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, const std::size_t numColumns = 0,
                   bool shuffle = true)
      : fTreeName(treeName),
        fFileName(fileName),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fCols(cols),
        fFilters(filters),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fNumColumns((numColumns != 0) ? numColumns : cols.size()),
        fShuffle(shuffle),
        fUseWholeFile(maxChunks == 0)
   {
      // limits the number of batches that can be contained in the batchqueue based on the chunksize
      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));

      // get the number of fNumEntries in the dataframe
      std::unique_ptr<TFile> f{TFile::Open(fFileName.c_str())};
      std::unique_ptr<TTree> t{f->Get<TTree>(fTreeName.c_str())};
      fNumEntries = t->GetEntries();

      fChunkLoader = std::make_unique<TMVA::Experimental::Internal::RChunkLoader<Args...>>(
         fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);
      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader>(fBatchSize, fNumColumns, fMaxBatches);

      // Create tensor to load the chunk into
      fChunkTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>((std::vector<std::size_t>){fChunkSize, fNumColumns});
   }

   ~RBatchGenerator() { DeActivate(); }

   /// \brief De-activate the loading process by deactivating the batchgenerator
   /// and joining the loading thread
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveLock);
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
         std::lock_guard<std::mutex> lock(fIsActiveLock);
         fIsActive = true;
      }

      fCurrentRow = 0;
      fBatchLoader->Activate();
      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   /// \brief Returns the next batch of training data if available.
   /// Returns empty RTensor otherwise.
   /// \return
   const TMVA::Experimental::RTensor<float> &GetTrainBatch()
   {
      // Get next batch if available
      return fBatchLoader->GetTrainBatch();
   }

   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
   /// \return
   const TMVA::Experimental::RTensor<float> &GetValidationBatch()
   {
      // Get next batch if available
      return fBatchLoader->GetValidationBatch();
   }

   bool HasTrainData() { return fBatchLoader->HasTrainData(); }

   bool HasValidationData() { return fBatchLoader->HasValidationData(); }

   void LoadChunks()
   {
      ROOT::EnableThreadSafety();

      for (std::size_t current_chunk = 0; ((current_chunk < fMaxChunks) || fUseWholeFile) && fCurrentRow < fNumEntries;
           current_chunk++) {

         // stop the loop when the loading is not active anymore
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         // A pair that consists the proccessed, and passed events while loading the chunk
         std::pair<std::size_t, std::size_t> report = fChunkLoader->LoadChunk(*fChunkTensor, fCurrentRow);
         fCurrentRow += report.first;

         CreateBatches(current_chunk, report.second);

         // Stop loading if the number of processed events is smaller than the desired chunk size
         if (report.first < fChunkSize) {
            break;
         }
      }

      fBatchLoader->DeActivate();
   }

   /// \brief Create batches for the current_chunk.
   /// \param currentChunk
   /// \param processedEvents
   void CreateBatches(std::size_t currentChunk, std::size_t processedEvents)
   {

      // Check if the indices in this chunk where already split in train and validations
      if (fTrainingIdxs.size() > currentChunk) {
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[currentChunk], fShuffle);
      } else {
         // Create the Validation batches if this is not the first epoch
         createIdxs(processedEvents);
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[currentChunk], fShuffle);
         fBatchLoader->CreateValidationBatches(*fChunkTensor, fValidationIdxs[currentChunk]);
      }
   }

   /// \brief plit the events of the current chunk into validation and training events
   /// \param processedEvents
   void createIdxs(std::size_t processedEvents)
   {
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(processedEvents);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      // calculate the number of events used for validation
      std::size_t num_validation = ceil(processedEvents * fValidationSplit);

      // Devide the vector into training and validation
      std::vector<std::size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
      std::vector<std::size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValidationIdxs.push_back(valid_idx);
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
   bool IsActive() { return fIsActive; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR
