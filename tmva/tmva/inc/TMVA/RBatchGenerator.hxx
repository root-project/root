#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>
#include <iostream>

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

   std::vector<std::string> fCols;
   std::string fFilters;

   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fMaxBatches;
   std::size_t fNumColumns;
   std::size_t fNumEntries;
   std::size_t fCurrentRow = 0;
   std::size_t fTrainingRemainderRow = 0;
   std::size_t fValidationRemainderRow = 0;

   float fValidationSplit;

   std::unique_ptr<TMVA::Experimental::Internal::RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<TMVA::Experimental::Internal::RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fUseWholeFile = true;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fTrainingRemainder;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainder;
   
   ROOT::RDF::RNode &f_rdf;

   std::vector<std::vector<std::size_t>> fTrainingIdxs;
   std::vector<std::vector<std::size_t>> fValidationIdxs;

   // filled batch elements
   std::mutex fIsActiveLock;

   bool fDropRemainder = true;
   bool fShuffle = true;
   bool fIsActive = false;

   std::vector<std::size_t> fVecSizes;
   float fVecPadding;

public:
   RBatchGenerator(ROOT::RDF::RNode &rdf, const std::size_t chunkSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols, const std::string &filters = "",
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, const std::size_t numColumns = 0,
                   bool shuffle = true, bool dropRemainder = true)
      : f_rdf(rdf),
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
        fDropRemainder(dropRemainder),
        fUseWholeFile(maxChunks == 0)
   {
      // limits the number of batches that can be contained in the batchqueue based on the chunksize
      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));

      fNumEntries = f_rdf.Count().GetValue();

      fChunkLoader = std::make_unique<TMVA::Experimental::Internal::RChunkLoader<Args...>>(
         f_rdf, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);
      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader>(fBatchSize, fNumColumns, fMaxBatches);

      // Create tensor to load the chunk into
      fChunkTensor =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fChunkSize, fNumColumns});
      // Create remainders tensors
      fTrainingRemainder =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
      fValidationRemainder =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize - 1, fNumColumns});
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

      if (!fDropRemainder){
         fBatchLoader->LastBatches(*fTrainingRemainder, fTrainingRemainderRow, *fValidationRemainder, fValidationRemainderRow);
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
         fTrainingRemainderRow = fBatchLoader->CreateTrainingBatches(*fChunkTensor, *fTrainingRemainder, fTrainingRemainderRow, fTrainingIdxs[currentChunk]);
      } else {
         // Create the Validation batches if this is not the first epoch
         createIdxs(processedEvents);
         std::cout << "Training indices size: " << fTrainingIdxs[currentChunk].size() << "\n";
         std::cout << "Validation indices size: " << fValidationIdxs[currentChunk].size() << "\n";
         fTrainingRemainderRow = fBatchLoader->CreateTrainingBatches(*fChunkTensor, *fTrainingRemainder, fTrainingRemainderRow, fTrainingIdxs[currentChunk]);
         fValidationRemainderRow = fBatchLoader->CreateValidationBatches(*fChunkTensor, *fValidationRemainder, fValidationRemainderRow, fValidationIdxs[currentChunk]);
      }
   }

   /// \brief plit the events of the current chunk into validation and training events
   /// \param processedEvents
   void createIdxs(std::size_t processedEvents)
   {  
      std::cout << "Processed events: " << processedEvents << "\n";
      std::cout << "Validation split: " << fValidationSplit << "\n";
      // Create a vector of number 1..processedEvents
      std::vector<std::size_t> row_order = std::vector<std::size_t>(processedEvents);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      // calculate the number of events used for validation
      std::size_t num_validation = ceil(processedEvents * fValidationSplit);

      std::cout << "Num validation: " << num_validation << "\n";

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
