#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>
#include <variant>
#include <algorithm>

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
   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fNumEntries;

   float fValidationSplit;

   std::unique_ptr<RChunkLoader<Args...>> fChunkLoader;

   std::unique_ptr<RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fUseWholeFile = true;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatches;
   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatches;

   ROOT::RDF::RNode &f_rdf;

   // filled batch elements
   std::mutex fIsActiveLock;

   bool fDropRemainder = true;
   bool fIsActive = false;

public:
   RBatchGenerator(ROOT::RDF::RNode &rdf, const std::size_t chunkSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols,
                   const std::size_t numColumns,
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0,
                   bool shuffle = true, bool dropRemainder = true)
      : f_rdf(rdf),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fDropRemainder(dropRemainder),
        fUseWholeFile(maxChunks == 0)
   {
      fNumEntries = f_rdf.Count().GetValue();

      fChunkLoader = std::make_unique<TMVA::Experimental::Internal::RChunkLoader<Args...>>(
         f_rdf, fTrainingBatches, fValidationBatches, fChunkSize, cols, fBatchSize,
            numColumns, validationSplit, shuffle, vecSizes, vecPadding);
      
      std::size_t maxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));

      // limits the number of batches that can be contained in the batchqueue based on the chunksize
      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader>(*fChunkTensor,
                  fBatchSize, numColumns, maxBatches);
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

      fBatchLoader->Activate();
      fChunkLoader->Activate();

      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksNoFilters, this);
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

   std::size_t NumberOfTrainingBatches(){
      std::size_t entriesForTraining = (fNumEntries / fChunkSize) * (fChunkSize - floor(fChunkSize * fValidationSplit)) +
            fNumEntries % fChunkSize - floor(fValidationSplit * (fNumEntries % fChunkSize));

      if (fDropRemainder || !(entriesForTraining % fBatchSize))
      {
         return entriesForTraining / fBatchSize;
      }

      return entriesForTraining / fBatchSize + 1;
   }

   /// @brief Return number of training remainder rows
   /// @return 
   std::size_t TrainRemainderRows(){
      std::size_t entriesForTraining = (fNumEntries / fChunkSize) * (fChunkSize - floor(fChunkSize * fValidationSplit)) +
            fNumEntries % fChunkSize - floor(fValidationSplit * (fNumEntries % fChunkSize));

      if (fDropRemainder || !(entriesForTraining % fBatchSize))
      {
         return 0;
      }

      return entriesForTraining % fBatchSize;
   }

   /// @brief Calculate number of validation batches and return it
   /// @return 
   std::size_t NumberOfValidationBatches(){
      std::size_t entriesForValidation = (fNumEntries / fChunkSize) * floor(fChunkSize * fValidationSplit) +
            floor((fNumEntries % fChunkSize) * fValidationSplit);

      if (fDropRemainder || !(entriesForValidation%fBatchSize)){

         return entriesForValidation / fBatchSize;
      }
      
      return entriesForValidation / fBatchSize + 1; 
   }

   /// @brief Return number of validation remainder rows
   /// @return 
   std::size_t ValidationRemainderRows(){
      std::size_t entriesForValidation = (fNumEntries / fChunkSize) * floor(fChunkSize * fValidationSplit) +
            floor((fNumEntries % fChunkSize) * fValidationSplit);

      if (fDropRemainder || !(entriesForValidation%fBatchSize)){

         return 0;
      }
      
      return entriesForValidation % fBatchSize; 
   }

   /// @brief Load chunks when no filters are applied on rdataframe
   void LoadChunksNoFilters()
   {
      for (std::size_t currentChunk = 0, currentRow = 0; ((currentChunk < fMaxChunks) || fUseWholeFile) && currentRow < fNumEntries;
           currentChunk++) {

         // stop the loop when the loading is not active anymore
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         std::size_t valuesLeft = std::min(fChunkSize, fNumEntries - currentRow);

         // A pair that consists the proccessed, and passed events while loading the chunk
         std::size_t report = fChunkLoader->LoadChunk(currentRow, valuesLeft);
         currentRow += report;

         fBatchLoader->UnloadTrainingVectors(fTrainingBatches);
         fBatchLoader->UnloadValidationVectors(fValidationBatches);
      }

      if (!fDropRemainder){
         fBatchLoader->UnloadRemainder(fChunkLoader->ReturnRemainderBatches());
      }

      fBatchLoader->DeActivate();
   }

   bool IsActive() { return fIsActive; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR
