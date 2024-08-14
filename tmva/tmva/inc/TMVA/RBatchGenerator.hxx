#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>
#include <variant>

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
   UInt_t fFixedSeed;
   TMVA::RandomGenerator<TRandom3> fFixedRng;

   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fNumEntries;

   float fValidationSplit;

   std::variant<std::shared_ptr<RChunkLoader<Args...>>, std::shared_ptr<RChunkLoaderFilters<Args...>>> fChunkLoader;

   std::unique_ptr<RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fUseWholeFile = true;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   
   ROOT::RDF::RNode &f_rdf;

   // filled batch elements
   std::mutex fIsActiveLock;

   bool fDropRemainder = true;
   bool fShuffle = true;
   bool fIsActive = false;
   bool fNotFiltered;

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
        fShuffle(shuffle),
        fDropRemainder(dropRemainder),
        fUseWholeFile(maxChunks == 0),
        fNotFiltered(f_rdf.GetFilterNames().empty())
   {

      do {
         fFixedSeed = fRng();
      } while (fFixedSeed == 0);

      // Create tensor to load the chunk into
      fChunkTensor =
         std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fChunkSize, numColumns});

      if(fNotFiltered){
         fNumEntries = f_rdf.Count().GetValue();

         fChunkLoader = std::make_unique<TMVA::Experimental::Internal::RChunkLoader<Args...>>(
            f_rdf, fChunkTensor, fChunkSize, cols, vecSizes, vecPadding);
      }
      else{
         auto report = f_rdf.Report();
         fNumEntries = f_rdf.Count().GetValue();
         std::size_t numAllEntries = report.begin()->GetAll();

         fChunkLoader = std::make_unique<TMVA::Experimental::Internal::RChunkLoaderFilters<Args...>>(
            f_rdf, fChunkTensor, fChunkSize, cols, fNumEntries, numAllEntries, vecSizes, vecPadding);
      }
      
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

      fFixedRng.seed(fFixedSeed);
      fBatchLoader->Activate();
      // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
      if (fNotFiltered){
            fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksNoFilters, this);
      }
      else{
            fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunksFilters, this);
      }
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

         // A pair that consists the proccessed, and passed events while loading the chunk
         std::size_t report = std::get<std::shared_ptr<RChunkLoader<Args...>>>(fChunkLoader)->LoadChunk(currentRow);
         currentRow += report;

         CreateBatches(report);
      }

      if (!fDropRemainder){
         fBatchLoader->LastBatches();
      }

      fBatchLoader->DeActivate();
   }

   void LoadChunksFilters()
   {     
      std::size_t currentChunk = 0;
      for (std::size_t processedEvents = 0, currentRow = 0; ((currentChunk < fMaxChunks) || fUseWholeFile) && processedEvents < fNumEntries;
           currentChunk++) {

         // stop the loop when the loading is not active anymore
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            if (!fIsActive)
               return;
         }

         // A pair that consists the proccessed, and passed events while loading the chunk
         std::pair<std::size_t, std::size_t> report = std::get<std::shared_ptr<RChunkLoaderFilters<Args...>>>(fChunkLoader)->LoadChunk(currentRow);

         currentRow += report.first;
         processedEvents += report.second;

         CreateBatches(report.second);
      }

      if (currentChunk < fMaxChunks || fUseWholeFile){
         CreateBatches(std::get<std::shared_ptr<RChunkLoaderFilters<Args...>>>(fChunkLoader)->LastChunk());
      }

      if (!fDropRemainder){
         fBatchLoader->LastBatches();
      }

      fBatchLoader->DeActivate();
   }

   /// \brief Create batches
   /// \param processedEvents
   void CreateBatches(std::size_t processedEvents)
   {
      std::pair<std::vector<std::size_t>, std::vector<std::size_t>> indices = createIndices(processedEvents);

      fBatchLoader->CreateTrainingBatches(indices.first);
      fBatchLoader->CreateValidationBatches(indices.second);
   }

   /// \brief split the events of the current chunk into training and validation events, shuffle if needed
   /// \param events
   std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
   createIndices(std::size_t events)
   {  
      // Create a vector of number 1..events
      std::vector<std::size_t> row_order = std::vector<std::size_t>(events);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fFixedRng);
      }

      // calculate the number of events used for validation
      std::size_t num_validation = floor(events * fValidationSplit);

      // Devide the vector into training and validation and return
      std::vector<std::size_t> trainingIndices = std::vector<std::size_t>({row_order.begin(), row_order.end() - num_validation});
      std::vector<std::size_t> validationIndices = std::vector<std::size_t>({row_order.end() - num_validation, row_order.end()});

      if (fShuffle) {
         std::shuffle(trainingIndices.begin(), trainingIndices.end(), fRng);
      }

      return std::make_pair(trainingIndices, validationIndices);
   }

   bool IsActive() { return fIsActive; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR
