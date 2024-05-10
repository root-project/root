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

   std::vector<std::string> fCols;

   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fMaxBatches;
   std::size_t fNumEntries;
   std::size_t fNumAllEntries = 0;

   float fValidationSplit;

   std::variant<std::shared_ptr<RChunkLoader<Args...>>, std::shared_ptr<RChunkLoaderFilters<Args...>>> fChunkLoaderNoFilters, fChunkLoaderFilters;

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

   std::vector<std::size_t> fVecSizes;
   float fVecPadding;

public:
   RBatchGenerator(ROOT::RDF::RNode &rdf, const std::size_t chunkSize,
                   const std::size_t batchSize, const std::vector<std::string> &cols,
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                   const float validationSplit = 0.0, const std::size_t maxChunks = 0, std::size_t numColumns = 0,
                   bool shuffle = true, bool dropRemainder = true)
      : f_rdf(rdf),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fShuffle(shuffle),
        fDropRemainder(dropRemainder),
        fUseWholeFile(maxChunks == 0),
        fNotFiltered(f_rdf.GetFilterNames().empty())
   {
      // limits the number of batches that can be contained in the batchqueue based on the chunksize
      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));

      {
         std::function<UInt_t(UInt_t)> GetSeedNumber;
         GetSeedNumber = [&](UInt_t seed_number)->UInt_t{return seed_number != 0? seed_number: GetSeedNumber(fRng());};
         fFixedSeed = GetSeedNumber(fRng());
      }
      
      if (numColumns == 0) {numColumns = cols.size();}

      // Create tensor to load the chunk into
      fChunkTensor =
         std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fChunkSize, numColumns});
      
      if(fNotFiltered){
         fNumEntries = f_rdf.Count().GetValue();

         fChunkLoaderNoFilters = std::make_unique<TMVA::Experimental::Internal::RChunkLoader<Args...>>(
            f_rdf, fChunkTensor, fChunkSize, fCols, fVecSizes, fVecPadding);
      }
      else{
         auto report = f_rdf.Report();
         fNumEntries = f_rdf.Count().GetValue();
         fNumAllEntries = report.begin()->GetAll();

         fChunkLoaderFilters = std::make_unique<TMVA::Experimental::Internal::RChunkLoaderFilters<Args...>>(
            f_rdf, fChunkTensor, fChunkSize, fCols, fNumEntries, fNumAllEntries, fVecSizes, fVecPadding);
      }
      
      fBatchLoader = std::make_unique<TMVA::Experimental::Internal::RBatchLoader>(*fChunkTensor,
                  fBatchSize, numColumns, fMaxBatches);
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
      if (fDropRemainder && (std::size_t)ceil(fNumEntries * (1 - fValidationSplit)) % fBatchSize){
         return ((fNumEntries / fChunkSize) * (fChunkSize - floor(fChunkSize * fValidationSplit)) +
            ceil((fNumEntries % fChunkSize) * (1 - fValidationSplit))) / fBatchSize;
      }

      return ((fNumEntries / fChunkSize) * (fChunkSize - floor(fChunkSize * fValidationSplit)) +
         ceil((fNumEntries % fChunkSize) * (1 - fValidationSplit))) / fBatchSize + 1;
   }

   std::size_t NumberOfValidationBatches(){
      if (std::size_t remainderRows = fNumEntries % fBatchSize;
          remainderRows == floor(remainderRows * fValidationSplit) ||
            (fDropRemainder && (std::size_t)floor(fNumEntries * fValidationSplit) % fBatchSize)){
               
         return ((fNumEntries / fChunkSize) * floor(fChunkSize * fValidationSplit) +
            floor((fNumEntries % fChunkSize) * fValidationSplit)) / fBatchSize;
      }
      
      return ((fNumEntries / fChunkSize) * floor(fChunkSize * fValidationSplit) +
         floor((fNumEntries % fChunkSize) * fValidationSplit)) / fBatchSize + 1;
   }

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
         std::pair<std::size_t, std::size_t> report =
            std::get<std::shared_ptr<RChunkLoader<Args...>>>(fChunkLoaderNoFilters)->LoadChunk(currentRow);
         currentRow += report.first;

         CreateBatches(report.second);
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
         std::pair<std::size_t, std::size_t> report = std::get<std::shared_ptr<RChunkLoaderFilters<Args...>>>(fChunkLoaderFilters)->LoadChunk(currentRow);

         currentRow += report.first;
         processedEvents += report.second;

         CreateBatches(report.second);
      }

      if (currentChunk < fMaxChunks || fUseWholeFile){
         CreateBatches(std::get<std::shared_ptr<RChunkLoaderFilters<Args...>>>(fChunkLoaderFilters)->LastChunk());
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
