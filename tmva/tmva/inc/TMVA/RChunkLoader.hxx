#ifndef TMVA_CHUNKLOADER
#define TMVA_CHUNKLOADER

#include <vector>
#include <utility>
#include <iostream>
#include <iterator>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "TRandom3.h"
#include "ROOT/RVec.hxx"
#include "TMVA/Tools.h"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class RChunkLoaderFunctor {

private:
   std::size_t fOffset = 0;
   std::size_t fVecSizeIdx = 0;
   std::size_t fEntries = 0;
   std::vector<std::size_t> fMaxVecSizes;

   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fTrainingBatches;
   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fValidationBatches;

   std::size_t fThreshold;
   std::vector<std::size_t> & fIndices;
   std::size_t fBatchSize;
   std::size_t fNumColumns;

   std::size_t fTrainRemainderEntries;
   std::size_t fValidationRemainderEntries;

   float fVecPadding;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;

   /// \brief Load the final given value into fChunkTensor
   /// \tparam First_T
   /// \param first
   template <typename First_T>
   void AssignToTensor(First_T first)
   {  
      fChunkTensor->GetData()[fOffset++] = first;
      fEntries++;
   }

   /// \brief Load the final given value into fChunkTensor
   /// \tparam VecType
   /// \param first
   template <typename VecType>
   void AssignToTensor(const ROOT::RVec<VecType> &first)
   {
      AssignVector(first);
      fEntries++;
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam First_T
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename First_T, typename... Rest_T>
   void AssignToTensor(First_T first, Rest_T... rest)
   {  
      fChunkTensor->GetData()[fOffset++] = first;

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam VecType
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename VecType, typename... Rest_T>
   void AssignToTensor(const ROOT::RVec<VecType> &first, Rest_T... rest)
   {
      AssignVector(first);

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Loop through the values of a given vector and load them into the RTensor
   /// Note: the given vec_size does not have to be the same size as the given vector
   ///       If the size is bigger than the given vector, zeros are used as padding.
   ///       If the size is smaller, the remaining values are ignored.
   /// \tparam VecType
   /// \param vec
   template <typename VecType>
   void AssignVector(const ROOT::RVec<VecType> &vec)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();

      for (std::size_t i = 0; i < max_vec_size; i++) {
         if (i < vec_size) {
            fChunkTensor->GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor->GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   RChunkLoaderFunctor(std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & trainingBatches,
                       std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & validationBatches,
                       std::size_t threshold, std::vector<std::size_t> & indices,
                       std::size_t batchSize, std::size_t numColumns,
                       std::size_t trainRemainderEntries, std::size_t validationRemainderEntries,
                       const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                       const float vecPadding = 0.0)
      : fTrainingBatches(trainingBatches),
        fValidationBatches(validationBatches),
        fThreshold(threshold),
        fIndices(indices),
        fBatchSize(batchSize),
        fNumColumns(numColumns),
        fTrainRemainderEntries(trainRemainderEntries),
        fValidationRemainderEntries(validationRemainderEntries),
        fMaxVecSizes(maxVecSizes),
        fVecPadding(vecPadding)
   {
   }

   /// \brief Loop through all columns of an event and put their values into an RTensor
   /// \param first
   /// \param ...rest
   void operator()(First first, Rest... rest)
   {  
      fVecSizeIdx = 0;

      if (fIndices[fEntries] < fThreshold){
         std::size_t index = fIndices[fEntries] + fTrainRemainderEntries;
         fChunkTensor = fTrainingBatches[index / fBatchSize];
         fOffset = (index % fBatchSize) * fNumColumns;
      }
      else{
         std::size_t index = fIndices[fEntries] + fValidationRemainderEntries - fThreshold;
         fChunkTensor = fValidationBatches[index / fBatchSize];
         fOffset = (index % fBatchSize) * fNumColumns;
      }

      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }
};

template <typename... Args>
class RChunkLoaderNoFilters {

private:
   std::size_t fChunkSize;

   std::vector<std::string> fCols;

   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;

   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fTrainRemainderEntries = 0;
   std::size_t fValRemainderEntries = 0;

   ROOT::RDF::RNode & f_rdf;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fTrainingRemainder;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainder;

   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fTrainingBatches;
   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fValidationBatches;

   std::vector<std::size_t> fIndices;

   float fValidationSplit;

   RandomGenerator<TRandom3> fRng;
   UInt_t fFixedSeed;
   RandomGenerator<TRandom3> fFixedRng;

   bool fShuffle = true;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoaderNoFilters(ROOT::RDF::RNode &rdf,
                std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & trainingBatches,
                std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & validationBatches,
                const std::size_t chunkSize, const std::vector<std::string> &cols,
                std::size_t batchSize, std::size_t numColumns, float validationSplit,
                bool shuffle, const std::vector<std::size_t> &vecSizes = {},
                const float vecPadding = 0.0)
      : f_rdf(rdf),
        fTrainingBatches(trainingBatches),
        fValidationBatches(validationBatches),
        fChunkSize(chunkSize),
        fCols(cols),
        fBatchSize(batchSize),
        fNumColumns(numColumns),
        fValidationSplit(validationSplit),
        fShuffle(shuffle),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding)
   {  
      if (fShuffle){
         fRng = TMVA::RandomGenerator<TRandom3>(0);

         do {
            fFixedSeed = fRng();
         } while (fFixedSeed == 0);
      }
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::size_t LoadChunk(const std::size_t currentRow, std::size_t valuesLeft)
   {  
      std::size_t sizeValidation = floor(valuesLeft * fValidationSplit);
      std::size_t sizeTraining = valuesLeft - sizeValidation;

      fTrainingBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fTrainingBatches.reserve(sizeTraining);

      if (fTrainRemainderEntries){
         fTrainingBatches.emplace_back(std::move(fTrainingRemainder));
      }

      for (std::size_t i = 0; i < sizeTraining / fBatchSize; i++){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      fValidationBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fValidationBatches.reserve(sizeValidation);

      if (fValRemainderEntries){
         fValidationBatches.emplace_back(std::move(fValidationRemainder));
      }

      for (std::size_t i = 0; i < sizeValidation / fBatchSize; i++){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }
      
      CreateIndices(valuesLeft, sizeTraining);

      RChunkLoaderFunctor<Args...> func(fTrainingBatches, fValidationBatches, sizeTraining, fIndices, fBatchSize,
         fNumColumns, fTrainRemainderEntries, fValRemainderEntries, fVecSizes, fVecPadding);

      fTrainRemainderEntries = (fTrainRemainderEntries + sizeTraining) % fBatchSize;
      fValRemainderEntries = (fValRemainderEntries + sizeValidation) % fBatchSize;

      if (fTrainRemainderEntries){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      if (fValRemainderEntries){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fChunkSize);
      auto myCount = f_rdf.Count();

      // load data
      f_rdf.Foreach(func, fCols);

      if (fTrainRemainderEntries){
         fTrainingRemainder = fTrainingBatches.back();
         fTrainingBatches.pop_back();
      }

      if (fValRemainderEntries){
         fValidationRemainder = fValidationBatches.back();
         fValidationBatches.pop_back();
      }

      // get loading info
      return myCount.GetValue();
   }

   void Activate(){
      fTrainRemainderEntries = 0;
      fValRemainderEntries = 0;
      fFixedRng.seed(fFixedSeed);
   }

   void CreateIndices(std::size_t valuesLeft, std::size_t threshold){
      fIndices = std::vector<std::size_t>(valuesLeft);
      std::iota(fIndices.begin(), fIndices.end(), 0);

      if (fShuffle) {
         std::shuffle(fIndices.begin(), fIndices.begin() + threshold, fRng);
         std::shuffle(fIndices.begin(), fIndices.end(), fFixedRng);
      }
   }

   std::pair<std::shared_ptr<TMVA::Experimental::RTensor<float>>,std::shared_ptr<TMVA::Experimental::RTensor<float>>>
   ReturnRemainderBatches(){
      std::shared_ptr<TMVA::Experimental::RTensor<float>> trainingBatch;
      std::shared_ptr<TMVA::Experimental::RTensor<float>> validationBatch;

      if (fTrainRemainderEntries){
         trainingBatch = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fTrainRemainderEntries, fNumColumns});

         for (std::size_t i = 0; i < fTrainRemainderEntries; i++){
            std::copy(fTrainingRemainder->GetData() + i * fNumColumns, fTrainingRemainder->GetData() + (i + 1) * fNumColumns,
               trainingBatch->GetData() + i * fNumColumns);
         }
      }

      if (fValRemainderEntries){
         validationBatch = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fValRemainderEntries, fNumColumns});

         for (std::size_t i = 0; i < fValRemainderEntries; i++){
            std::copy(fValidationRemainder->GetData() + i * fNumColumns, fValidationRemainder->GetData() + (i + 1) * fNumColumns,
               validationBatch->GetData() + i * fNumColumns);
         }
      }

      return std::make_pair(trainingBatch, validationBatch);
   }
};

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class RChunkLoaderFunctorFilters {

private:
   std::size_t fOffset = 0;
   std::size_t fVecSizeIdx = 0;
   std::size_t fEntries = 0;
   std::vector<std::size_t> fMaxVecSizes;

   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fTrainingBatches;
   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fValidationBatches;

   std::size_t fThreshold;
   std::vector<std::size_t> & fIndices;
   std::size_t fBatchSize;
   std::size_t fNumColumns;

   std::size_t fTrainRemainderEntries;
   std::size_t fValidationRemainderEntries;

   std::size_t fRemainderTensorEntries = 0;

   float fVecPadding;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fRemainderTensor;

   /// \brief Load the final given value into fChunkTensor
   /// \tparam First_T
   /// \param first
   template <typename First_T>
   void AssignToTensor(First_T first)
   {  
      fChunkTensor->GetData()[fOffset++] = first;
      fEntries++;
   }

   /// \brief Load the final given value into fChunkTensor
   /// \tparam VecType
   /// \param first
   template <typename VecType>
   void AssignToTensor(const ROOT::RVec<VecType> &first)
   {
      AssignVector(first);
      fEntries++;
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam First_T
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename First_T, typename... Rest_T>
   void AssignToTensor(First_T first, Rest_T... rest)
   {  
      fChunkTensor->GetData()[fOffset++] = first;

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Recursively loop through the given values, and load them onto the fChunkTensor
   /// \tparam VecType
   /// \tparam ...Rest_T
   /// \param first
   /// \param ...rest
   template <typename VecType, typename... Rest_T>
   void AssignToTensor(const ROOT::RVec<VecType> &first, Rest_T... rest)
   {
      AssignVector(first);

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   /// \brief Loop through the values of a given vector and load them into the RTensor
   /// Note: the given vec_size does not have to be the same size as the given vector
   ///       If the size is bigger than the given vector, zeros are used as padding.
   ///       If the size is smaller, the remaining values are ignored.
   /// \tparam VecType
   /// \param vec
   template <typename VecType>
   void AssignVector(const ROOT::RVec<VecType> &vec)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();

      for (std::size_t i = 0; i < max_vec_size; i++) {
         if (i < vec_size) {
            fChunkTensor->GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor->GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   RChunkLoaderFunctorFilters(std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & trainingBatches,
                       std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & validationBatches,
                       std::shared_ptr<TMVA::Experimental::RTensor<float>> remainderTensor,
                       std::size_t threshold, std::vector<std::size_t> & indices,
                       std::size_t batchSize, std::size_t numColumns,
                       std::size_t trainRemainderEntries, std::size_t validationRemainderEntries,
                       std::size_t entries,
                       const std::vector<std::size_t> &maxVecSizes = std::vector<std::size_t>(),
                       const float vecPadding = 0.0)
      : fTrainingBatches(trainingBatches),
        fValidationBatches(validationBatches),
        fRemainderTensor(remainderTensor),
        fThreshold(threshold),
        fIndices(indices),
        fBatchSize(batchSize),
        fNumColumns(numColumns),
        fTrainRemainderEntries(trainRemainderEntries),
        fValidationRemainderEntries(validationRemainderEntries),
        fEntries(entries),
        fMaxVecSizes(maxVecSizes),
        fVecPadding(vecPadding)
   {
   }

   /// \brief Loop through all columns of an event and put their values into an RTensor
   /// \param first
   /// \param ...rest
   void operator()(First first, Rest... rest)
   {  
      fVecSizeIdx = 0;

      if (fEntries >= fIndices.size()){
         fChunkTensor = fRemainderTensor;
         fOffset = fRemainderTensorEntries;
         fRemainderTensorEntries++;
      }
      else if (fIndices[fEntries] < fThreshold){
         std::size_t index = fIndices[fEntries] + fTrainRemainderEntries;
         fChunkTensor = fTrainingBatches[index / fBatchSize];
         fOffset = (index % fBatchSize) * fNumColumns;
      }
      else{
         std::size_t index = fIndices[fEntries] + fValidationRemainderEntries - fThreshold;
         fChunkTensor = fValidationBatches[index / fBatchSize];
         fOffset = (index % fBatchSize) * fNumColumns;
      }

      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }

   std::size_t & SetEntries(){ return fEntries; }
};

template <typename... Args>
class RChunkLoaderFilters {

private:
   std::vector<std::string> fCols;

   std::vector<std::size_t> fVecSizes;
   std::size_t fVecPadding;

   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fTrainRemainderEntries = 0;
   std::size_t fValRemainderEntries = 0;
   std::size_t fNumEntries;
   std::size_t fChunkTensorEntries = 0;

   ROOT::RDF::RNode & f_rdf;

   std::shared_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fTrainingRemainder;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fValidationRemainder;

   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fTrainingBatches;
   std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & fValidationBatches;

   std::vector<std::size_t> fIndices;

   float fValidationSplit;

   RandomGenerator<TRandom3> fRng;
   UInt_t fFixedSeed;
   RandomGenerator<TRandom3> fFixedRng;

   bool fShuffle = true;

   const std::size_t fPartOfChunkSize;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoaderFilters(ROOT::RDF::RNode &rdf,
                std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & trainingBatches,
                std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & validationBatches,
                const std::size_t chunkSize, const std::vector<std::string> &cols,
                std::size_t batchSize, std::size_t numColumns, std::size_t numEntries, float validationSplit,
                bool shuffle, const std::vector<std::size_t> &vecSizes = {},
                const float vecPadding = 0.0)
      : f_rdf(rdf),
        fTrainingBatches(trainingBatches),
        fValidationBatches(validationBatches),
        fCols(cols),
        fBatchSize(batchSize),
        fNumColumns(numColumns),
        fNumEntries(numEntries),
        fValidationSplit(validationSplit),
        fShuffle(shuffle),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fPartOfChunkSize(chunkSize / 2)
   {  
      if (fShuffle){
         fRng = TMVA::RandomGenerator<TRandom3>(0);

         do {
            fFixedSeed = fRng();
         } while (fFixedSeed == 0);
      }

      fChunkTensor = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fPartOfChunkSize, fNumColumns});
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::pair<std::size_t,std::size_t>
   LoadChunk(std::size_t valuesLeft, std::size_t processedEvents)
   {  
      std::size_t sizeValidation = floor(valuesLeft * fValidationSplit);
      std::size_t sizeTraining = valuesLeft - sizeValidation;

      std::size_t nTrainingBatches = sizeTraining / fBatchSize;
      std::size_t nValidationBatches = sizeValidation / fBatchSize;

      fTrainingBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fTrainingBatches.reserve(nTrainingBatches + 2);

      if (fTrainRemainderEntries){
         fTrainingBatches.emplace_back(std::move(fTrainingRemainder));
      }

      for (std::size_t i = 0; i < nTrainingBatches; i++){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      fValidationBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fValidationBatches.reserve(nValidationBatches + 2);

      if (fValRemainderEntries){
         fValidationBatches.emplace_back(std::move(fValidationRemainder));
      }

      for (std::size_t i = 0; i < nValidationBatches; i++){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }
      
      CreateIndices(valuesLeft, sizeTraining);

      if (fChunkTensorEntries){
         std::shared_ptr<TMVA::Experimental::RTensor<float>> batch;

         for (std::size_t entries = 0, offset = 0; entries < fChunkTensorEntries; entries++){
            if (fIndices[entries] < sizeTraining){
               std::size_t index = fIndices[entries] + fTrainRemainderEntries;
               batch = fTrainingBatches[index / fBatchSize];
               offset = (index % fBatchSize) * fNumColumns;

            }
            else{
               std::size_t index = fIndices[entries] + fValRemainderEntries - sizeTraining;
               batch = fValidationBatches[index / fBatchSize];
               offset = (index % fBatchSize) * fNumColumns;
            }

            std::copy(fChunkTensor->GetData() + entries * fNumColumns, fChunkTensor->GetData() + (entries + 1) * fNumColumns,
                  batch->GetData() + entries * fNumColumns);
         }
      }

      RChunkLoaderFunctorFilters<Args...> func(fTrainingBatches, fValidationBatches, fChunkTensor, sizeTraining, fIndices, fBatchSize,
         fNumColumns, fTrainRemainderEntries, fValRemainderEntries, fChunkTensorEntries, fVecSizes, fVecPadding);
      
      fTrainRemainderEntries = (fTrainRemainderEntries + sizeTraining) % fBatchSize;
      fValRemainderEntries = (fValRemainderEntries + sizeValidation) % fBatchSize;

      if (fTrainRemainderEntries){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      if (fValRemainderEntries){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }
      
      std::size_t passedEvents = fChunkTensorEntries;

      while(passedEvents < valuesLeft){
         ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, processedEvents, processedEvents + fPartOfChunkSize);
         auto report = f_rdf.Report();

         f_rdf.Foreach(func, fCols);

         processedEvents += report.begin()->GetAll();
         passedEvents += (report.end() - 1)->GetPass();

         func.SetEntries() = passedEvents;
      }

      fChunkTensorEntries = passedEvents - valuesLeft;

      if (fTrainRemainderEntries){
         fTrainingRemainder = fTrainingBatches.back();
         fTrainingBatches.pop_back();
      }

      if (fValRemainderEntries){
         fValidationRemainder = fValidationBatches.back();
         fValidationBatches.pop_back();
      }

      // get loading info
      return std::make_pair(passedEvents, processedEvents);
   }

   void LastChunk(bool dropRemainder){
      std::shared_ptr<TMVA::Experimental::RTensor<float>> batch;

      std::size_t sizeValidation = floor(fChunkTensorEntries * fValidationSplit);
      std::size_t sizeTraining = fChunkTensorEntries - sizeValidation;

      std::size_t nTrainingBatches = sizeTraining / fBatchSize;
      std::size_t nValidationBatches = sizeValidation / fBatchSize;

      fTrainingBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fTrainingBatches.reserve(nTrainingBatches + 2);

      if (fTrainRemainderEntries){
         fTrainingBatches.emplace_back(std::move(fTrainingRemainder));
      }

      for (std::size_t i = 0; i < nTrainingBatches; i++){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      fValidationBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fValidationBatches.reserve(nValidationBatches + 2);

      if (fValRemainderEntries){
         fValidationBatches.emplace_back(std::move(fValidationRemainder));
      }

      for (std::size_t i = 0; i < nValidationBatches; i++){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      std::cout << "fTrainRemainderEntries: " << fTrainRemainderEntries << "\n" << "sizeTraining: " << sizeTraining << "\n";

      std::size_t trainRemainderEntries = fTrainRemainderEntries + sizeTraining > fBatchSize ? (fTrainRemainderEntries + sizeTraining) % fBatchSize: 0;
      std::size_t valRemainderEntries = fValRemainderEntries + sizeValidation > fBatchSize ? (fValRemainderEntries + sizeValidation) % fBatchSize: 0;

      if (trainRemainderEntries){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      if (valRemainderEntries){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      for (std::size_t entries = 0, offset = 0; entries < fChunkTensorEntries; entries++){
         if (fIndices[entries] < sizeTraining){
            std::size_t index = fIndices[entries] + fTrainRemainderEntries;
            batch = fTrainingBatches[index / fBatchSize];
            offset = (index % fBatchSize) * fNumColumns;

         }
         else{
            std::size_t index = fIndices[entries] + fValRemainderEntries - sizeTraining;
            batch = fValidationBatches[index / fBatchSize];
            offset = (index % fBatchSize) * fNumColumns;
         }

         std::copy(fChunkTensor->GetData() + entries * fNumColumns, fChunkTensor->GetData() + (entries + 1) * fNumColumns,
               batch->GetData() + entries * fNumColumns);
      }

      if (dropRemainder){
         if (!trainRemainderEntries){
            fTrainingBatches.pop_back();
         }
         if (!valRemainderEntries){
            fValidationBatches.pop_back();
         }
      }
      else {
         std::cout << "DropRemainder else is called in LoadChunks\n";
         std::cout << "trainRemainderEntries: " << trainRemainderEntries << "\n";
         std::cout << "valRemainderEntries: " << valRemainderEntries << "\n";
         if (trainRemainderEntries){
            fTrainingRemainder = fTrainingBatches.back();
            fTrainingBatches.pop_back();

            std::shared_ptr<TMVA::Experimental::RTensor<float>> trainingBatch =
               std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{trainRemainderEntries, fNumColumns});
            
            for (std::size_t i = 0; i < trainRemainderEntries; i++){
               std::copy(fTrainingRemainder->GetData() + i * fNumColumns, fTrainingRemainder->GetData() + (i + 1) * fNumColumns,
                  trainingBatch->GetData() + i * fNumColumns);
            }

            fTrainingBatches.emplace_back(trainingBatch);
         }
         else if (fTrainRemainderEntries){
            std::cout << "fTrainRemainderEntries: " << fTrainRemainderEntries << "\n";
            fTrainingRemainder = fTrainingBatches.back();
            fTrainingBatches.pop_back();

            std::shared_ptr<TMVA::Experimental::RTensor<float>> trainingBatch =
               std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fTrainRemainderEntries, fNumColumns});
            
            for (std::size_t i = 0; i < fTrainRemainderEntries; i++){
               std::cout << "COPYING\n";
               std::copy(fTrainingRemainder->GetData() + i * fNumColumns, fTrainingRemainder->GetData() + (i + 1) * fNumColumns,
                  trainingBatch->GetData() + i * fNumColumns);
            }

            std::cout << "Printing remainder number: ";
            std::copy(fTrainingRemainder->GetData(), fTrainingRemainder->GetData() + 1,
                  std::ostream_iterator<int>(std::cout, ", "));
            std::cout << "\n";

            std::cout << "Printing remainder number copied: ";
            std::copy(trainingBatch->GetData(), trainingBatch->GetData() + 1,
                  std::ostream_iterator<int>(std::cout, ", "));
            std::cout << "\n";

            fTrainingBatches.emplace_back(trainingBatch);
         }

         if (valRemainderEntries){
            fValidationRemainder = fValidationBatches.back();
            fValidationBatches.pop_back();

            std::shared_ptr<TMVA::Experimental::RTensor<float>> validationBatch =
               std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{valRemainderEntries, fNumColumns});
            
            for (std::size_t i = 0; i < fValRemainderEntries; i++){
               std::copy(fValidationRemainder->GetData() + i * fNumColumns, fValidationRemainder->GetData() + (i + 1) * fNumColumns,
                  validationBatch->GetData() + i * fNumColumns);
            }

            fValidationBatches.emplace_back(validationBatch);
         }
         else if (fValRemainderEntries){
            fValidationRemainder = fValidationBatches.back();
            fValidationBatches.pop_back();

            std::shared_ptr<TMVA::Experimental::RTensor<float>> validationBatch =
               std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fValRemainderEntries, fNumColumns});
            
            for (std::size_t i = 0; i < fValRemainderEntries; i++){
               std::copy(fValidationRemainder->GetData() + i * fNumColumns, fValidationRemainder->GetData() + (i + 1) * fNumColumns,
                  validationBatch->GetData() + i * fNumColumns);
            }

            fValidationBatches.emplace_back(validationBatch);
         }
      }
   }

   std::pair<std::shared_ptr<TMVA::Experimental::RTensor<float>>,std::shared_ptr<TMVA::Experimental::RTensor<float>>>
   ReturnRemainderBatches(){
      std::shared_ptr<TMVA::Experimental::RTensor<float>> trainingBatch;
      std::shared_ptr<TMVA::Experimental::RTensor<float>> validationBatch;

      if (fTrainRemainderEntries){
         trainingBatch = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fTrainRemainderEntries, fNumColumns});

         for (std::size_t i = 0; i < fTrainRemainderEntries; i++){
            std::copy(fTrainingRemainder->GetData() + i * fNumColumns, fTrainingRemainder->GetData() + (i + 1) * fNumColumns,
               trainingBatch->GetData() + i * fNumColumns);
         }
      }

      if (fValRemainderEntries){
         validationBatch = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fValRemainderEntries, fNumColumns});

         for (std::size_t i = 0; i < fValRemainderEntries; i++){
            std::copy(fValidationRemainder->GetData() + i * fNumColumns, fValidationRemainder->GetData() + (i + 1) * fNumColumns,
               validationBatch->GetData() + i * fNumColumns);
         }
      }

      return std::make_pair(trainingBatch, validationBatch);
   }

   void CreateIndices(std::size_t valuesLeft, std::size_t threshold){
      fIndices = std::vector<std::size_t>(valuesLeft);
      std::iota(fIndices.begin(), fIndices.end(), 0);

      if (fShuffle) {
         std::shuffle(fIndices.begin(), fIndices.begin() + threshold, fRng);
         std::shuffle(fIndices.begin(), fIndices.end(), fFixedRng);
      }
   }

   void Activate(){
      fTrainRemainderEntries = 0;
      fValRemainderEntries = 0;
      fChunkTensorEntries = 0;
      fFixedRng.seed(fFixedSeed);
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_CHUNKLOADER
