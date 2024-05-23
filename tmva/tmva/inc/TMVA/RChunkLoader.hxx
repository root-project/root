#ifndef TMVA_CHUNKLOADER
#define TMVA_CHUNKLOADER

#include <vector>
#include <iostream>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

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
      std::cout << "Second assign to tensor: " << first << "\n";
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
      std::cout << "First assign to tensor " << first << "\n";
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
      std::cout << "Entering operator\n";
      fVecSizeIdx = 0;

      if (fIndices[fEntries] < fThreshold){
         std::cout << "Index in if: " << fIndices[fEntries] << "\n";
         std::cout << "Batch number: " << fIndices[fEntries] / fBatchSize << "\n";
         std::size_t index = fIndices[fEntries] + fTrainRemainderEntries;
         fChunkTensor = fTrainingBatches[index / fBatchSize];
         fOffset = (index % fBatchSize) * fNumColumns;
      }
      else{
         std::size_t index = fIndices[fEntries] + fValidationRemainderEntries - fThreshold;
         std::cout << "Index in else: " << index << "\n";
         fChunkTensor = fValidationBatches[index / fBatchSize];
         fOffset = (index % fBatchSize) * fNumColumns;
      }

      std::cout << "Offset " << fOffset << "\n";
      std::cout << "Leaving operator\n";

      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }
};

template <typename... Args>
class RChunkLoader {

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

   float fValidationSplit;

public:
   /// \brief Constructor for the RChunkLoader
   /// \param rdf
   /// \param chunkSize
   /// \param cols
   /// \param vecSizes
   /// \param vecPadding
   RChunkLoader(ROOT::RDF::RNode &rdf,
                std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & trainingBatches,
                std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & validationBatches,
                const std::size_t chunkSize, const std::vector<std::string> &cols,
                std::size_t batchSize, std::size_t numColumns, float validationSplit,
                const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0)
      : f_rdf(rdf),
        fTrainingBatches(trainingBatches),
        fValidationBatches(validationBatches),
        fChunkSize(chunkSize),
        fCols(cols),
        fBatchSize(batchSize),
        fNumColumns(numColumns),
        fValidationSplit(validationSplit),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding)
   {
   }

   /// \brief Load a chunk of data using the RChunkLoaderFunctor
   /// \param chunkTensor
   /// \param currentRow
   /// \return A pair of size_t defining the number of events processed and how many passed all filters
   std::size_t LoadChunk(const std::size_t currentRow, std::size_t valuesLeft)
   {  
      std::cout << "Values left: " << valuesLeft << "\n";

      std::size_t sizeValidation = floor(valuesLeft * fValidationSplit);
      std::size_t sizeTraining = valuesLeft - sizeValidation;

      fTrainingBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fTrainingBatches.reserve(sizeTraining);

      std::cout << "fBatchSize: " << fBatchSize << "\n fNumColumns: " << fNumColumns << "\n";

      if (fTrainRemainderEntries){
         std::cout << "Train remainder works\n"; 
         fTrainingBatches.emplace_back(std::move(fTrainingRemainder));
      }

      std::cout << "Number of training batches: " << sizeTraining / fBatchSize << "\n";

      for (std::size_t i = 0; i < sizeTraining / fBatchSize; i++){
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      std::cout << "Size of fTrainingBatches: " << fTrainingBatches.size() << "\n";

      if (fTrainingBatches[1])
         std::cout << "Size of batch inside " << fTrainingBatches[1]->GetSize() << "\n";

      fValidationBatches = std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>>();
      fValidationBatches.reserve(sizeValidation);

      if (fValRemainderEntries){
         fValidationBatches.emplace_back(std::move(fValidationRemainder));
      }

      for (std::size_t i = 0; i < sizeValidation / fBatchSize; i++){
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      std::cout << "Size of fValidationBatches: " << fValidationBatches.size() << "\n";

      std::vector<std::size_t> indices(valuesLeft);
      std::iota(indices.begin(), indices.end(), 0);

      std::cout << "Print out indices:\n";
      for (std::size_t a: indices){
         std::cout << a << ", ";
      } 

      std::cout << "\n";
      
      RChunkLoaderFunctor<Args...> func(fTrainingBatches, fValidationBatches, sizeTraining, indices, fBatchSize,
         fNumColumns, fTrainRemainderEntries, fValRemainderEntries, fVecSizes, fVecPadding);

      std::cout << "Size fTrainremainderEntries" << fTrainRemainderEntries << "\n";
      std::cout << "Size fValRemainderEntries" << fValRemainderEntries << "\n";
      std::cout << "Size training: " << sizeTraining << "\n";
      std::cout << "Size validation: " << sizeValidation << "\n";

      std::cout << "mid of train remainder entries" << fTrainRemainderEntries + sizeTraining << "\n";

      fTrainRemainderEntries = (fTrainRemainderEntries + sizeTraining) % fBatchSize;
      fValRemainderEntries = (fValRemainderEntries + sizeValidation) % fBatchSize;

      std::cout << "fTrainRemainderEntries: " << fTrainRemainderEntries << "\n";
      std::cout << "fValRemainderEntries: " << fValRemainderEntries << "\n";

      if (fTrainRemainderEntries){
         std::cout << "Emplacing train remainder\n";
         fTrainingBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      if (fValRemainderEntries){
         std::cout << "Emplacing validation remainder\n";
         fValidationBatches.emplace_back(std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns}));
      }

      ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, currentRow, currentRow + fChunkSize);
      auto myCount = f_rdf.Count();

      std::cout << "Before Foreach\n";

      // load data
      f_rdf.Foreach(func, fCols);

      std::cout << "After Foreach\n";

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
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_CHUNKLOADER
