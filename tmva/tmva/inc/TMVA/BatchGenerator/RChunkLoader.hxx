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

#ifndef TMVA_RCHUNKLOADER
#define TMVA_RCHUNKLOADER

#include <vector>
#include <random>

#include "TMVA/BatchGenerator/RChunkConstructor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "TMVA/BatchGenerator/RFlat2DMatrix.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RChunkLoaderFunctor
\ingroup tmva
\brief Loading chunks made in RChunkLoader into tensors from data from RDataFrame.
*/

template <typename... ColTypes>
class RChunkLoaderFunctor {
   // clang-format on   
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};
   RFlat2DMatrix &fChunkTensor;

   std::size_t fNumChunkCols;

   int fI;
   int fNumColumns;

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RTensor when the column consits of vectors 
   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec, int i, int numColumns)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();

      float *dst = fChunkTensor.GetData() + fOffset + numColumns * i;
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), dst);
         std::fill(dst + vec_size, dst + max_vec_size, fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, dst);
      }
       fOffset += max_vec_size;
    }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RTensor when the column consits of single values 
   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val, int i, int numColumns)
   {
      fChunkTensor.GetData()[fOffset + numColumns * i] = val;
      fOffset++;
      // fChunkTensor.GetData()[numColumns * i] = val;
   }

public:
   RChunkLoaderFunctor(RFlat2DMatrix &chunkTensor, std::size_t numColumns,
                       const std::vector<std::size_t> &maxVecSizes, float vecPadding, int i)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding), fI(i), fNumColumns(numColumns)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      (AssignToTensor(cols, fI, fNumColumns), ...);
   }
};

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RChunkLoader
\ingroup tmva
\brief Building and loading the chunks from the blocks and chunks constructed in RChunkConstructor

In this class the blocks are stiches together to form chunks that are loaded into memory. The blocks used to create each chunk comes from different parts of the dataset. This is achieved by shuffling the blocks before distributing them into chunks. The purpose of this process is to reduce bias during machine learning training by ensuring that the data is well mixed. The dataset is also spit into training and validation sets with the user-defined validation split fraction.
*/

template <typename... Args>
class RChunkLoader {
private:
   // clang-format on   
   std::size_t fNumEntries;
   std::size_t fChunkSize;
   std::size_t fBlockSize;
   float fValidationSplit;

   std::vector<std::size_t> fVecSizes;
   std::size_t fSumVecSizes;
   std::size_t fVecPadding;
   std::size_t fNumChunkCols;

   std::size_t fNumTrainEntries;
   std::size_t fNumValidationEntries;

   ROOT::RDF::RNode &f_rdf;
   std::vector<std::string> fCols;
   std::size_t fNumCols;
   std::size_t fSetSeed;

   bool fNotFiltered;
   bool fShuffle;

   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> fEntries;

   std::unique_ptr<RChunkConstructor> fTraining;
   std::unique_ptr<RChunkConstructor> fValidation;

public:
   RChunkLoader(ROOT::RDF::RNode &rdf, std::size_t numEntries,
                ROOT::RDF::RResultPtr<std::vector<ULong64_t>> rdf_entries, const std::size_t chunkSize,
                const std::size_t blockSize, const float validationSplit, const std::vector<std::string> &cols,
                const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0, bool shuffle = true,
                const std::size_t setSeed = 0)
      : f_rdf(rdf),
        fNumEntries(numEntries),
        fEntries(rdf_entries),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fChunkSize(chunkSize),
        fBlockSize(blockSize),
        fValidationSplit(validationSplit),
        fNotFiltered(f_rdf.GetFilterNames().empty()),
        fShuffle(shuffle),
        fSetSeed(setSeed)
   {
      fNumCols = fCols.size();
      fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);

      fNumChunkCols = fNumCols + fSumVecSizes - fVecSizes.size();

      // number of training and validation entries after the split
      fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
      fNumTrainEntries = fNumEntries - fNumValidationEntries;

      fTraining = std::make_unique<RChunkConstructor>(fNumTrainEntries, fChunkSize, fBlockSize);
      fValidation = std::make_unique<RChunkConstructor>(fNumValidationEntries, fChunkSize, fBlockSize);
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Distribute the blocks into training and validation datasets
   void SplitDataset()
   {
      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      std::vector<Long_t> BlockSizes = {};

      // fill the training and validation block sizes
      for (size_t i = 0; i < fTraining->NumberOfDifferentBlocks.size(); i++) {
         BlockSizes.insert(BlockSizes.end(), fTraining->NumberOfDifferentBlocks[i], fTraining->SizeOfBlocks[i]);
      }

      for (size_t i = 0; i < fValidation->NumberOfDifferentBlocks.size(); i++) {
         BlockSizes.insert(BlockSizes.end(), fValidation->NumberOfDifferentBlocks[i], fValidation->SizeOfBlocks[i]);
      }

      // make an identity permutation map
      std::vector<Long_t> indices(BlockSizes.size());

      for (int i = 0; i < indices.size(); ++i) {
         indices[i] = i;
      }

      // shuffle the identity permutation to create a new permutation
      if (fShuffle) {
         std::shuffle(indices.begin(), indices.end(), g);
      }

      // use the permuation to shuffle the vector of block sizes  
      std::vector<Long_t> PermutedBlockSizes(BlockSizes.size());
      for (int i = 0; i < BlockSizes.size(); ++i) {
         PermutedBlockSizes[i] = BlockSizes[indices[i]];
      }

      // create a vector for storing the boundaries of the blocks
      std::vector<Long_t> BlockBoundaries(BlockSizes.size());

      // get the boundaries of the blocks with the partial sum of the block sizes
      // insert 0 at the beginning for the lower boundary of the first block
      std::partial_sum(PermutedBlockSizes.begin(), PermutedBlockSizes.end(), BlockBoundaries.begin());
      BlockBoundaries.insert(BlockBoundaries.begin(), 0);

      // distribute the neighbouring block boudaries into pairs to get the intevals for the blocks
      std::vector<std::pair<Long_t, Long_t>> BlockIntervals;
      for (size_t i = 0; i < BlockBoundaries.size() - 1; ++i) {
         BlockIntervals.emplace_back(BlockBoundaries[i], BlockBoundaries[i + 1]);
      }

      // use the inverse of the permutation above to order the block intervals in the same order as
      // the original vector of block sizes
      std::vector<std::pair<Long_t, Long_t>> UnpermutedBlockIntervals(BlockIntervals.size());
      for (int i = 0; i < BlockIntervals.size(); ++i) {
         UnpermutedBlockIntervals[indices[i]] = BlockIntervals[i];
      }

      // distribute the block intervals between training and validation
      fTraining->BlockIntervals.insert(fTraining->BlockIntervals.begin(), UnpermutedBlockIntervals.begin(),
                                       UnpermutedBlockIntervals.begin() + fTraining->NumberOfBlocks);
      fValidation->BlockIntervals.insert(fValidation->BlockIntervals.begin(),
                                         UnpermutedBlockIntervals.begin() + fTraining->NumberOfBlocks,
                                         UnpermutedBlockIntervals.end());

      // distribute the different block intervals types for training and validation
      fTraining->DistributeBlockIntervals();
      fValidation->DistributeBlockIntervals();
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Create training chunks consisiting of block intervals of different types 
   void CreateTrainingChunksIntervals()
   {

      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      // shuffle the block intervals within each type of block
      if (fShuffle) {
         std::shuffle(fTraining->FullBlockIntervalsInFullChunks.begin(),
                      fTraining->FullBlockIntervalsInFullChunks.end(), g);
         std::shuffle(fTraining->LeftoverBlockIntervalsInFullChunks.begin(),
                      fTraining->LeftoverBlockIntervalsInFullChunks.end(), g);
         std::shuffle(fTraining->FullBlockIntervalsInLeftoverChunks.begin(),
                      fTraining->FullBlockIntervalsInLeftoverChunks.end(), g);
         std::shuffle(fTraining->LeftoverBlockIntervalsInLeftoverChunks.begin(),
                      fTraining->LeftoverBlockIntervalsInLeftoverChunks.end(), g);
      }

      // reset the chunk intervals and sizes before each epoch
      fTraining->ChunksIntervals = {};
      fTraining->ChunksSizes = {};

      // create the chunks each consisiting of block intervals
      fTraining->CreateChunksIntervals();

      if (fShuffle) {
         std::shuffle(fTraining->ChunksIntervals.begin(), fTraining->ChunksIntervals.end(), g);
      }

      fTraining->SizeOfChunks();
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Create training chunks consisiting of block intervals of different types 
   void CreateValidationChunksIntervals()
   {
      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      if (fShuffle) {
         std::shuffle(fValidation->FullBlockIntervalsInFullChunks.begin(),
                      fValidation->FullBlockIntervalsInFullChunks.end(), g);
         std::shuffle(fValidation->LeftoverBlockIntervalsInFullChunks.begin(),
                      fValidation->LeftoverBlockIntervalsInFullChunks.end(), g);
         std::shuffle(fValidation->FullBlockIntervalsInLeftoverChunks.begin(),
                      fValidation->FullBlockIntervalsInLeftoverChunks.end(), g);
         std::shuffle(fValidation->LeftoverBlockIntervalsInLeftoverChunks.begin(),
                      fValidation->LeftoverBlockIntervalsInLeftoverChunks.end(), g);
      }

      fValidation->ChunksIntervals = {};
      fValidation->ChunksSizes = {};

      fValidation->CreateChunksIntervals();

      if (fShuffle) {
         std::shuffle(fValidation->ChunksIntervals.begin(), fValidation->ChunksIntervals.end(), g);
      }

      fValidation->SizeOfChunks();
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Load the nth chunk from the training dataset into a tensor
   /// \param[in] TrainChunkTensor RTensor for the training chunk
   /// \param[in] chunk Index of the chunk in the dataset
   void LoadTrainingChunk(RFlat2DMatrix &TrainChunkTensor, std::size_t chunk)
    {

      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      std::size_t chunkSize = fTraining->ChunksSizes[chunk];

      if (chunk < fTraining->Chunks) {
         RFlat2DMatrix Tensor(chunkSize, fNumChunkCols);
         TrainChunkTensor.Resize(chunkSize, fNumChunkCols);

         // make an identity permutation map        
         std::vector<int> indices(chunkSize);
         std::iota(indices.begin(), indices.end(), 0);

         // shuffle the identity permutation to create a new permutation         
         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         // fill a chunk by looping over the blocks in a chunk (see RChunkConstructor)
         std::size_t chunkEntry = 0;
         std::vector<std::pair<Long_t, Long_t>> BlocksInChunk = fTraining->ChunksIntervals[chunk];

         std::sort(BlocksInChunk.begin(), BlocksInChunk.end(),
                   [](const std::pair<Long_t, Long_t>& a, const std::pair<Long_t, Long_t>& b) {
                      return a.first < b.first;
                   });
         
         for (std::size_t i = 0; i < BlocksInChunk.size(); i++) {

            // Use the block start and end entry to load into the chunk if the dataframe is not filtered
            if (fNotFiltered) {
               RChunkLoaderFunctor<Args...> func(Tensor, fNumChunkCols, fVecSizes, fVecPadding, chunkEntry);
               ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, BlocksInChunk[i].first, BlocksInChunk[i].second);

               f_rdf.Foreach(func, fCols);
               chunkEntry += BlocksInChunk[i].second - BlocksInChunk[i].first;
            }

            // use the entry column of the dataframe as a map to load the entries that passed the filters
            else {
               std::size_t blockSize = BlocksInChunk[i].second - BlocksInChunk[i].first;
               for (std::size_t j = 0; j < blockSize; j++) {
                  RChunkLoaderFunctor<Args...> func(Tensor, fNumChunkCols, fVecSizes, fVecPadding, chunkEntry);
                  ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, (*fEntries)[BlocksInChunk[i].first + j],
                                                                (*fEntries)[BlocksInChunk[i].first + j + 1]);
                  f_rdf.Foreach(func, fCols);
                  chunkEntry++;
               }
            }
         }

         // shuffle data in RTensor with the permutation map defined above
         for (std::size_t i = 0; i < chunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumChunkCols,
                      Tensor.GetData() + (indices[i] + 1) * fNumChunkCols,
                      TrainChunkTensor.GetData() + i * fNumChunkCols);
         }
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Load the nth chunk from the validation dataset into a tensor
   /// \param[in] ValidationChunkTensor RTensor for the validation chunk
   /// \param[in] chunk Index of the chunk in the dataset
   void LoadValidationChunk(RFlat2DMatrix &ValidationChunkTensor, std::size_t chunk)
    {

      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      std::size_t chunkSize = fValidation->ChunksSizes[chunk];

      if (chunk < fValidation->Chunks) {
         RFlat2DMatrix Tensor(chunkSize, fNumChunkCols);
         ValidationChunkTensor.Resize(chunkSize, fNumChunkCols);

         // make an identity permutation map        
         std::vector<int> indices(chunkSize);
         std::iota(indices.begin(), indices.end(), 0);

         // shuffle the identity permutation to create a new permutation
         if (fShuffle) {
            std::shuffle(indices.begin(), indices.end(), g);
         }

         std::size_t chunkEntry = 0;
         std::vector<std::pair<Long_t, Long_t>> BlocksInChunk = fValidation->ChunksIntervals[chunk];

         std::sort(BlocksInChunk.begin(), BlocksInChunk.end(),
                   [](const std::pair<Long_t, Long_t>& a, const std::pair<Long_t, Long_t>& b) {
                      return a.first < b.first;
                   });
         
         for (std::size_t i = 0; i < BlocksInChunk.size(); i++) {

            // use the block start and end entry to load into the chunk if the dataframe is not filtered
            if (fNotFiltered) {
               RChunkLoaderFunctor<Args...> func(Tensor, fNumChunkCols, fVecSizes, fVecPadding, chunkEntry);
               ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, BlocksInChunk[i].first, BlocksInChunk[i].second);
               f_rdf.Foreach(func, fCols);
               chunkEntry += BlocksInChunk[i].second - BlocksInChunk[i].first;
            }

            // use the entry column of the dataframe as a map to load the entries that passed the filters
            else {
               std::size_t blockSize = BlocksInChunk[i].second - BlocksInChunk[i].first;
               for (std::size_t j = 0; j < blockSize; j++) {
                  RChunkLoaderFunctor<Args...> func(Tensor, fNumChunkCols, fVecSizes, fVecPadding, chunkEntry);
                  ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, (*fEntries)[BlocksInChunk[i].first + j],
                                                                (*fEntries)[BlocksInChunk[i].first + j + 1]);

                  f_rdf.Foreach(func, fCols);
                  chunkEntry++;
               }
            }
         }

         // shuffle data in RTensor with the permutation map defined above         
         for (std::size_t i = 0; i < chunkSize; i++) {
            std::copy(Tensor.GetData() + indices[i] * fNumChunkCols,
                      Tensor.GetData() + (indices[i] + 1) * fNumChunkCols,
                      ValidationChunkTensor.GetData() + i * fNumChunkCols);
         }
      }
   }

   std::vector<std::size_t> GetTrainingChunkSizes() { return fTraining->ChunksSizes; }
   std::vector<std::size_t> GetValidationChunkSizes() { return fValidation->ChunksSizes; }

   std::size_t GetNumTrainingEntries() { return fNumTrainEntries; }
   std::size_t GetNumValidationEntries() { return fNumValidationEntries; }

   void CheckIfUnique(RFlat2DMatrix &Tensor)
   {
      const auto &rvec = Tensor.fRVec;
      if(std::set<float>(rvec.begin(), rvec.end()).size() == rvec.size()) {
         std::cout << "Tensor consists of only unique elements" << std::endl;
      }
   };

   void CheckIfOverlap(RFlat2DMatrix &Tensor1, RFlat2DMatrix &Tensor2)
   {
      std::set<float> result;

      // Call the set_intersection(), which computes the
      // intersection of set1 and set2 and
      // inserts the result into the 'result' set
      std::set<float> set1(Tensor1.fRVec.begin(), Tensor1.fRVec.end());
      std::set<float> set2(Tensor2.fRVec.begin(), Tensor2.fRVec.end());
      std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::inserter(result, result.begin()));
      // std::list<int> result = intersection(allEntries1, allEntries2);

      if (result.size() == 0) {
         std::cout << "No overlap between the tensors" << std::endl;
      } else {
         std::cout << "Intersection between tensors: ";
         for (auto num : result) {
            std::cout << num << " ";
         }
         std::cout << std::endl;
      }
   };

   std::size_t GetNumTrainingChunks() { return fTraining->Chunks; }

   std::size_t GetNumValidationChunks() { return fValidation->Chunks; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_RCHUNKLOADER
