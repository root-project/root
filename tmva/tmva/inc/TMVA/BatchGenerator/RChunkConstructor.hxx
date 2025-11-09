// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RCHUNKCONSTRUCTOR
#define TMVA_RCHUNKCONSTRUCTOR

#include <vector>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RChunkConstructor
\ingroup tmva
\brief The logic for constructing chunks from a dataset.

This struct handles the logic for splitting a dataset into smaller subsets 
known as chunks, which are constructed from blocks.
 
A chunk is the largest portion of the dataset loaded into memory at once, 
and each chunk is further divided into batches for machine learning training.
 
The dataset is split into disjoint chunks based on a user-defined chunk size.
There are two types of chunks:
 - Full chunks: contain exactly the number of entries specified by the chunk size.
 - Leftover chunk: contains any remaining entries that don't make up a full chunk.
 
Each chunk is constructed from blocks based on a user-defined block size.
There are two types of blocks:
 - Full blocks: contain exactly the number of entries specified by the block size.
 - Leftover block: contains any remaining entries that don't make up a full block.

The blocks are defined by their start and end entries, which correspond to positions within the dataset’s total number of entries.
*/

struct RChunkConstructor {
   // clang-format on
   std::size_t fNumEntries{};
   std::size_t fChunkSize{};
   std::size_t fBlockSize{};

   // size of full and leftover chunks
   std::size_t SizeOfFullChunk;
   std::size_t SizeOfLeftoverChunk;

   // size of full and leftover blocks in a full and leftover chunk
   std::size_t SizeOfFullBlockInFullChunk;
   std::size_t SizeOfLeftoverBlockInFullChunk;
   std::size_t SizeOfFullBlockInLeftoverChunk;
   std::size_t SizeOfLeftoverBlockInLeftoverChunk;

   // number of full, leftover and total chunks
   std::size_t FullChunks;
   std::size_t LeftoverChunks;
   std::size_t Chunks;

   // number of full, leftover and total blocks in a full chunk
   std::size_t FullBlocksPerFullChunk;
   std::size_t LeftoverBlocksPerFullChunk;
   std::size_t BlockPerFullChunk;

   // number of full, leftover and total blocks in the leftover chunk
   std::size_t FullBlocksPerLeftoverChunk;
   std::size_t LeftoverBlocksPerLeftoverChunk;
   std::size_t BlockPerLeftoverChunk;

   // total number of full and leftover blocks in the full chunks
   std::size_t FullBlocksInFullChunks;
   std::size_t LeftoverBlocksInFullChunks;

   // total number of full and leftover blocks in the leftover chunks
   std::size_t FullBlocksInLeftoverChunks;
   std::size_t LeftoverBlocksInLeftoverChunks;

   // vector of the different block sizes
   std::vector<std::size_t> SizeOfBlocks;

   // vector with the number of the different block
   std::vector<std::size_t> NumberOfDifferentBlocks;

   // total number of blocks
   std::size_t NumberOfBlocks;

   // pair of start and end entries in the different block types
   std::vector<std::pair<Long_t, Long_t>> BlockIntervals;

   std::vector<std::pair<Long_t, Long_t>> FullBlockIntervalsInFullChunks;
   std::vector<std::pair<Long_t, Long_t>> LeftoverBlockIntervalsInFullChunks;

   std::vector<std::pair<Long_t, Long_t>> FullBlockIntervalsInLeftoverChunks;
   std::vector<std::pair<Long_t, Long_t>> LeftoverBlockIntervalsInLeftoverChunks;

   std::vector<std::vector<std::pair<Long_t, Long_t>>> ChunksIntervals;

   std::vector<std::size_t> ChunksSizes;

   RChunkConstructor(const std::size_t numEntries, const std::size_t chunkSize, const std::size_t blockSize)
      : fNumEntries(numEntries), fChunkSize(chunkSize), fBlockSize(blockSize)
   {
      // size of full and leftover chunks
      SizeOfFullChunk = chunkSize;
      SizeOfLeftoverChunk = fNumEntries % SizeOfFullChunk;

      // size of full and leftover blocks in a full and leftover chunk
      SizeOfFullBlockInFullChunk = blockSize;
      SizeOfLeftoverBlockInFullChunk = SizeOfFullChunk % blockSize;
      SizeOfFullBlockInLeftoverChunk = blockSize;
      SizeOfLeftoverBlockInLeftoverChunk = SizeOfLeftoverChunk % blockSize;

      // number of full, leftover and total chunks
      FullChunks = numEntries / SizeOfFullChunk;
      LeftoverChunks = SizeOfLeftoverChunk == 0 ? 0 : 1;
      Chunks = FullChunks + LeftoverChunks;

      // number of full, leftover and total blocks in a full chunk
      FullBlocksPerFullChunk = SizeOfFullChunk / blockSize;
      LeftoverBlocksPerFullChunk = SizeOfLeftoverBlockInFullChunk == 0 ? 0 : 1;
      BlockPerFullChunk = FullBlocksPerFullChunk + LeftoverBlocksPerFullChunk;

      // number of full, leftover and total blocks in the leftover chunk
      FullBlocksPerLeftoverChunk = SizeOfLeftoverChunk / blockSize;
      LeftoverBlocksPerLeftoverChunk = SizeOfLeftoverBlockInLeftoverChunk == 0 ? 0 : 1;
      BlockPerLeftoverChunk = FullBlocksPerLeftoverChunk + LeftoverBlocksPerLeftoverChunk;

      // total number of full and leftover blocks in the full chunks
      FullBlocksInFullChunks = FullBlocksPerFullChunk * FullChunks;
      LeftoverBlocksInFullChunks = LeftoverBlocksPerFullChunk * FullChunks;

      // total number of full and leftover blocks in the leftover chunks
      FullBlocksInLeftoverChunks = FullBlocksPerLeftoverChunk * LeftoverChunks;
      LeftoverBlocksInLeftoverChunks = LeftoverBlocksPerLeftoverChunk * LeftoverChunks;

      // vector of the different block sizes
      SizeOfBlocks = {SizeOfFullBlockInFullChunk, SizeOfLeftoverBlockInFullChunk, SizeOfFullBlockInLeftoverChunk,
                      SizeOfLeftoverBlockInLeftoverChunk};

      // vector with the number of the different block
      NumberOfDifferentBlocks = {FullBlocksInFullChunks, LeftoverBlocksInFullChunks, FullBlocksInLeftoverChunks,
                                 LeftoverBlocksInLeftoverChunks};

      // total number of blocks
      NumberOfBlocks = std::accumulate(NumberOfDifferentBlocks.begin(), NumberOfDifferentBlocks.end(), 0);
   };

   //////////////////////////////////////////////////////////////////////////
   /// \brief Group the blocks based on the block type (full or leftover) based on the size of the block.
   void DistributeBlockIntervals()
   {

      std::vector<std::vector<std::pair<Long_t, Long_t>> *> TypesOfBlockIntervals = {
         &FullBlockIntervalsInFullChunks, &LeftoverBlockIntervalsInFullChunks, &FullBlockIntervalsInLeftoverChunks,
         &LeftoverBlockIntervalsInLeftoverChunks};

      std::vector<std::size_t> IndexOfDifferentBlocks(NumberOfDifferentBlocks.size());
      std::partial_sum(NumberOfDifferentBlocks.begin(), NumberOfDifferentBlocks.end(), IndexOfDifferentBlocks.begin());
      IndexOfDifferentBlocks.insert(IndexOfDifferentBlocks.begin(), 0);

      for (size_t i = 0; i < TypesOfBlockIntervals.size(); ++i) {
         size_t start = IndexOfDifferentBlocks[i];
         size_t end = IndexOfDifferentBlocks[i + 1];

         TypesOfBlockIntervals[i]->insert(TypesOfBlockIntervals[i]->begin(), BlockIntervals.begin() + start,
                                          BlockIntervals.begin() + end);
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Creates chunks from the dataset consisting of blocks with the begin and end entry. 
   void CreateChunksIntervals()
   {

      ChunksIntervals.resize(Chunks);
      for (size_t i = 0; i < FullChunks; i++) {

         size_t start_FullBlock = FullBlocksPerFullChunk * i;
         size_t end_FullBlock = FullBlocksPerFullChunk * (i + 1);

         size_t start_LeftoverBlock = LeftoverBlocksPerFullChunk * i;
         size_t end_LeftoverBlock = LeftoverBlocksPerFullChunk * (i + 1);

         ChunksIntervals[i].insert(ChunksIntervals[i].end(), FullBlockIntervalsInFullChunks.begin() + start_FullBlock,
                                   FullBlockIntervalsInFullChunks.begin() + end_FullBlock);
         ChunksIntervals[i].insert(ChunksIntervals[i].end(),
                                   LeftoverBlockIntervalsInFullChunks.begin() + start_LeftoverBlock,
                                   LeftoverBlockIntervalsInFullChunks.begin() + end_LeftoverBlock);
      }

      for (size_t i = 0; i < LeftoverChunks; i++) {

         size_t j = i + FullChunks;
         size_t start_FullBlock = FullBlocksPerLeftoverChunk * i;
         size_t end_FullBlock = FullBlocksPerLeftoverChunk * (i + 1);

         size_t start_LeftoverBlock = LeftoverBlocksPerLeftoverChunk * i;
         size_t end_LeftoverBlock = LeftoverBlocksPerLeftoverChunk * (i + 1);

         ChunksIntervals[j].insert(ChunksIntervals[j].end(),
                                   FullBlockIntervalsInLeftoverChunks.begin() + start_FullBlock,
                                   FullBlockIntervalsInLeftoverChunks.begin() + end_FullBlock);
         ChunksIntervals[j].insert(ChunksIntervals[j].end(),
                                   LeftoverBlockIntervalsInLeftoverChunks.begin() + start_LeftoverBlock,
                                   LeftoverBlockIntervalsInLeftoverChunks.begin() + end_LeftoverBlock);
      }
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Fills a vector with the size of every chunk from the dataset 
   void SizeOfChunks()
   {

      for (size_t i = 0; i < Chunks; i++) {
         std::size_t chunkSize = 0;
         for (size_t j = 0; j < ChunksIntervals[i].size(); j++) {
            std::size_t start = ChunksIntervals[i][j].first;
            std::size_t end = ChunksIntervals[i][j].second;

            std::size_t intervalSize = end - start;
            chunkSize += intervalSize;
         }

         ChunksSizes.insert(ChunksSizes.end(), chunkSize);
      }
   }
};
} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RCHUNKCONSTRUCTOR
