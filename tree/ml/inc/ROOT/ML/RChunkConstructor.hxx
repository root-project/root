// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RCHUNKCONSTRUCTOR
#define ROOT_INTERNAL_ML_RCHUNKCONSTRUCTOR

#include <utility>
#include <vector>

#include "Rtypes.h"

namespace ROOT::Experimental::Internal::ML {
/**
\class ROOT::Experimental::Internal::ML::RChunkConstructor

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

The blocks are defined by their start and end entries, which correspond to positions within the dataset’s total number
of entries.
*/

struct RChunkConstructor {
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

   RChunkConstructor(const std::size_t numEntries, const std::size_t chunkSize, const std::size_t blockSize);

   void DistributeBlockIntervals();
   void CreateChunksIntervals();
   void SizeOfChunks();
};
} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RCHUNKCONSTRUCTOR
