#include "ROOT/ML/RChunkConstructor.hxx"

#include <numeric>

namespace ROOT::Experimental::Internal::ML {

RChunkConstructor::RChunkConstructor(const std::size_t numEntries, const std::size_t chunkSize,
                                     const std::size_t blockSize)
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
void RChunkConstructor::DistributeBlockIntervals()
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
void RChunkConstructor::CreateChunksIntervals()
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

      ChunksIntervals[j].insert(ChunksIntervals[j].end(), FullBlockIntervalsInLeftoverChunks.begin() + start_FullBlock,
                                FullBlockIntervalsInLeftoverChunks.begin() + end_FullBlock);
      ChunksIntervals[j].insert(ChunksIntervals[j].end(),
                                LeftoverBlockIntervalsInLeftoverChunks.begin() + start_LeftoverBlock,
                                LeftoverBlockIntervalsInLeftoverChunks.begin() + end_LeftoverBlock);
   }
}

//////////////////////////////////////////////////////////////////////////
/// \brief Fills a vector with the size of every chunk from the dataset
void RChunkConstructor::SizeOfChunks()
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
} // namespace ROOT::Experimental::Internal::ML
