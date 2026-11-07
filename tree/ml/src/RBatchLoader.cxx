
#include "RBatchLoader.hxx"

#include <algorithm>
#include <numeric>
#include <utility>

#include "RFlat2DMatrixOperators.hxx"

namespace ROOT::Experimental::Internal::ML {

RBatchLoader::RBatchLoader(std::size_t batchSize, const std::vector<std::string> &cols, std::mutex &sharedMutex,
                           std::condition_variable &sharedCV, const std::vector<std::size_t> &vecSizes,
                           std::size_t numEntries, bool dropRemainder)
   : fBatchSize(batchSize),
     fCols(cols),
     fLock(sharedMutex),
     fCV(sharedCV),
     fVecSizes(vecSizes),
     fNumEntries(numEntries),
     fDropRemainder(dropRemainder)
{
   fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);
   fNumColumns = fCols.size() + fSumVecSizes - fVecSizes.size();

   RecalculateBatchCounts(numEntries);

   fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
}

/// \brief Activate the batchloader. This means that batches can be created and loaded.
void RBatchLoader::Activate()
{
   {
      std::lock_guard<std::mutex> lock(fLock);
      if (fIsActive)
         return;
      fIsActive = true;
      fProducerDone = false;
   }

   fCV.notify_all();
}

/// \brief DeActivate the batchloader. This means that no more batches are created.
/// Batches can still be returned if they are already loaded.
void RBatchLoader::DeActivate()
{
   {
      std::lock_guard<std::mutex> lock(fLock);
      if (!fIsActive)
         return;
      fIsActive = false;
   }

   fCV.notify_all();
}

/// \brief Reset the batchloader state.
void RBatchLoader::Reset()
{
   {
      std::lock_guard<std::mutex> lock(fLock);

      while (!fBatchQueue.empty()) {
         fBatchQueue.pop();
      }

      fCurrentBatch.reset();
      fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
   }

   fCV.notify_all();
}

/// \brief Signal that the producer has finished pushing all batches for this epoch.
void RBatchLoader::MarkProducerDone()
{
   fProducerDone = true;
   fCV.notify_all();
}

/// \brief Return a batch of data as a unique pointer.
/// After the batch has been processed, it should be destroyed.
/// \param[in] chunkTensor Tensor with the data from the chunk
/// \param[in] permutation Order in which to read the rows of the chunk, empty to read them in order
/// \param[in] firstRow First row of the batch in the chunk
/// \return Batch
std::unique_ptr<RFlat2DMatrix> RBatchLoader::CreateBatch(const RFlat2DMatrix &chunkTensor,
                                                         const std::vector<std::size_t> &permutation,
                                                         std::size_t firstRow)
{
   auto batch = std::make_unique<RFlat2DMatrix>();
   batch->Reserve(fBatchSize, fNumColumns);
   RFlat2DMatrixOperators::AppendRows(*batch, chunkTensor, permutation, firstRow, firstRow + fBatchSize);

   return batch;
}

/// \brief Loading the batch from the queue.
/// \return Batch
RFlat2DMatrix RBatchLoader::GetBatch()
{
   std::unique_lock<std::mutex> lock(fLock);

   // Wait until:
   //  - there is data in the queue
   //  - or producer declares "done"
   //  - or we are deactivated
   fCV.wait(lock, [&] { return !fBatchQueue.empty() || fProducerDone || !fIsActive; });

   if (fBatchQueue.empty()) {
      // producer done and no queued data -> end-of-epoch signal
      fCurrentBatch = std::make_unique<RFlat2DMatrix>();
      return *fCurrentBatch;
   }

   fCurrentBatch = std::move(fBatchQueue.front());
   fBatchQueue.pop();
   // Notify the loading thread that the queue has drained
   fCV.notify_all();

   // move the buffer over to the caller
   return std::move(*fCurrentBatch);
}

/// \brief Creating the batches from a chunk and add them to the queue.
/// \param[in] chunkTensor Tensor with the data from the chunk
/// \param[in] permutation Order in which to read the rows of the chunk, empty to read them in order
/// \param[in] beginRow First row of the chunk to create batches from
/// \param[in] endRow End of the rows of the chunk to create batches from
/// \param[in] isLastBatch Check if the batch in the chunk is the last one
void RBatchLoader::CreateBatches(const RFlat2DMatrix &chunkTensor, const std::vector<std::size_t> &permutation,
                                 std::size_t beginRow, std::size_t endRow, bool isLastBatch)
{
   std::size_t chunkSize = endRow - beginRow;
   std::size_t numBatches = chunkSize / fBatchSize;

   // create a vector of batches
   std::vector<std::unique_ptr<RFlat2DMatrix>> batches;

   // fill the full batches from the chunk into a vector
   for (std::size_t i = 0; i < numBatches; i++) {
      batches.emplace_back(CreateBatch(chunkTensor, permutation, beginRow + i * fBatchSize));
   }

   // append the remaining entries of the chunk to fPrimaryLeftoverBatch, and add it to the batches once it is full
   for (std::size_t row = beginRow + numBatches * fBatchSize; row < endRow;) {
      const std::size_t numRows = std::min(endRow - row, fBatchSize - fPrimaryLeftoverBatch->GetRows());
      fPrimaryLeftoverBatch->Reserve(fBatchSize, fNumColumns);
      RFlat2DMatrixOperators::AppendRows(*fPrimaryLeftoverBatch, chunkTensor, permutation, row, row + numRows);
      row += numRows;

      if (fPrimaryLeftoverBatch->GetRows() == fBatchSize) {
         batches.emplace_back(std::move(fPrimaryLeftoverBatch));
         fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
      }
   }

   // add the content of fPrimaryLeftoverBatch as the leftover batch of the last chunk
   if (isLastBatch) {
      if (!fDropRemainder && fLeftoverBatchSize > 0) {
         fPrimaryLeftoverBatch->Resize(fLeftoverBatchSize, fNumColumns);
         batches.emplace_back(std::move(fPrimaryLeftoverBatch));
      }

      fPrimaryLeftoverBatch = std::make_unique<RFlat2DMatrix>();
   }

   {
      std::lock_guard<std::mutex> lock(fLock);
      for (auto &batch : batches) {
         fBatchQueue.push(std::move(batch));
      }
   }

   fCV.notify_all();
}

/// \brief Recalculate batch counts from the given number of entries.
/// Used at construction or when the true entry count is discovered lazily (filtered case).
void RBatchLoader::RecalculateBatchCounts(std::size_t numEntries)
{
   fNumEntries = numEntries;

   if (fBatchSize == 0) {
      fBatchSize = fNumEntries;
   }

   fLeftoverBatchSize = fNumEntries % fBatchSize;
   fNumFullBatches = fNumEntries / fBatchSize;

   const std::size_t numLeftoverBatches = fLeftoverBatchSize == 0 ? 0 : 1;
   fNumBatches = fDropRemainder ? fNumFullBatches : fNumFullBatches + numLeftoverBatches;
}
} // namespace ROOT::Experimental::Internal::ML
