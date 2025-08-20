// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 06/06/17

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Partial specialization of the TDataLoader class to adapt it to  //
// the TMatrix class. Also the data transfer is kept simple, since //
// this implementation (being intended as reference and fallback   //
// is not optimized for performance.                               //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_REFERENCE_DATALOADER
#define TMVA_DNN_ARCHITECTURES_REFERENCE_DATALOADER

#include "TMVA/DNN/DataLoader.h"

#include <random>

namespace TMVA {
namespace DNN {

template <typename AReal>
class TReference;

template <typename AData, typename AReal>
class TDataLoader<AData, TReference<AReal>> {
private:
   using BatchIterator_t = TBatchIterator<AData, TReference<AReal>>;

   const AData &fData;

   size_t fNSamples;
   size_t fBatchSize;
   size_t fNInputFeatures;
   size_t fNOutputFeatures;
   size_t fBatchIndex;

   TMatrixT<AReal> inputMatrix;
   TMatrixT<AReal> outputMatrix;
   TMatrixT<AReal> weightMatrix;

   std::vector<size_t> fSampleIndices; ///< Ordering of the samples in the epoch.

public:
   TDataLoader(const AData &data, size_t nSamples, size_t batchSize, size_t nInputFeatures, size_t nOutputFeatures,
               size_t nthreads = 1);
   TDataLoader(const TDataLoader &) = default;
   TDataLoader(TDataLoader &&) = default;
   TDataLoader &operator=(const TDataLoader &) = default;
   TDataLoader &operator=(TDataLoader &&) = default;

   /** Copy input matrix into the given host buffer. Function to be specialized by
    *  the architecture-specific backend. */
   void CopyInput(TMatrixT<AReal> &matrix, IndexIterator_t begin);
   /** Copy output matrix into the given host buffer. Function to be specialized
    * by the architecture-specific backend. */
   void CopyOutput(TMatrixT<AReal> &matrix, IndexIterator_t begin);
   /** Copy weight matrix into the given host buffer. Function to be specialized
    * by the architecture-specific backend. */
   void CopyWeights(TMatrixT<AReal> &matrix, IndexIterator_t begin);

   BatchIterator_t begin() { return BatchIterator_t(*this); }
   BatchIterator_t end() { return BatchIterator_t(*this, fNSamples / fBatchSize); }

   /** Shuffle the order of the samples in the batch. The shuffling is indirect,
    *  i.e. only the indices are shuffled. No input data is moved by this
    * routine. */
   void Shuffle();

   /** Return the next batch from the training set. The TDataLoader object
    *  keeps an internal counter that cycles over the batches in the training
    *  set. */
   TBatch<TReference<AReal>> GetBatch();
};

template <typename AData, typename AReal>
TDataLoader<AData, TReference<AReal>>::TDataLoader(const AData &data, size_t nSamples, size_t batchSize,
                                                   size_t nInputFeatures, size_t nOutputFeatures, size_t /*nthreads*/)
   : fData(data), fNSamples(nSamples), fBatchSize(batchSize), fNInputFeatures(nInputFeatures),
     fNOutputFeatures(nOutputFeatures), fBatchIndex(0), inputMatrix(batchSize, nInputFeatures),
     outputMatrix(batchSize, nOutputFeatures), weightMatrix(batchSize, 1), fSampleIndices()
{
   fSampleIndices.reserve(fNSamples);
   for (size_t i = 0; i < fNSamples; i++) {
      fSampleIndices.push_back(i);
   }
}

template <typename AData, typename AReal>
TBatch<TReference<AReal>> TDataLoader<AData, TReference<AReal>>::GetBatch()
{
   fBatchIndex %= (fNSamples / fBatchSize); // Cycle through samples.

   size_t sampleIndex = fBatchIndex * fBatchSize;
   IndexIterator_t sampleIndexIterator = fSampleIndices.begin() + sampleIndex;

   CopyInput(inputMatrix, sampleIndexIterator);
   CopyOutput(outputMatrix, sampleIndexIterator);
   CopyWeights(weightMatrix, sampleIndexIterator);

   fBatchIndex++;

   return TBatch<TReference<AReal>>(inputMatrix, outputMatrix, weightMatrix);
}

//______________________________________________________________________________
template <typename AData, typename AReal>
void TDataLoader<AData, TReference<AReal>>::Shuffle()
{
   std::shuffle(fSampleIndices.begin(), fSampleIndices.end(), std::default_random_engine{});
}

} // namespace DNN
} // namespace TMVA

#endif // TMVA_DNN_ARCHITECTURES_REFERENCE_DATALOADER
