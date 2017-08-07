// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TTensorDataLoader                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Specialization of the Tensor Data Loader Class                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//////////////////////////////////////////////////////////////////////////
// Partial specialization of the TTensorDataLoader class to adapt       //
// it to the TMatrix class. Also the data transfer is kept simple,      //
// since this implementation (being intended as reference and fallback) //
// is not optimized for performance.                                    //
//////////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_REFERENCE_TENSORDATALOADER
#define TMVA_DNN_ARCHITECTURES_REFERENCE_TENSORDATALOADER

#include "TMVA/DNN/TensorDataLoader.h"

namespace TMVA {
namespace DNN {

template <typename AReal>
class TReference;

template <typename AData, typename AReal>
class TTensorDataLoader<AData, TReference<AReal>> {
private:
   using BatchIterator_t = TTensorBatchIterator<AData, TReference<AReal>>;

   const AData &fData; ///< The data that should be loaded in the batches.

   size_t fNSamples;        ///< The total number of samples in the dataset.
   size_t fBatchSize;       ///< The size of a batch.
   size_t fBatchDepth;      ///< The number of matrices in the tensor.
   size_t fBatchHeight;     ///< The number od rows in each matrix.
   size_t fBatchWidth;      ///< The number of columns in each matrix.
   size_t fNOutputFeatures; ///< The number of outputs from the classifier/regressor.
   size_t fBatchIndex;      ///< The index of the batch when there are multiple batches in parallel.

   std::vector<TMatrixT<AReal>> inputTensor; ///< The 3D tensor used to keep the input data.
   TMatrixT<AReal> outputMatrix;             ///< The matrix used to keep the output.
   TMatrixT<AReal> weightMatrix;             ///< The matrix used to keep the batch weights.

   std::vector<size_t> fSampleIndices; ///< Ordering of the samples in the epoch.

public:
   /*! Constructor. */
   TTensorDataLoader(const AData &data, size_t nSamples, size_t batchSize, size_t batchDepth, size_t batchHeight,
                     size_t batchWidth, size_t nOutputFeatures, size_t nStreams = 1);

   TTensorDataLoader(const TTensorDataLoader &) = default;
   TTensorDataLoader(TTensorDataLoader &&) = default;
   TTensorDataLoader &operator=(const TTensorDataLoader &) = default;
   TTensorDataLoader &operator=(TTensorDataLoader &&) = default;

   /** Copy input tensor into the given host buffer. Function to be specialized by
    *  the architecture-specific backend. */
   void CopyTensorInput(std::vector<TMatrixT<AReal>> &tensor, IndexIterator_t sampleIterator);
   /** Copy output matrix into the given host buffer. Function to be specialized
    * by the architecture-spcific backend. */
   void CopyTensorOutput(TMatrixT<AReal> &matrix, IndexIterator_t sampleIterator);
   /** Copy weight matrix into the given host buffer. Function to be specialized
    * by the architecture-spcific backend. */
   void CopyTensorWeights(TMatrixT<AReal> &matrix, IndexIterator_t sampleIterator);

   BatchIterator_t begin() { return BatchIterator_t(*this); }
   BatchIterator_t end() { return BatchIterator_t(*this, fNSamples / fBatchSize); }

   /** Shuffle the order of the samples in the batch. The shuffling is indirect,
    *  i.e. only the indices are shuffled. No input data is moved by this
    * routine. */
   void Shuffle();

   /** Return the next batch from the training set. The TTensorDataLoader object
    *  keeps an internal counter that cycles over the batches in the training
    *  set. */
   TTensorBatch<TReference<AReal>> GetTensorBatch();
};

//
// TTensorDataLoader Class.
//______________________________________________________________________________
template <typename AData, typename AReal>
TTensorDataLoader<AData, TReference<AReal>>::TTensorDataLoader(const AData &data, size_t nSamples, size_t batchSize,
                                                               size_t batchDepth, size_t batchHeight, size_t batchWidth,
                                                               size_t nOutputFeatures, size_t /* nStreams */)
   : fData(data), fNSamples(nSamples), fBatchSize(batchSize), fBatchDepth(batchDepth), fBatchHeight(batchHeight),
     fBatchWidth(batchWidth), fNOutputFeatures(nOutputFeatures), fBatchIndex(0), inputTensor(),
     outputMatrix(batchSize, nOutputFeatures), weightMatrix(batchSize, 1), fSampleIndices()
{
   fSampleIndices.reserve(fNSamples);
   for (size_t i = 0; i < fNSamples; i++) {
      fSampleIndices.push_back(i);
   }
}

template <typename AData, typename AReal>
void TTensorDataLoader<AData, TReference<AReal>>::Shuffle()
{
   std::random_shuffle(fSampleIndices.begin(), fSampleIndices.end());
}

template <typename AData, typename AReal>
auto TTensorDataLoader<AData, TReference<AReal>>::GetTensorBatch() -> TTensorBatch<TReference<AReal>>
{
   fBatchIndex %= (fNSamples / fBatchSize); // Cycle through samples.

   size_t sampleIndex = fBatchIndex * fBatchSize;
   IndexIterator_t sampleIndexIterator = fSampleIndices.begin() + sampleIndex;

   CopyInput(inputTensor, sampleIndexIterator);
   CopyOutput(outputMatrix, sampleIndexIterator);
   CopyWeights(weightMatrix, sampleIndexIterator);

   fBatchIndex++;
   return TTensorBatch<TReference<AReal>>(inputTensor, outputMatrix, weightMatrix);
}

} // namespace DNN
} // namespace TMVA

#endif