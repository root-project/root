// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////
// Declaration of the TCudaDataLoader class, which implements a        //
// prefetching data loader for CUDA architecture. Also contains       //
// the TCudaBatch class and the TCudaBatchIterator representing batches //
// and an iterator over batches in a data set for CUDA architectures  //
////////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_DATA
#define TMVA_DNN_ARCHITECTURES_CUDA_DATA

#include "CudaMatrix.h"
#include "TMVA/Event.h"
#include <algorithm>
#include <iterator>

namespace TMVA {
namespace DNN  {

// Input Data Types
using MatrixInput_t    = std::pair<const TMatrixT<Double_t> &,
                                   const TMatrixT<Double_t> &>;
using TMVAInput_t      = std::vector<Event*>;

using IndexIterator_t = typename std::vector<size_t>::iterator;

/** TCudaBatch class.
 *
 * Lightweight representation of a batch of data on the device. Holds
 * pointer to the input (samples) and output data (labels) as well as
 * to the data stream in which this batch is transferred.
 *
 * Provides GetInput() and GetOutput() member functions that return
 * TCudaMatrix representations of the input and output data in the
 * batch.
 */
class TCudaBatch
{
private:
    cudaStream_t   fDataStream;       ///< Cuda stream in which the data is transferred
    CudaDouble_t * fInputMatrixData;  ///< Pointer to the input data buffer.
    CudaDouble_t * fOutputMatrixData; ///< Pointer to the ouput data buffer.

    size_t fNinputFeatures;  ///< Number of input features.
    size_t fNoutputFeatures; ///< Number of output features.
    size_t fBatchSize;       ///< Size of the batch.
public:

    TCudaBatch(size_t batchSize,
        size_t ninputFeatures,
        size_t noutputFeatures,
        CudaDouble_t * inputMatrixData,
        CudaDouble_t * outputMatrixData,
        cudaStream_t dataStream)
    : fDataStream(dataStream), fInputMatrixData(inputMatrixData),
    fOutputMatrixData(outputMatrixData), fNinputFeatures(ninputFeatures),
    fNoutputFeatures(noutputFeatures), fBatchSize(batchSize)
    {
        // Nothing to do here.
    }

    /** Return the batch input data as a TCudaMatrix. The TCudaMatrix is passed
     *  the data stream in which the async. data transfer to the corresponding
     *  buffer is performed, so that operations on the matrix can synchronize
     *  with it. */
    TCudaMatrix GetInput()
    {
        return TCudaMatrix(fInputMatrixData, fBatchSize, fNinputFeatures,
                        fDataStream);
    }

    /** Return the outpur data as a TCudaMatrix. Also forwards the data stream in
     *  which the async. data transfer is performed to the matrix. See above.
     */
    TCudaMatrix GetOutput()
    {
        return TCudaMatrix(fOutputMatrixData, fBatchSize, fNoutputFeatures,
                        fDataStream);
    }
};

template<typename Data_t>
class TCudaDataLoader;

/** TCudaBatchIterator Class
 *
 * Class that implements an iterator over data sets on a CUDA device. The
 * batch iterator has to take care of the preloading of the data which is
 * why a special implementation is required.
 */
template <typename Data_t>
class TCudaBatchIterator
{
private:
    using SampleIndexIterator_t = typename std::vector<size_t>::iterator;

    TCudaDataLoader<Data_t>  & fDataLoader;   ///< Dataloader managing data transfer.
    IndexIterator_t fSampleIndexIterator;    ///< Sample indices in this batch.
    IndexIterator_t fSampleIndexIteratorEnd; ///< End of this batch.

    const size_t fNbatchesInEpoch;    ///< Total number of batches in the data set.
    const size_t fBatchSize;          ///< Size of this batch.
    const size_t fTransferBatchSize;  ///< No. of. batches in a data transfer batch.
          size_t fBatchIndex;         ///< Index of this batch in the current epoch.

public:

    TCudaBatchIterator(TCudaDataLoader<Data_t> &dataLoader,
                      IndexIterator_t sampleIndexIterator,
                      IndexIterator_t sampleIndexIteratorEnd);

    /** Preloads the number of transfer batches as specified by the
     *  corresponding TCudaDataLoader object. */
    void PrepareStream();
    /** Advance to the next batch and check if data should be preloaded. */
    TCudaBatchIterator & operator++();
    /** Return TCudaBatch object corresponding to the current iterator position. */
    TCudaBatch operator*();

    bool operator==(const TCudaBatchIterator & other);
    bool operator!=(const TCudaBatchIterator & other);
};

/** The TCudaDataLoader Class
 *
 * The TCudaDataLoader class takes care of transferring training and test data
 * from the host to the device. The data transfer is performed asynchronously
 * and multiple data set batches can be transferred combined into transfer batches,
 * which contain a fixed number of data set batches. The range of the preloading
 * is defined in multiples of transfer batches. */
template <typename Data_t>
class TCudaDataLoader
{
private:

   const Data_t & fInputData;

   size_t fNsamples;          ///< No. of samples in the data set.
   size_t fNinputFeatures;    ///< No. of features in input sample.
   size_t fNoutputFeatures;   ///< No. of features in output sample (truth).
   size_t fBatchSize;         ///< No. of samples in a (mini-)batch
   size_t fTransferBatchSize; ///< No. of batches combined in a single data transfer.
   size_t fPreloadOffset;     ///< How many batch-batches data is loaded in advance.

   size_t fNbatchesInEpoch; ///< No. of batches in one epoch.
   size_t fInputMatrixSize; ///< No. of elements in input matrix.
   size_t fOutputMatrixSize;///< No. of elements in output matrix.
   size_t fTransferSize;    ///< Total size of one  data transfer in bytes;

   size_t fStreamIndex; ///< Transfer stream index.

   CudaDouble_t **fHostData;    ///< Host-side buffer array.
   CudaDouble_t **fDeviceData;  ///< Device-side buffer array.
   cudaStream_t * fDataStreams; ///< Data stream array.

   std::vector<size_t> fSampleIndices; ///< Shuffled sample indices.

public:

   TCudaDataLoader(const Data_t & inputData,
                  size_t nsamples,
                  size_t batchSize,
                  size_t ninputFeatures,
                  size_t noutputFeatures,
                  size_t batchBatchSize = 5,
                  size_t preloadOffset  = 2);

   ~TCudaDataLoader();

   /** Return iterator to batches in the training set. Samples in batches are
    *  are sampled randomly from the data set without replacement. */
   TCudaBatchIterator<Data_t> begin();

   TCudaBatchIterator<Data_t> end();

   /** Called by the iterator to indicate that the current buffer containing a
    *  transfer batch of batchs has been consumed. */
   void NextBuffer() {fStreamIndex = (fStreamIndex + 1) % (fTransferBatchSize + 1);}
   /** Invoke transfer of the current buffer in the buffer cycle. */
   void InvokeTransfer();
   /** Return the batch object corresponding to the given batchIndex in the current
    *  epoch. */
   TCudaBatch GetCurrentBatch(size_t batchIndex);

   /** Copy data from the input data object to the pinned transfer buffer on
    *  the host. Must be specialized for any type of input data that is used. */
   inline static void CopyBatches(Data_t data,
                                  IndexIterator_t sampleIndexIteratorBegin,
                                  IndexIterator_t sampleIndexIteratorEnd,
                                  size_t batchSize,
                                  size_t batchBatchSize,
                                  CudaDouble_t * inputBuffer,
                                  CudaDouble_t * outputBuffer);

   size_t GetNBatchesInEpoch() const {return fNbatchesInEpoch;}
   size_t GetPreloadOffset()   const {return fPreloadOffset;}
   size_t GetBatchSize()       const {return fBatchSize;}
   size_t GetBatchBatchSize()  const {return fTransferBatchSize;}
   const Data_t & GetInputData() const {return fInputData;}
   inline CudaDouble_t * GetInputTransferBuffer()  const;
   inline CudaDouble_t * GetOutputTransferBuffer() const;
};

} // namespace TMVA
} // namespace DNN

#endif
