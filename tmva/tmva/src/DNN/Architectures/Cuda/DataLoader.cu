// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////
// Implementation of the data loader for Cuda architectures. //
///////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/DataLoader.h"

namespace TMVA {
namespace DNN  {


// Inline Function Bodies
//____________________________________________________________________________



// TCudaBatchIterator
//____________________________________________________________________________
template <typename Data_t>
TCudaBatchIterator<Data_t>::TCudaBatchIterator(TCudaDataLoader<Data_t> &dataLoader,
                                             IndexIterator_t sampleIndexIterator,
                                             IndexIterator_t sampleIndexIteratorEnd)
  : fDataLoader(dataLoader),
    fSampleIndexIterator(sampleIndexIterator),
    fSampleIndexIteratorEnd(sampleIndexIteratorEnd),
    fNbatchesInEpoch(dataLoader.GetNBatchesInEpoch()),
    fBatchSize(dataLoader.GetBatchSize()),
    fTransferBatchSize(dataLoader.GetBatchBatchSize())
{
    size_t distance = std::distance(fSampleIndexIterator,
                                    fSampleIndexIteratorEnd);
    fBatchIndex = fNbatchesInEpoch - (distance / fBatchSize);
}

//____________________________________________________________________________
template <typename Data_t>
inline void TCudaBatchIterator<Data_t>::PrepareStream()
{
   for (size_t i = 0; i < fDataLoader.GetPreloadOffset() + 1; i++) {
       TCudaDataLoader<Data_t>::CopyBatches(fDataLoader.GetInputData(),
                                           fSampleIndexIterator,
                                           fSampleIndexIteratorEnd,
                                           fBatchSize,
                                           fTransferBatchSize,
                                           fDataLoader.GetInputTransferBuffer(),
                                           fDataLoader.GetOutputTransferBuffer());
      fDataLoader.InvokeTransfer();
      fSampleIndexIterator += fTransferBatchSize * fBatchSize;
   }
}

//____________________________________________________________________________
template <typename Data_t>
inline TCudaBatchIterator<Data_t> & TCudaBatchIterator<Data_t>::operator++()
{
   fBatchIndex++;
   if ((fBatchIndex % fTransferBatchSize) == 0) {
      if (fBatchIndex < fNbatchesInEpoch) {
          TCudaDataLoader<Data_t>::CopyBatches(fDataLoader.GetInputData(),
                                              fSampleIndexIterator,
                                              fSampleIndexIteratorEnd,
                                              fBatchSize,
                                              fTransferBatchSize,
                                              fDataLoader.GetInputTransferBuffer(),
                                              fDataLoader.GetOutputTransferBuffer());
         fDataLoader.InvokeTransfer();
         fSampleIndexIterator += fTransferBatchSize * fBatchSize;
      }
   }
   return *this;
}

//____________________________________________________________________________
template <typename Data_t>
inline TCudaBatch TCudaBatchIterator<Data_t>::operator*()
{
   return fDataLoader.GetCurrentBatch(fBatchIndex);
}

//____________________________________________________________________________
template <typename Data_t>
inline bool TCudaBatchIterator<Data_t>::operator==(const TCudaBatchIterator & other)
{
   return fBatchIndex == other.fBatchIndex;
}

//____________________________________________________________________________
template <typename Data_t>
inline bool TCudaBatchIterator<Data_t>::operator!=(const TCudaBatchIterator & other)
{
   return fBatchIndex != other.fBatchIndex;
}

// TCudaDataLoader
//____________________________________________________________________________
template <typename Data_t>
TCudaDataLoader<Data_t>::TCudaDataLoader(const Data_t & inputData,
                                       size_t nsamples,
                                       size_t batchSize,
                                       size_t ninputFeatures,
                                       size_t noutputFeatures,
                                       size_t batchBatchSize,
                                       size_t preloadOffset)
: fInputData(inputData), fNsamples(nsamples), fNinputFeatures(ninputFeatures),
  fNoutputFeatures(noutputFeatures), fBatchSize(batchSize),
  fTransferBatchSize(batchBatchSize), fPreloadOffset(preloadOffset),
  fStreamIndex(0), fSampleIndices(nsamples)
{
   fNbatchesInEpoch = fNsamples / fBatchSize;

   fInputMatrixSize  = fNinputFeatures  * fBatchSize;
   fOutputMatrixSize = fNoutputFeatures * fBatchSize;
   fTransferSize     = fTransferBatchSize * (fInputMatrixSize + fOutputMatrixSize);
   fTransferSize    *= sizeof(CudaDouble_t);

   fHostData    = new CudaDouble_t * [fPreloadOffset + 1];
   fDeviceData  = new CudaDouble_t * [fPreloadOffset + 1];
   fDataStreams = new cudaStream_t   [fPreloadOffset + 1];

   for (size_t i = 0; i < fPreloadOffset + 1; i++) {
      CUDACHECK(cudaMallocHost(fHostData + i, fTransferSize));
      CUDACHECK(cudaMalloc(fDeviceData + i,   fTransferSize));
      cudaStreamCreate(fDataStreams + i);
   }

   for (size_t i = 0; i < fNsamples; i++)
       fSampleIndices[i] = i;
}

//____________________________________________________________________________
template <typename Data_t>
TCudaDataLoader<Data_t>::~TCudaDataLoader()
{
   for (size_t i = 0; i < fPreloadOffset + 1; i++) {
      cudaFree(fHostData + i);
      cudaFree(fDeviceData + i);
   }
}

//____________________________________________________________________________
template <typename Data_t>
inline TCudaBatchIterator<Data_t> TCudaDataLoader<Data_t>::begin() {
   std::random_shuffle(fSampleIndices.begin(), fSampleIndices.end());
   TCudaBatchIterator<Data_t> iterator(*this, fSampleIndices.begin(),
                                      fSampleIndices.end());
   iterator.PrepareStream();
   return iterator;
}

//____________________________________________________________________________
template <typename Data_t>
inline TCudaBatchIterator<Data_t> TCudaDataLoader<Data_t>::end() {
   return TCudaBatchIterator<Data_t>(*this, fSampleIndices.end(),
                                    fSampleIndices.end());
}


//____________________________________________________________________________
template <typename Data_t>
inline CudaDouble_t * TCudaDataLoader<Data_t>::GetInputTransferBuffer() const
{
   return fHostData[fStreamIndex];
}

//____________________________________________________________________________
template <typename Data_t>
inline CudaDouble_t * TCudaDataLoader<Data_t>::GetOutputTransferBuffer() const
{
   return fHostData[fStreamIndex] + fTransferBatchSize * fInputMatrixSize;
}

//____________________________________________________________________________
template <typename InputDataType>
void TCudaDataLoader<InputDataType>::InvokeTransfer()
{
    cudaMemcpyAsync(fDeviceData[fStreamIndex],
                    fHostData[fStreamIndex],
                    fTransferSize,
                    cudaMemcpyHostToDevice,
                    fDataStreams[fStreamIndex]);

    // Cycle through buffers and data streams.
    fStreamIndex = (fStreamIndex + 1) % (fPreloadOffset + 1);
}

//____________________________________________________________________________
template <typename Data_t>
auto TCudaDataLoader<Data_t>::GetCurrentBatch(size_t batchIndex)
    -> TCudaBatch
{
    size_t bufferIndex = batchIndex % fTransferBatchSize;
    size_t nextStreamIndex = fStreamIndex;

    CudaDouble_t * inputDataPointer = fDeviceData[nextStreamIndex];
    inputDataPointer  += bufferIndex * fInputMatrixSize;
    CudaDouble_t * outputDataPointer = fDeviceData[nextStreamIndex];
    outputDataPointer += fTransferBatchSize * fInputMatrixSize;
    outputDataPointer += bufferIndex * fOutputMatrixSize;

    cudaStreamSynchronize(fDataStreams[nextStreamIndex]);
    return TCudaBatch(fBatchSize, fNinputFeatures, fNoutputFeatures,
                     inputDataPointer, outputDataPointer, fDataStreams[nextStreamIndex]);
}

//____________________________________________________________________________
template <>
inline void TCudaDataLoader<MatrixInput_t>::CopyBatches(
    MatrixInput_t data,
    IndexIterator_t sampleIndexIteratorBegin,
    IndexIterator_t sampleIndexIteratorEnd,
    size_t batchSize,
    size_t batchBatchSize,
    CudaDouble_t * inputBuffer,
    CudaDouble_t * outputBuffer)
{

   const TMatrixT<Double_t> &inputData  = std::get<0>(data);
   const TMatrixT<Double_t> &outputData = std::get<1>(data);

   size_t n_input  = inputData.GetNcols();
   size_t n_output = outputData.GetNcols();

   // Copy input matrices;

   auto sampleIndexIterator = sampleIndexIteratorBegin;
   size_t bufferOffset = 0;
   for (size_t b = 0; b < batchBatchSize; b++) {
      for (size_t i = 0; i < batchSize; i++) {
         if (sampleIndexIterator < sampleIndexIteratorEnd) {
            size_t sampleIndex = *sampleIndexIterator;
            // Copy input matrices.
            for (size_t j = 0; j < n_input; j++) {
               size_t bufferIndex = bufferOffset + j * batchSize + i;
               inputBuffer[bufferIndex] = inputData(sampleIndex, j);
            }
            sampleIndexIterator++;
         }
      }
      bufferOffset += batchSize * n_input;
   }

   // Copy output matrices;

   sampleIndexIterator = sampleIndexIteratorBegin;
   bufferOffset = 0;
   for (size_t b = 0; b < batchBatchSize; b++) {
      for (size_t i = 0; i < batchSize; i++) {
         if (sampleIndexIterator < sampleIndexIteratorEnd) {
            size_t sampleIndex = *sampleIndexIterator;
            for (size_t j = 0; j < n_output; j++) {
               size_t bufferIndex = bufferOffset + j * batchSize + i;
               outputBuffer[bufferIndex] = outputData(sampleIndex, j);
            }
            sampleIndexIterator++;
         }
      }
      bufferOffset += batchSize * n_output;
   }
}

//____________________________________________________________________________
template <>
inline void TCudaDataLoader<TMVAInput_t>::CopyBatches(
   TMVAInput_t data,
   IndexIterator_t sampleIndexIteratorBegin,
   IndexIterator_t sampleIndexIteratorEnd,
   size_t batchSize,
   size_t batchBatchSize,
   CudaDouble_t * inputBuffer,
   CudaDouble_t * outputBuffer)
{

   Event * event = data.front();
   size_t nInput  = event->GetNVariables();
   size_t nOutput = (event->GetNTargets() == 0) ? 1 : event->GetNTargets();

   // Copy input matrices;

   auto sampleIndexIterator = sampleIndexIteratorBegin;
   size_t bufferOffset = 0;
   for (size_t b = 0; b < batchBatchSize; b++) {
      for (size_t i = 0; i < batchSize; i++) {
         if (sampleIndexIterator < sampleIndexIteratorEnd) {
            size_t sampleIndex = *sampleIndexIterator;
            event = data[sampleIndex];
            // Copy input matrices.
            for (size_t j = 0; j < nInput; j++) {
               size_t bufferIndex = bufferOffset + j * batchSize + i;
               inputBuffer[bufferIndex] = event->GetValue(j);
            }
            sampleIndexIterator++;
         }
      }
      bufferOffset += batchSize * nInput;
   }

   // Copy output matrices;

   sampleIndexIterator = sampleIndexIteratorBegin;
   bufferOffset = 0;
   for (size_t b = 0; b < batchBatchSize; b++) {
      for (size_t i = 0; i < batchSize; i++) {
         if (sampleIndexIterator < sampleIndexIteratorEnd) {
            size_t sampleIndex = *sampleIndexIterator;
            event = data[sampleIndex];
            for (size_t j = 0; j < nOutput; j++) {
               size_t bufferIndex = bufferOffset + j * batchSize + i;
               if (event->GetNTargets() == 0) {
                  outputBuffer[bufferIndex] = (event->GetClass() == 0) ? 1.0 : 0.0;
               } else {
                  outputBuffer[bufferIndex] = event->GetTarget(j);
               }
            }
            sampleIndexIterator++;
         }
      }
      bufferOffset += batchSize * nOutput;
   }
}

// Explicit instantiation.
template class TCudaBatchIterator<MatrixInput_t>;
template class TCudaBatchIterator<TMVAInput_t>;
template class TCudaDataLoader<MatrixInput_t>;
template class TCudaDataLoader<TMVAInput_t>;

template<>
void TDataLoader<MatrixInput_t, TCuda>::CopyInput(TCudaHostBuffer & buffer,
                                                  IndexIterator_t sampleIterator,
                                                  size_t batchSize)
{
   const TMatrixT<Double_t> &inputMatrix  = std::get<0>(fData);
   size_t n = inputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = inputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

template<>
void TDataLoader<MatrixInput_t, TCuda>::CopyOutput(TCudaHostBuffer & buffer,
                                                   IndexIterator_t sampleIterator,
                                                   size_t batchSize)
{
   const TMatrixT<Double_t> &outputMatrix  = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         size_t bufferIndex = j * batchSize + i;
         buffer[bufferIndex] = outputMatrix(sampleIndex, j);
      }
      sampleIterator++;
   }
}

template class TDataLoader<MatrixInput_t, TCuda>;

} // namespace TMVA
} // namespace DNN
