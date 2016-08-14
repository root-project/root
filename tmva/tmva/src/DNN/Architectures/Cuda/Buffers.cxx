// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 07/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////
// Implementation of device and host buffers for CUDA architectures.  //
////////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/DataLoader.h"
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Buffers.h"
#include "cuda_runtime.h"
#include <iostream>

namespace TMVA {
namespace DNN  {

//
// TCudaHostBuffer
//______________________________________________________________________________
void TCudaHostBuffer::TDestructor::operator()(CudaDouble_t ** devicePointer)
{
   cudaFreeHost(*devicePointer);
   delete[] devicePointer;
}

//______________________________________________________________________________
TCudaHostBuffer::TCudaHostBuffer(size_t size)
    : fOffset(0), fComputeStream(0), fDestructor()
{
   CudaDouble_t ** pointer = new CudaDouble_t * [1];
   cudaMallocHost(pointer, size * sizeof(CudaDouble_t));
   fDevicePointer = std::shared_ptr<CudaDouble_t *>(pointer, fDestructor);
}

//______________________________________________________________________________
TCudaHostBuffer::operator CudaDouble_t * () const
{
   return *fDevicePointer + fOffset;
}

//______________________________________________________________________________
TCudaHostBuffer TCudaHostBuffer::GetSubBuffer(size_t offset, size_t /*size*/)
{
   TCudaHostBuffer buffer = *this;
   buffer.fOffset         = offset;
   return buffer;
}

//
// TCudaDevicePointer
//______________________________________________________________________________
void TCudaDeviceBuffer::TDestructor::operator()(CudaDouble_t ** devicePointer)
{
   cudaFree(*devicePointer);
   delete[] devicePointer;
}

//______________________________________________________________________________
TCudaDeviceBuffer::TCudaDeviceBuffer(size_t size)
    : fOffset(0), fSize(size), fDestructor()
{
   CudaDouble_t ** pointer = new CudaDouble_t * [1];
   cudaMalloc(pointer, size * sizeof(CudaDouble_t));
   fDevicePointer = std::shared_ptr<CudaDouble_t *>(pointer, fDestructor);
   cudaStreamCreate(&fComputeStream);
}

//______________________________________________________________________________
TCudaDeviceBuffer::TCudaDeviceBuffer(size_t size, cudaStream_t stream)
    : fOffset(0), fSize(size), fComputeStream(stream), fDestructor()
{
   CudaDouble_t ** pointer = new CudaDouble_t * [1];
   cudaMalloc(pointer, size * sizeof(CudaDouble_t));
   fDevicePointer = std::shared_ptr<CudaDouble_t *>(pointer, fDestructor);
}

//______________________________________________________________________________
TCudaDeviceBuffer::TCudaDeviceBuffer(CudaDouble_t * devicePointer,
                                     size_t size,
                                     cudaStream_t stream)
    : fOffset(0), fSize(size), fComputeStream(stream), fDestructor()
{
   CudaDouble_t ** pointer = new CudaDouble_t * [1];
   *pointer       = devicePointer;
   fDevicePointer = std::shared_ptr<CudaDouble_t *>(pointer, fDestructor);
}

//______________________________________________________________________________
TCudaDeviceBuffer TCudaDeviceBuffer::GetSubBuffer(size_t offset, size_t size)
{
   TCudaDeviceBuffer buffer = *this;
   buffer.fOffset           = offset;
   buffer.fSize             = size;
   return buffer;
}

//______________________________________________________________________________
TCudaDeviceBuffer::operator CudaDouble_t * () const
{
    return *fDevicePointer + fOffset;
}

//______________________________________________________________________________
void TCudaDeviceBuffer::CopyFrom(const TCudaHostBuffer &buffer) const
{
   cudaStreamSynchronize(fComputeStream);
   cudaMemcpyAsync(*this, buffer, fSize * sizeof(CudaDouble_t),
                   cudaMemcpyHostToDevice, fComputeStream);
}

//______________________________________________________________________________
void TCudaDeviceBuffer::CopyTo(const TCudaHostBuffer &buffer) const
{
   cudaMemcpyAsync(*this, buffer, fSize * sizeof(CudaDouble_t),
                   cudaMemcpyDeviceToHost, fComputeStream);
   buffer.fComputeStream = fComputeStream;
}

//______________________________________________________________________________
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

//______________________________________________________________________________
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

//______________________________________________________________________________
template<>
void TDataLoader<TMVAInput_t, TCuda>::CopyInput(TCudaHostBuffer & buffer,
                                                IndexIterator_t sampleIterator,
                                                size_t batchSize)
{
   Event * event  = fData.front();
   size_t n  = event->GetNVariables();
   size_t nOutput = (event->GetNTargets() == 0) ? 1 : event->GetNTargets();

   // Copy input variables.

   for (size_t i = 0; i < batchSize; i++) {
      for (size_t j = 0; j < n; j++) {
          size_t sampleIndex = * sampleIterator++;
          event = fData[sampleIndex];
          // Copy input matrices.
          for (size_t j = 0; j < n; j++) {
              size_t bufferIndex = j * batchSize + i;
              buffer[bufferIndex] = event->GetValue(j);
          }
      }
   }
}

//______________________________________________________________________________
template<>
void TDataLoader<TMVAInput_t, TCuda>::CopyOutput(TCudaHostBuffer & buffer,
                                                 IndexIterator_t sampleIterator,
                                                 size_t batchSize)
{
   Event * event  = fData.front();
   size_t n       = (event->GetNTargets() == 0) ? 1 : event->GetNTargets();

   // Copy target(s).

   for (size_t i = 0; i < batchSize; i++) {
      for (size_t j = 0; j < n; j++) {
         size_t sampleIndex = * sampleIterator++;
         event = fData[sampleIndex];
         // Copy input matrices.
         for (size_t j = 0; j < n; j++) {
            size_t bufferIndex = j * batchSize + i;
            if (event->GetNTargets() == 0) {
               buffer[bufferIndex] = (event->GetClass() == 0) ? 1.0 : 0.0;
            } else {
               buffer[bufferIndex] = event->GetTarget(j);
            }
         }
      }
   }
}

template class TDataLoader<MatrixInput_t, TCuda>;
template class TDataLoader<TMVAInput_t, TCuda>;

} // TMVA
} // DNN
