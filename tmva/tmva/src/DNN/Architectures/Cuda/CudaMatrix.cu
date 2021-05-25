// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////
// Implementation of the TCudaMatrix class. //
/////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda/CudaMatrix.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"

#include <iostream>

namespace TMVA {
namespace DNN  {


//____________________________________________________________________________
__global__ void CurandInitializationKernel(unsigned long long seed,
                                           curandState_t *state)
{
   int i   = blockDim.y * blockIdx.y + threadIdx.y;
   int j   = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = i * gridDim.x + j;
   curand_init(seed + tid, 0, tid, state + tid);
}

// Static members.
//____________________________________________________________________________
template<typename AFloat>
size_t          TCudaMatrix<AFloat>::fInstances     = 0;
template<typename AFloat>
cublasHandle_t  TCudaMatrix<AFloat>::fCublasHandle  = nullptr;
template<typename AFloat>
AFloat        * TCudaMatrix<AFloat>::fDeviceReturn  = nullptr;
template<typename AFloat>
AFloat        * TCudaMatrix<AFloat>::fOnes          = nullptr;
template<typename AFloat>
curandState_t * TCudaMatrix<AFloat>::fCurandStates  = nullptr;
template<typename AFloat>
size_t          TCudaMatrix<AFloat>::fNCurandStates = 0;
template<typename AFloat>
size_t          TCudaMatrix<AFloat>::fNOnes         = 0;
template <typename AFloat>
Bool_t TCudaMatrix<AFloat>::gInitializeCurand = kFALSE;

// Constructors.
//____________________________________________________________________________
template<typename AFloat>
TCudaMatrix<AFloat>::TCudaMatrix()
    : fNRows(0), fNCols(0), fElementBuffer()
{
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaMatrix<AFloat>::TCudaMatrix(size_t m, size_t n)
    : fNRows(m), fNCols(n), fElementBuffer(m * n, 0)
{
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaMatrix<AFloat>::TCudaMatrix(const TMatrixT<AFloat> & Host)
    : fNRows(Host.GetNrows()), fNCols(Host.GetNcols()),
      fElementBuffer(Host.GetNoElements(), 0)
{
   InitializeCuda();

   AFloat * buffer = new AFloat[fNRows * fNCols];
   size_t index = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         buffer[index] = static_cast<AFloat>(Host(i, j));
         index++;
      }
   }

   cudaMemcpy(fElementBuffer, buffer, fNRows * fNCols * sizeof(AFloat),
              cudaMemcpyHostToDevice);
}

//____________________________________________________________________________
template<typename AFloat>
TCudaMatrix<AFloat>::TCudaMatrix(TCudaDeviceBuffer<AFloat> buffer,
                         size_t m, size_t n)
    : fNRows(m), fNCols(n), fElementBuffer(buffer)
{
   InitializeCuda();
}

//____________________________________________________________________________
template <typename AFloat>
inline void TCudaMatrix<AFloat>::InitializeCuda()
{
   if (fInstances == 0) {
       cublasCreate(&fCublasHandle);
       CUDACHECK(cudaMalloc(& fDeviceReturn, sizeof(AFloat)));
       CUDACHECK(cudaMalloc(& fCurandStates, TDevice::NThreads(*this)));
   }
   if (gInitializeCurand && TDevice::NThreads(*this) > (int) fNCurandStates) {
       fNCurandStates = TDevice::NThreads(*this);
       if (fNCurandStates > 10000000)
          std::cout << "***** Warning - initialize a BIG curandstate for matrix " << fNRows << "," << fNCols << " nstate "
                    << fNCurandStates << std::endl;
       //R__ASSERT( fNRows*fNCols <= 8*8*128*128);
       if (fCurandStates) {
          cudaFree(fCurandStates);
       }
       cudaMalloc(&fCurandStates, TDevice::NThreads(*this) * sizeof(curandState_t));
       InitializeCurandStates();
   }
   if (fNRows >  fNOnes) {
      fNOnes = fNRows;
      if (fOnes) {
         cudaFree(fOnes);
      }
      cudaMalloc(&fOnes, fNRows * sizeof(AFloat));
      AFloat * buffer = new AFloat[fNRows];
      for (size_t i = 0; i < fNRows; i++) {
         buffer[i] = 1.0;
      }
      cudaMemcpy(fOnes, buffer, fNRows * sizeof(AFloat),
                 cudaMemcpyHostToDevice);
   }
   fInstances++;
}

//____________________________________________________________________________
template<typename AFloat>
void TCudaMatrix<AFloat>::InitializeCurandStates()
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(*this);
   CurandInitializationKernel<<<gridDims, blockDims>>>(time(nullptr), fCurandStates);
}

// Conversion to TMatrixT.
//____________________________________________________________________________
template<typename AFloat>
TCudaMatrix<AFloat>::operator TMatrixT<AFloat>() const
{
   TMatrixT<AFloat> hostMatrix(GetNrows(), GetNcols());

   AFloat * buffer = new AFloat[fNRows * fNCols];
   cudaMemcpy(buffer, fElementBuffer, fNRows * fNCols * sizeof(AFloat),
              cudaMemcpyDeviceToHost);

   size_t index = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         hostMatrix(i, j) = static_cast<Double_t>(buffer[index]);
         index++;
      }
   }

   delete[] buffer;
   return hostMatrix;
}

// Explicit Instantiations.

template class TCudaMatrix<float>;
template class TCudaMatrix<double>;

} // namespace DNN
} // namespace TMVA
