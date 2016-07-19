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
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"

namespace TMVA {
namespace DNN  {

// Static members.
//____________________________________________________________________________
size_t          TCudaMatrix::fInstances     = 0;
cublasHandle_t  TCudaMatrix::fCublasHandle  = nullptr;
CudaDouble_t  * TCudaMatrix::fDeviceReturn  = nullptr;
cudaStream_t    TCudaMatrix::fComputeStream = 0;
curandState_t * TCudaMatrix::fCurandStates  = nullptr;
size_t          TCudaMatrix::fNCurandStates = 0;

// Constructors.
//____________________________________________________________________________
TCudaMatrix::TCudaMatrix()
    : fNRows(0), fNCols(0), fDeviceData(nullptr), fDataStream(0), fOwner(true)
{
   InitializeCuda();
}

//____________________________________________________________________________
TCudaMatrix::TCudaMatrix(size_t m, size_t n)
    : fNRows(m), fNCols(n), fDataStream(0), fOwner(true)
{
   InitializeCuda();
   CUDACHECK(cudaMalloc(&fDeviceData, fNRows * fNCols * sizeof(CudaDouble_t)));
}

//____________________________________________________________________________
TCudaMatrix::TCudaMatrix(TCudaMatrix && A)
    : fNRows(A.fNRows), fNCols(A.fNCols), fDataStream(0), fOwner(true)
{
   fDeviceData   = A.fDeviceData;
   A.fDeviceData = nullptr;
}

//____________________________________________________________________________
TCudaMatrix::TCudaMatrix(const TMatrixT<CudaDouble_t> & Host)
    : fNRows(Host.GetNrows()), fNCols(Host.GetNcols()), fDataStream(0), fOwner(true)
{
   InitializeCuda();
   cudaMalloc(&fDeviceData, fNRows * fNCols * sizeof(CudaDouble_t));

   CudaDouble_t * buffer = new CudaDouble_t[fNRows * fNCols];
   size_t index = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         buffer[index] = Host(i, j);
         index++;
      }
   }

   cudaMemcpy(fDeviceData, buffer, fNRows * fNCols * sizeof(CudaDouble_t),
              cudaMemcpyHostToDevice);
}

//____________________________________________________________________________
TCudaMatrix::TCudaMatrix(CudaDouble_t * deviceData,
                        size_t m, size_t n,
                        cudaStream_t dataStream)
    : fDeviceData(deviceData), fNRows(m), fNCols(n), fDataStream(dataStream),
      fOwner(false)
{
   InitializeCuda();
}

//____________________________________________________________________________
inline void TCudaMatrix::InitializeCuda()
{
   if (fInstances == 0) {
       CUDACHECK(cudaStreamCreate(&fComputeStream));
       cublasCreate(&fCublasHandle);
       cublasSetStream(fCublasHandle, fComputeStream);
       CUDACHECK(cudaMalloc(& fDeviceReturn, sizeof(CudaDouble_t)));
       CUDACHECK(cudaMalloc(& fCurandStates, TDevice::NThreads(*this)));
   }
   if (TDevice::NThreads(*this) > (int) fNCurandStates) {
       fNCurandStates = TDevice::NThreads(*this);
       if (fCurandStates) {
           cudaFree(fCurandStates);
       }
       cudaMalloc(&fCurandStates, TDevice::NThreads(*this) * sizeof(curandState_t));
       InitializeCurandStates();
   }
   fInstances++;
}

//____________________________________________________________________________
void TCudaMatrix::InitializeCurandStates()
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(*this);
   ::TMVA::DNN::Cuda::InitializeCurandStates<<<gridDims, blockDims>>>(time(nullptr),
                                                                      fCurandStates);

}


// Conversion to TMatrixT.
//____________________________________________________________________________
TCudaMatrix::operator TMatrixT<CudaDouble_t>() const
{
   TMatrixT<CudaDouble_t> hostMatrix(GetNrows(), GetNcols());

   CudaDouble_t * buffer = new CudaDouble_t[fNRows * fNCols];
   cudaMemcpy(buffer, fDeviceData, fNRows * fNCols * sizeof(CudaDouble_t),
              cudaMemcpyDeviceToHost);

   size_t index = 0;
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         hostMatrix(i, j) = buffer[index];
         index++;
      }
   }

   return hostMatrix;
}

} // namespace DNN
} // namespace TMVA
