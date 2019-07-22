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
// Implementation of the TCudaTensor class. //
/////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda/CudaTensor.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"

#include <cassert>

namespace TMVA {
namespace DNN  {



// Static members.
//____________________________________________________________________________
template<typename AFloat>
size_t          TCudaTensor<AFloat>::fInstances     = 0;
template<typename AFloat>
cublasHandle_t  TCudaTensor<AFloat>::fCublasHandle  = nullptr;
template<typename AFloat>
AFloat        * TCudaTensor<AFloat>::fDeviceReturn  = nullptr;
//template<typename AFloat>
//AFloat        * TCudaTensor<AFloat>::fOnes          = nullptr;
template<typename AFloat>
curandState_t * TCudaTensor<AFloat>::fCurandStates  = nullptr;
template<typename AFloat>
size_t          TCudaTensor<AFloat>::fNCurandStates = 0;
//template<typename AFloat>
//size_t          TCudaTensor<AFloat>::fNOnes         = 0;

// Constructors.
//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor()
    : fShape(nullptr), fSize(0), fElementBuffer()
{
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(size_t size, size_t dim, const size_t * shape)
    : fNDim(dim), fSize(size), fElementBuffer(size, 0)
{
   InitializeCuda();

   fShape = new size_t[fNDim];
   size_t comp_size = 1; 
   for (int i = 0; i < fNDim; ++i) {
       fShape[i] = shape[i];
       comp_size *= shape[i];  // to check input size
   }  
   assert(fSize == comp_size);
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(size_t size, const AFloat * host_data, size_t dim, const size_t * shape)
    : fNDim(dim), fSize(size) , fElementBuffer(size, 0)
{
   InitializeCuda();

   fShape = new size_t[fNDim];
   size_t comp_size = 1; 
   for (int i = 0; i < fNDim; ++i) {
       fShape[i] = shape[i];
       comp_size *= shape[i];
   }  
   assert(fSize == comp_size);
   
   // do I need to allocate this buffer ???? 
   // is not a mem leak
   // AFloat * buffer = new AFloat[fSize];
   // size_t index = 0;
   // for (size_t j = 0; j < fSize; ++j) {
   //       buffer[j] = static_cast<AFloat>(host_data[j]);
   //    }
   // }

   cudaMemcpy(fElementBuffer, host_data, fSize * sizeof(AFloat),
              cudaMemcpyHostToDevice);
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(TCudaDeviceBuffer<AFloat> buffer, size_t dim, const size_t * shape)
    : fNDim(dim) , fElementBuffer(buffer)
{ 
   fShape = new size_t[fNDim];
   size_t size = 1; 
   for (int i = 0; i < fNDim; ++i) {
      fShape[i] = shape[i];
      size *= shape[i];
   }
   fSize = size;

   InitializeCuda();
}

//____________________________________________________________________________
template <typename AFloat>
inline void TCudaTensor<AFloat>::InitializeCuda()
{
   // add further initialization than done in TMatrixcPU::iNITIALIZEcUDA
   // if (fInstances == 0) {
   //     cublasCreate(&fCublasHandle);
   //     CUDACHECK(cudaMalloc(& fDeviceReturn, sizeof(AFloat)));
   //     CUDACHECK(cudaMalloc(& fCurandStates, TDevice::NThreads(*this)));
   // }
   // if (TDevice::NThreads(*this) > (int) fNCurandStates) {
   //     fNCurandStates = TDevice::NThreads(*this);
   //     if (fCurandStates) {
   //         cudaFree(fCurandStates);
   //     }
   //     cudaMalloc(&fCurandStates, TDevice::NThreads(*this) * sizeof(curandState_t));
   //     InitializeCurandStates();
   // }
  
   fInstances++;
}

//____________________________________________________________________________
template<typename AFloat>
void TCudaTensor<AFloat>::InitializeCurandStates()
{
   // dim3 blockDims = TDevice::BlockDims2D();
   // dim3 gridDims  = TDevice::GridDims2D(*this);
   // CurandInitializationKernel<<<gridDims, blockDims>>>(time(nullptr), fCurandStates);
}

#if 0
// Conversion to RTensor
//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::operator Experimental::RTensor<AFloat>() const
{
   std::vector<size_t> shape(fNDims, fNDims + fDim)
   
   Experimental::RTensor<AFloat> hostTensor( shape)

   AFloat * buffer = new AFloat[fSize];
   cudaMemcpy(buffer, fElementBuffer, fSize * sizeof(AFloat),
              cudaMemcpyDeviceToHost);

   size_t index = 0;
   for (size_t j = 0; j < fSize; j++) {
         hostTensor.GetData()[j] = static_cast<AFloat>(buffer[j]);
      }
   }

   delete[] buffer;
   return hostTensor;
}
#endif
// Explicit Instantiations.

template class TCudaTensor<float>;
template class TCudaTensor<double>;

} // namespace DNN
} // namespace TMVA
