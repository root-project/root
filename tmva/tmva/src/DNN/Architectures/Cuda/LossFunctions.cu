// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Implementation of the loss functions for the TCuda implementation //
// of the low-level interface.                                       //
///////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
template<typename AFloat>
AFloat TCuda<AFloat>::MeanSquaredError(const TCudaMatrix<AFloat> & Y,
                                       const TCudaMatrix<AFloat> & output,
                                       const TCudaMatrix<AFloat> & weights)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(Y);
    cudaStream_t s = Y.GetComputeStream();
    TCudaMatrix<AFloat>::ResetDeviceReturn();
    ::TMVA::DNN::Cuda::MeanSquaredError<<<gridDims, blockDims, 0, s>>>(
        TCudaMatrix<AFloat>::GetDeviceReturnPointer(),
        Y.GetDataPointer(),
        output.GetDataPointer(),
        weights.GetDataPointer(),
        (int) Y.GetNrows(),
        (int) Y.GetNcols());
    return TCudaMatrix<AFloat>::GetDeviceReturn();
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::MeanSquaredErrorGradients(TCudaMatrix<AFloat> & dY,
                                              const TCudaMatrix<AFloat> & Y,
                                              const TCudaMatrix<AFloat> & output,
                                              const TCudaMatrix<AFloat> &weights)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(Y);
   cudaStream_t s = output.GetComputeStream();
   ::TMVA::DNN::Cuda::MeanSquaredErrorGradients<<<gridDims, blockDims, 0, s>>>(
       dY.GetDataPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       weights.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   dY.SetComputeStream(s);
}

//____________________________________________________________________________
template<typename AFloat>
AFloat TCuda<AFloat>::CrossEntropy(const TCudaMatrix<AFloat> & Y,
                                   const TCudaMatrix<AFloat> & output,
                                   const TCudaMatrix<AFloat> &weights)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(Y);
   TCudaMatrix<AFloat>::ResetDeviceReturn();
   cudaStream_t s = Y.GetComputeStream();
   ::TMVA::DNN::Cuda::CrossEntropy<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix<AFloat>::GetDeviceReturnPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       weights.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   return TCudaMatrix<AFloat>::GetDeviceReturn();
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CrossEntropyGradients(TCudaMatrix<AFloat> & dY,
                                          const TCudaMatrix<AFloat> & Y,
                                          const TCudaMatrix<AFloat> & output,
                                          const TCudaMatrix<AFloat> &weights)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(Y);
   cudaStream_t s = output.GetComputeStream();
   ::TMVA::DNN::Cuda::CrossEntropyGradients<<<gridDims, blockDims, 0, s>>>(
       dY.GetDataPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       weights.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   dY.SetComputeStream(s);
}

//____________________________________________________________________________
template<typename AFloat>
AFloat TCuda<AFloat>::SoftmaxCrossEntropy(const TCudaMatrix<AFloat> & Y,
                                          const TCudaMatrix<AFloat> & output,
                                          const TCudaMatrix<AFloat> &weights)
{
   dim3 blockDims = TDevice::BlockDims1D();
   dim3 gridDims  = TDevice::GridDims1D(Y);
   TCudaMatrix<AFloat>::ResetDeviceReturn();
   cudaStream_t s = Y.GetComputeStream();
   ::TMVA::DNN::Cuda::SoftmaxCrossEntropy<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix<AFloat>::GetDeviceReturnPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       weights.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   return TCudaMatrix<AFloat>::GetDeviceReturn();
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SoftmaxCrossEntropyGradients(TCudaMatrix<AFloat> & dY,
                                                 const TCudaMatrix<AFloat> & Y,
                                                 const TCudaMatrix<AFloat> & output,
                                                 const TCudaMatrix<AFloat> &weights)
{
   dim3 blockDims = TDevice::BlockDims1D();
   dim3 gridDims  = TDevice::GridDims1D(Y);
   cudaStream_t s = output.GetComputeStream();
   ::TMVA::DNN::Cuda::SoftmaxCrossEntropyGradients<<<gridDims, blockDims, 0, s>>>(
       dY.GetDataPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       weights.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   dY.SetComputeStream(s);
}

} // namespace DNN
} // namespace TMVA
