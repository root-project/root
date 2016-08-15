// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Implementation of the loss functions for the TCuda implementation //
// of the low-level interface.                                      //
//////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"

namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
CudaDouble_t TCuda::MeanSquaredError(const TCudaMatrix & Y,
                                    const TCudaMatrix & output)
{
    dim3 blockDims = TDevice::BlockDims();
    dim3 gridDims  = TDevice::GridDims(Y);
    cudaStream_t s = Y.GetComputeStream();
    TCudaMatrix::ResetDeviceReturn();
    ::TMVA::DNN::Cuda::MeanSquaredError<<<gridDims, blockDims, 0, s>>>(
        TCudaMatrix::GetDeviceReturnPointer(),
        Y.GetDataPointer(),
        output.GetDataPointer(),
        (int) Y.GetNrows(),
        (int) Y.GetNcols());
    return TCudaMatrix::GetDeviceReturn();
}

//____________________________________________________________________________
void TCuda::MeanSquaredErrorGradients(TCudaMatrix & dY,
                                    const TCudaMatrix & Y,
                                    const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   cudaStream_t s = output.GetComputeStream();
   ::TMVA::DNN::Cuda::MeanSquaredErrorGradients<<<gridDims, blockDims, 0, s>>>(
       dY.GetDataPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   dY.SetComputeStream(s);
}

//____________________________________________________________________________
CudaDouble_t TCuda::CrossEntropy(const TCudaMatrix & Y,
                                const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   TCudaMatrix::ResetDeviceReturn();
   cudaStream_t s = Y.GetComputeStream();
   ::TMVA::DNN::Cuda::CrossEntropy<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix::GetDeviceReturnPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   return TCudaMatrix::GetDeviceReturn();
}

//____________________________________________________________________________
void TCuda::CrossEntropyGradients(TCudaMatrix & dY,
                                 const TCudaMatrix & Y,
                                 const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   cudaStream_t s = output.GetComputeStream();
   ::TMVA::DNN::Cuda::CrossEntropyGradients<<<gridDims, blockDims, 0, s>>>(
       dY.GetDataPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   dY.SetComputeStream(s);
}

} // namespace DNN
} // namespace TMVA
