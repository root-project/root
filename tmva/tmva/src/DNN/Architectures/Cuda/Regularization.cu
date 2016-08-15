// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Contains the definitions of the kernel calling functions for //
// computation of regularization functionals and gradients      //
// functions for CUDA architectures.                            //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"

namespace TMVA {
namespace DNN  {

//______________________________________________________________________________
CudaDouble_t TCuda::L1Regularization(const TCudaMatrix & A)
{
    dim3 blockDims = TDevice::BlockDims();
    dim3 gridDims  = TDevice::GridDims(A);
    cudaStream_t s = A.GetComputeStream();
    TCudaMatrix::ResetDeviceReturn();
    ::TMVA::DNN::Cuda::AbsoluteSum<<<gridDims, blockDims, 0, s>>>(
        TCudaMatrix::GetDeviceReturnPointer(),
        A.GetDataPointer(),
        (int) A.GetNrows(),
        (int) A.GetNcols());
    return TCudaMatrix::GetDeviceReturn();
}

//______________________________________________________________________________
void TCuda::AddL1RegularizationGradients(TCudaMatrix & B,
                                        const TCudaMatrix & A,
                                        CudaDouble_t weightDecay)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::AddL1RegularizationGradients<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       weightDecay,
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
CudaDouble_t TCuda::L2Regularization(const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   TCudaMatrix::ResetDeviceReturn();
   ::TMVA::DNN::Cuda::SquaredSum<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix::GetDeviceReturnPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   return TCudaMatrix::GetDeviceReturn();
}

//______________________________________________________________________________
void TCuda::AddL2RegularizationGradients(TCudaMatrix & B,
                                        const TCudaMatrix & A,
                                        CudaDouble_t weightDecay)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::AddL2RegularizationGradients<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       weightDecay,
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

} // namspace DNN
} // namspace TMVA
