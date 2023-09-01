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
#include "Kernels.cuh"

namespace TMVA {
namespace DNN  {

//______________________________________________________________________________
template<typename AFloat>
AFloat TCuda<AFloat>::L1Regularization(const TCudaMatrix<AFloat> & A)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(A);
    cudaStream_t s = A.GetComputeStream();
    TCudaMatrix<AFloat>::ResetDeviceReturn();
    ::TMVA::DNN::Cuda::AbsoluteSum<<<gridDims, blockDims, 0, s>>>(
        TCudaMatrix<AFloat>::GetDeviceReturnPointer(),
        A.GetDataPointer(),
        (int) A.GetNrows(),
        (int) A.GetNcols());
    return TCudaMatrix<AFloat>::GetDeviceReturn();
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AddL1RegularizationGradients(TCudaMatrix<AFloat> & B,
                                                 const TCudaMatrix<AFloat> & A,
                                                 AFloat weightDecay)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::AddL1RegularizationGradients<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       weightDecay,
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
AFloat TCuda<AFloat>::L2Regularization(const TCudaMatrix<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   TCudaMatrix<AFloat>::ResetDeviceReturn();
   ::TMVA::DNN::Cuda::SquaredSum<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix<AFloat>::GetDeviceReturnPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   return TCudaMatrix<AFloat>::GetDeviceReturn();
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AddL2RegularizationGradients(TCudaMatrix<AFloat> & B,
                                                 const TCudaMatrix<AFloat> & A,
                                                 AFloat weightDecay)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
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
