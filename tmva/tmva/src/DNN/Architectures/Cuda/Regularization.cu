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
template<bool doProfiling>
CudaDouble_t TCuda<doProfiling>::L1Regularization(const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();

   tick();
   TCudaMatrix::ResetDeviceReturn();
   absolute_sum<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix::GetDeviceReturnPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   CudaDouble_t result = TCudaMatrix::GetDeviceReturn();
   tock(fTimings.TimeL1Regularization);
   return result;
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::AddL1RegularizationGradients(TCudaMatrix & B,
                                                      const TCudaMatrix & A,
                                                      CudaDouble_t weightDecay)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();

   tick();
   add_l1_regularization_gradients<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       weightDecay,
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeAddL1RegularizationGradients);
}

//______________________________________________________________________________
template<bool doProfiling>
CudaDouble_t TCuda<doProfiling>::L2Regularization(const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();

   tick();
   TCudaMatrix::ResetDeviceReturn();
   squared_sum<<<gridDims, blockDims, 0, s>>>(TCudaMatrix::GetDeviceReturnPointer(),
                                              A.GetDataPointer(),
                                              (int) A.GetNrows(),
                                              (int) A.GetNcols());
   CudaDouble_t result = TCudaMatrix::GetDeviceReturn();
   tock(fTimings.TimeL2Regularization);
   return result;
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::AddL2RegularizationGradients(TCudaMatrix & B,
                                                      const TCudaMatrix & A,
                                                      CudaDouble_t weightDecay)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();

   tick();
   add_l2_regularization_gradients<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       weightDecay,
       (int) A.GetNrows(),
       (int) A.GetNcols());
   tock(fTimings.TimeAddL2RegularizationGradients);
}

} // namspace DNN
} // namspace TMVA
