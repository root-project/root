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
template<bool doProfiling>
CudaDouble_t TCuda<doProfiling>::MeanSquaredError(const TCudaMatrix & Y,
                                                  const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   cudaStream_t s = Y.GetComputeStream();

   tick();

   TCudaMatrix::ResetDeviceReturn();
   mean_squared_error<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix::GetDeviceReturnPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   CudaDouble_t result = TCudaMatrix::GetDeviceReturn();
   tock(fTimings.TimeMeanSquaredError);

   return result;
}

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::MeanSquaredErrorGradients(TCudaMatrix & dY,
                                                   const TCudaMatrix & Y,
                                                   const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   cudaStream_t s = Y.GetComputeStream();

   tick();
   mean_squared_error_gradients<<<gridDims, blockDims, 0, s>>>(
       dY.GetDataPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   tock(fTimings.TimeMeanSquaredErrorGradients);
}

//____________________________________________________________________________
template<bool doProfiling>
CudaDouble_t TCuda<doProfiling>::CrossEntropy(const TCudaMatrix & Y,
                                              const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   TCudaMatrix::ResetDeviceReturn();
   cudaStream_t s = Y.GetComputeStream();

   tick();
   cross_entropy<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix::GetDeviceReturnPointer(),
       Y.GetDataPointer(),
       output.GetDataPointer(),
       (int) Y.GetNrows(),
       (int) Y.GetNcols());
   CudaDouble_t result = TCudaMatrix::GetDeviceReturn();
   tock(fTimings.TimeCrossEntropy);

   return result;
}

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::CrossEntropyGradients(TCudaMatrix & dY,
                                               const TCudaMatrix & Y,
                                               const TCudaMatrix & output)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Y);
   cudaStream_t s = Y.GetComputeStream();

   tick();
   cross_entropy_gradients<<<gridDims, blockDims, 0, s>>>(dY.GetDataPointer(),
                                                          Y.GetDataPointer(),
                                                          output.GetDataPointer(),
                                                          (int) Y.GetNrows(),
                                                          (int) Y.GetNcols());
   tock(fTimings.TimeCrossEntropyGradients);
}

} // namespace DNN
} // namespace TMVA
