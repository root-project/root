// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 11/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////
// Explicit instantiation of the Reference architecture class //
// template for Double_t scalar types.                        //
////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

namespace TMVA
{
namespace DNN
{

template<typename AFloat>
void TCuda<AFloat>::Sigmoid(TCudaMatrix<AFloat> & B,
                            const TCudaMatrix<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Sigmoid<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                             A.GetDataPointer(),
                                                             (int) A.GetNrows(),
                                                             (int) A.GetNcols());
   B.SetComputeStream(s);
}

} // namespace DNN
} // namespace TMVA
