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
 // Implementation of the activation functions for the TCuda      //
 // implementation of the low-level interface.                   //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
void TCuda::IdentityDerivative(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   identity_derivative<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                                      (int) A.GetNrows(),
                                                      (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::Relu(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   relu<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                       (int) A.GetNrows(),
                                       (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::ReluDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   relu_derivative<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                  A.GetDataPointer(),
                                                  (int) A.GetNrows(),
                                                  (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::Sigmoid(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   sigmoid<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                          (int) A.GetNrows(),
                                          (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::SigmoidDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   sigmoid_derivative<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                     A.GetDataPointer(),
                                                     (int) A.GetNrows(),
                                                     (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::Tanh(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   tanh<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                       (int) A.GetNrows(),
                                       (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::TanhDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   tanh_derivative<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                  A.GetDataPointer(),
                                                  (int) A.GetNrows(),
                                                  (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::SymmetricRelu(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   symmetric_relu<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                                 (int) A.GetNrows(),
                                                 (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::SymmetricReluDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   symmetric_relu_derivative<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                            A.GetDataPointer(),
                                                            (int) A.GetNrows(),
                                                            (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::SoftSign(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   soft_sign<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                            (int) A.GetNrows(),
                                            (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::SoftSignDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   soft_sign_derivative<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                       A.GetDataPointer(),
                                                       (int) A.GetNrows(),
                                                       (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::Gauss(TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();
   gauss<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(),
                                        (int) A.GetNrows(),
                                        (int) A.GetNcols());
}

//______________________________________________________________________________
void TCuda::GaussDerivative(TCudaMatrix & B, const TCudaMatrix & A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   gauss_derivative<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                   A.GetDataPointer(),
                                                   (int) A.GetNrows(),
                                                   (int) A.GetNcols());
}

} // namespace DNN
} // namespace TMVA
