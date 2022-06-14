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
#include "TMVA/DNN/Functions.h"
#include "Kernels.cuh"

namespace TMVA
{
namespace DNN
{
//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ActivationFunctionForward(Tensor_t & X, EActivationFunction activFunct, 
                                              const ActivationDescriptor_t /* activationDescr */,  
                                             const double /* coef */, const AFloat /*alpha */, const AFloat /*beta*/)
{
   // scaling and translation is not yet implemented
   TMVA::DNN::evaluate<TCuda<AFloat>>( X, activFunct);
}
//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ActivationFunctionBackward(Tensor_t & dX, const Tensor_t & /* Y */,  
                                                const Tensor_t & dY, const Tensor_t & X, 
                                                EActivationFunction activFunct,
                                                const ActivationDescriptor_t /* activationDescr */, 
                                                const AFloat /* alpha */, const AFloat /* beta */)
{
   // scaling and translation not yet implemented
   // output tensor (Y) could also be used to speed up derivative calculation
   // compute dx = f'(x)
   TMVA::DNN::evaluateDerivative<TCuda<AFloat>>(dX, activFunct, X); 
    // Compute element-wise product.  dx = f'(x) * dY
   Hadamard(dX, dY);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::IdentityDerivative(TCudaTensor<AFloat> & B,
                                           const TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::IdentityDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       (int) B.GetNrows(),
       (int) B.GetNcols());
   B.SetComputeStream(s);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Relu(TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Relu<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ReluDerivative(TCudaTensor<AFloat> & B,
                                       const TCudaTensor<AFloat> & A)
{
    assert(B.GetNrows() == A.GetNrows() && B.GetNcols() == A.GetNcols());
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::ReluDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Sigmoid(TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Sigmoid<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SigmoidDerivative(TCudaTensor<AFloat> & B,
                                          const TCudaTensor<AFloat> & A)
{
    assert(B.GetNrows() == A.GetNrows() && B.GetNcols() == A.GetNcols());
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SigmoidDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Tanh(TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Tanh<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::TanhDerivative(TCudaTensor<AFloat> & B,
                                       const TCudaTensor<AFloat> & A)
{
    assert(B.GetNrows() == A.GetNrows() && B.GetNcols() == A.GetNcols());
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::TanhDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SymmetricRelu(TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SymmetricRelu<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SymmetricReluDerivative(TCudaTensor<AFloat> & B,
                                                const TCudaTensor<AFloat> & A)
{
    assert(B.GetNrows() == A.GetNrows() && B.GetNcols() == A.GetNcols());
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SymmetricReluDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SoftSign(TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SoftSign<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SoftSignDerivative(TCudaTensor<AFloat> & B,
                                           const TCudaTensor<AFloat> & A)
{
    assert(B.GetNrows() == A.GetNrows() && B.GetNcols() == A.GetNcols());
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SoftSignDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Gauss(TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Gauss<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//______________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::GaussDerivative(TCudaTensor<AFloat> & B,
                                    const TCudaTensor<AFloat> & A)
{
    assert(B.GetNrows() == A.GetNrows() && B.GetNcols() == A.GetNcols());
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::GaussDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}

} // namespace DNN
} // namespace TMVA
