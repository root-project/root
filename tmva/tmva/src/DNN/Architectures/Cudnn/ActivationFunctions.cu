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

#include "TMVA/DNN/Architectures/TCudnn.h"
/*#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"*/

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::ActivationFunctionForward(Tensor_t & X, EActivationFunction activFunct, const ActivationDescriptor_t activationDescr,  const double coef, const AFloat alpha, const AFloat beta)
{
   // compute forward activation in place
   // Nothing to do for identity function
   if (activFunct == EActivationFunction::kIdentity) return;

   CUDNNCHECK(cudnnActivationForward(X.GetCudnnHandle(),
                                     activationDescr,
                                     &alpha,
                                     X.GetTensorDescriptor(),
                                     X.GetDataPointer(),
                                     &beta,
                                     X.GetTensorDescriptor(),     // Can be computed in place
                                     X.GetDataPointer()));
}
//______________________________________________________________________________
template <typename AFloat>
void TCudnn<AFloat>::ActivationFunctionForward(Tensor_t & Y, const Tensor_t &X, EActivationFunction activFunct,
                                               const ActivationDescriptor_t activationDescr, const double coef,
                                               const AFloat alpha, const AFloat beta)
{
   // compute forward activation  with different input/output tensor (needed in training)
   // Nothing to do for identity function
   if (activFunct == EActivationFunction::kIdentity) {
      TCudnn<AFloat>::Copy(Y, X);
      return;
   }

   CUDNNCHECK(cudnnActivationForward(X.GetCudnnHandle(), activationDescr, &alpha, X.GetTensorDescriptor(),
                                     X.GetDataPointer(), &beta,
                                     Y.GetTensorDescriptor(), // Can be computed in place
                                     Y.GetDataPointer()));
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::ActivationFunctionBackward(Tensor_t & dX, const Tensor_t & Y,
                                                const Tensor_t & dY, const Tensor_t & X,
                                                EActivationFunction activFunct ,
                                                const ActivationDescriptor_t activationDescr,
                                                const AFloat alpha, const AFloat beta)
{
   // For identity function output dX is = dY
   if (activFunct == EActivationFunction::kIdentity) {
      Copy(dX,dY);
      return;
   }
   //std::cout << "No identityy\n";
   //Y.Print();
   // The activation descriptor is set in the forward pass
   CUDNNCHECK(cudnnActivationBackward(X.GetCudnnHandle(),
                                      activationDescr,
                                      &alpha,
                                      Y.GetTensorDescriptor(),
                                      Y.GetDataPointer(),
                                      dY.GetTensorDescriptor(),
                                      dY.GetDataPointer(),
                                      X.GetTensorDescriptor(),
                                      X.GetDataPointer(),
                                      &beta,
                                      dX.GetTensorDescriptor(),
                                      dX.GetDataPointer()));
}

#if 0
//______________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::Relu(Tensor_t & X, ActivationDescriptor_t activationDescr, const double coef, const AFloat alpha, const AFloat beta)
{
   Activation(X, EActivationFunction::kRelu, activationDescr, coef, alpha, beta);
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::ReluDerivative(const Tensor_t & Y, const Tensor_t & dY,
                                    const Tensor_t & X, Tensor_t & dX,
                                    const ActivationDescriptor_t activationDescr,
                                    const AFloat alpha, const AFloat beta)
{
   ActivationFunctionBackward(Y, dY, X, dX, activationDescr, alpha, beta);
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Sigmoid(Tensor_t & X, ActivationDescriptor_t activationDescr, const double coef, const AFloat alpha, const AFloat beta)
{
   Activation(X, EActivationFunction::kSigmoid, activationDescr, coef, alpha, beta);
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::SigmoidDerivative(const Tensor_t & Y, const Tensor_t & dY,
                                       const Tensor_t & X, Tensor_t & dX,
                                       const ActivationDescriptor_t activationDescr,
                                       const AFloat alpha, const AFloat beta)
{
   ActivationFunctionBackward(Y, dY, X, dX, activationDescr, alpha, beta);
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Tanh(Tensor_t & X, ActivationDescriptor_t activationDescr, const double coef, const AFloat alpha, const AFloat beta)
{
   Activation(X, EActivationFunction::kTanh, activationDescr, alpha, beta);
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::TanhDerivative(const Tensor_t & Y, const Tensor_t & dY,
                                    const Tensor_t & X, Tensor_t & dX,
                                    const ActivationDescriptor_t activationDescr,
                                    const AFloat alpha, const AFloat beta)
{
   ActivationFunctionBackward(Y, dY, X, dX, activationDescr, alpha, beta);
}*/

//______________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::SymmetricRelu(Tensor_t & A)
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
void TCudnn<AFloat>::SymmetricReluDerivative(Tensor_t & B,
                                                const Tensor_t & A)
{
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
void TCudnn<AFloat>::SoftSign(Tensor_t & A)
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
void TCudnn<AFloat>::SoftSignDerivative(Tensor_t & B,
                                           const Tensor_t & A)
{
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
void TCudnn<AFloat>::Gauss(Tensor_t & A)
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
void TCudnn<AFloat>::GaussDerivative(Tensor_t & B,
                                    const Tensor_t & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::GaussDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}*/
#endif

} // namespace DNN
} // namespace TMVA
