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
void TCudnn<AFloat>::Activation(TCudaTensor<AFloat> & A, EActivationFunction activFunct, const double coef, const AFloat alpha, const AFloat beta)
{
   cudnnActivationDescriptor_t activationDescriptor;
   
   CUDNNCHECK(cudnnCreateActivationDescriptor(&activationDescriptor));
   
   cudnnActivationMode_t activationMode;
   switch(activFunct) {
      case EActivationFunction::kIdentity: activationMode = CUDNN_ACTIVATION_IDENTITY;break;
      case EActivationFunction::kRelu:     activationMode = CUDNN_ACTIVATION_RELU;    break;
      case EActivationFunction::kSigmoid:  activationMode = CUDNN_ACTIVATION_SIGMOID; break;
      case EActivationFunction::kTanh:     activationMode = CUDNN_ACTIVATION_TANH;    break;
      // The activations otherwise used are not supported by cuDNN
      default:    activationMode = CUDNN_ACTIVATION_IDENTITY;     
   };
   CUDNNCHECK(cudnnSetActivationDescriptor(activationDescriptor,
                                           activationMode,
                                           CUDNN_PROPAGATE_NAN,
                                           coef));
                                           
   CUDNNCHECK(cudnnActivationForward(A.GetCudnnHandle(),
                                     activationDescriptor,
                                     &alpha,
                                     A.GetTensorDescriptor(),
                                     A.GetDataPointer(),
                                     &beta,
                                     A.GetTensorDescriptor(),     // Can be computed in place
                                     A.GetDataPointer()));

   CUDNNCHECK(cudnnDestroyActivationDescriptor(activationDescriptor));
}

//______________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::IdentityDerivative(TCudaTensor<AFloat> & B,
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
}*/

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Relu(TCudaTensor<AFloat> & A, const double coef, const AFloat alpha, const AFloat beta)
{
   Activation(A, EActivationFunction::kRelu, coef, alpha, beta);
}

//______________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::ReluDerivative(TCudaTensor<AFloat> & B,
                                       const TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::ReluDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}*/

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Sigmoid(TCudaTensor<AFloat> & A, const double coef, const AFloat alpha, const AFloat beta)
{
   Activation(A, EActivationFunction::kSigmoid, coef, alpha, beta);
}

//______________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::SigmoidDerivative(TCudaTensor<AFloat> & B,
                                          const TCudaTensor<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SigmoidDerivative<<<gridDims, blockDims, 0, s>>>(
       B.GetDataPointer(),
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
   B.SetComputeStream(s);
}*/

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Tanh(TCudaTensor<AFloat> & A, const double coef, const AFloat alpha, const AFloat beta)
{
   Activation(A, EActivationFunction::kTanh, coef, alpha, beta);
}

//______________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::TanhDerivative(TCudaTensor<AFloat> & B,
                                       const TCudaTensor<AFloat> & A)
{
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
void TCudnn<AFloat>::SymmetricRelu(TCudaTensor<AFloat> & A)
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
void TCudnn<AFloat>::SymmetricReluDerivative(TCudaTensor<AFloat> & B,
                                                const TCudaTensor<AFloat> & A)
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
void TCudnn<AFloat>::SoftSign(TCudaTensor<AFloat> & A)
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
void TCudnn<AFloat>::SoftSignDerivative(TCudaTensor<AFloat> & B,
                                           const TCudaTensor<AFloat> & A)
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
void TCudnn<AFloat>::Gauss(TCudaTensor<AFloat> & A)
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
void TCudnn<AFloat>::GaussDerivative(TCudaTensor<AFloat> & B,
                                    const TCudaTensor<AFloat> & A)
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

} // namespace DNN
} // namespace TMVA
