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
void TCudnn<AFloat>::ActivationFunctionForward(Tensor_t & X, EActivationFunction activFunct, ActivationDescriptor_t activationDescr,  const double coef, const AFloat alpha, const AFloat beta)
{
   cudnnActivationMode_t activationMode;
   //std::cout << activationDescr << std::endl;
   //activationDescr = (ActivationDescriptor_t) nullptr;
   switch(activFunct) {
      case EActivationFunction::kIdentity: return; // Identity activation only works for cudnnConvolutionBiasActivationForward()
      case EActivationFunction::kRelu:     activationMode = CUDNN_ACTIVATION_RELU;    break;
      case EActivationFunction::kSigmoid:  activationMode = CUDNN_ACTIVATION_SIGMOID; break;
      case EActivationFunction::kTanh:     activationMode = CUDNN_ACTIVATION_TANH;    break;
      // The activations otherwise used are not supported by cuDNN
      default:    return;    
   };
   
   CUDNNCHECK(cudnnSetActivationDescriptor(activationDescr,
                                           activationMode,
                                           CUDNN_PROPAGATE_NAN,
                                           coef));
                                           
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
template<typename AFloat>
void TCudnn<AFloat>::ActivationFunctionBackward(Tensor_t & dX, const Tensor_t & Y,  
                                                const Tensor_t & dY, const Tensor_t & X, 
                                                EActivationFunction /* activFunct */,
                                                const ActivationDescriptor_t activationDescr, 
                                                const AFloat alpha, const AFloat beta)
{
   //if (!activationDescr) return;
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
template<typename AFloat>
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
}

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
