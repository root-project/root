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
 // Implementation of the functions required for the forward and //
 // backward propagation of activations through a neural network //
 // for CUDA architectures.                                      //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<>
void TCuda<float>::MultiplyTranspose(TCudaMatrix<float> &output,
                                     const TCudaMatrix<float> &input,
                                     const TCudaMatrix<float> &Weights)
{
   int m, n, k;
   k = input.GetNcols();
   m = input.GetNrows();
   n = Weights.GetNrows();
   float alpha = 1.0, beta = 0.0;

   // Compute C = beta * C + alpha * (A * B^T)
   cudaStream_t s = input.GetComputeStream();
   cublasSetStream(input.GetCublasHandle(), s);
   cublasSgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                        // beta
               output.GetDataPointer(), m);   // *C, ldc
   output.SetComputeStream(s);
}

//____________________________________________________________________________
template<>
void TCuda<double>::MultiplyTranspose(TCudaMatrix<double> &output,
                                      const TCudaMatrix<double> &input,
                                      const TCudaMatrix<double> &Weights)
{
   int m, n, k;
   k = input.GetNcols();
   m = input.GetNrows();
   n = Weights.GetNrows();
   double alpha = 1.0, beta = 0.0;

   // Compute C = beta * C + alpha * (A * B^T)
   cudaStream_t s = input.GetComputeStream();
   cublasSetStream(input.GetCublasHandle(), s);
   cublasDgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                        // beta
               output.GetDataPointer(), m);   // *C, ldc
   output.SetComputeStream(s);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AddRowWise(TCudaMatrix<AFloat> &Weights,
                               const TCudaMatrix<AFloat> &theta)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(Weights);
   cudaStream_t s = Weights.GetComputeStream();
   ::TMVA::DNN::Cuda::AddRowWise<<<gridDims, blockDims, 0, s>>>(
       Weights.GetDataPointer(),
       theta.GetDataPointer(),
       Weights.GetNrows(),
       Weights.GetNcols());
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Backward(TCudaMatrix<AFloat> & activation_gradients_backward,
                             TCudaMatrix<AFloat> & weight_gradients,
                             TCudaMatrix<AFloat> & bias_gradients,
                             TCudaMatrix<AFloat> & df,
                             const TCudaMatrix<AFloat> & activation_gradients,
                             const TCudaMatrix<AFloat> & weights,
                             const TCudaMatrix<AFloat> & activation_backward)
{
   // Compute element-wise product.
   TCuda<AFloat>::Hadamard(df, activation_gradients);

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0) {
      TCuda<AFloat>::Multiply(activation_gradients_backward, df, weights);
   }

   // Weight gradients.
   if (weight_gradients.GetNoElements() > 0) {
      TCuda<AFloat>::TransposeMultiply(weight_gradients, df, activation_backward);
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      TCuda<AFloat>::SumColumns(bias_gradients, df);
   }

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Copy(TCudaMatrix<AFloat> & B,
                             const TCudaMatrix<AFloat> & A)
{
   size_t m = B.GetNrows();
   size_t n = B.GetNcols();
   cudaMemcpyAsync(B.GetDataPointer(), A.GetDataPointer(),
                   m * n * sizeof(AFloat), cudaMemcpyDeviceToDevice, 0);
}


//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Im2col(TCudaMatrix<AFloat> &A,
                           TCudaMatrix<AFloat> &B,
                           size_t imgHeight,
                           size_t imgWidth,
                           size_t fltHeight,
                           size_t fltWidth,
                           size_t strideRows,
                           size_t strideCols,
                           size_t zeroPaddingHeight,
                           size_t zeroPaddingWidth)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::RotateWeights(TCudaMatrix<AFloat> &A,
                                  const TCudaMatrix<AFloat> &B,
                                  size_t filterDepth,
                                  size_t filterHeight,
                                  size_t filterWidth,
                                  size_t numFilters)
{

}


//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ConvLayerBackward(std::vector<TCudaMatrix<AFloat>> & activation_gradients_backward,
                                      TCudaMatrix<AFloat> & weight_gradients,
                                      TCudaMatrix<AFloat> & bias_gradients,
                                      std::vector<TCudaMatrix<AFloat>> & df,
                                      const std::vector<TCudaMatrix<AFloat>> & activation_gradients,
                                      const TCudaMatrix<AFloat> & weights,
                                      const std::vector<TCudaMatrix<AFloat>> & activation_backward,
                                      size_t batchSize,
                                      size_t inputHeight,
                                      size_t inputWidth,
                                      size_t depth,
                                      size_t height,
                                      size_t width,
                                      size_t filterDepth,
                                      size_t filterHeight,
                                      size_t filterWidth,
                                      size_t nLocalViews)
{


}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvActivationGradients(
                                    std::vector<TCudaMatrix<AFloat>> & activation_gradients_backward,
                                    std::vector<TCudaMatrix<AFloat>> & df,
                                    const TCudaMatrix<AFloat> & weights,
                                    size_t batchSize,
                                    size_t inputHeight,
                                    size_t inputWidth,
                                    size_t depth,
                                    size_t height,
                                    size_t width,
                                    size_t filterDepth,
                                    size_t filterHeight,
                                    size_t filterWidth)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvWeightGradients(TCudaMatrix<AFloat> & weight_gradients,
                                                 std::vector<TCudaMatrix<AFloat>> & df,
                                                 const std::vector<TCudaMatrix<AFloat>> & activations_backward,
                                                 size_t batchSize,
                                                 size_t inputHeight,
                                                 size_t inputWidth,
                                                 size_t depth,
                                                 size_t height,
                                                 size_t width,
                                                 size_t filterDepth,
                                                 size_t filterHeight,
                                                 size_t filterWidth,
                                                 size_t nLocalViews)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvBiasGradients(TCudaMatrix<AFloat> & bias_gradients,
                                               std::vector<TCudaMatrix<AFloat>> & df,
                                               size_t batchSize,
                                               size_t depth,
                                               size_t nLocalViews)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AddConvBiases(TCudaMatrix<AFloat> &output,
                                  const TCudaMatrix<AFloat> &biases)
{

}


//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Downsample(TCudaMatrix<AFloat> &A,
                               TCudaMatrix<AFloat> &B,
                               const TCudaMatrix<AFloat> &C,
                               size_t imgHeight,
                               size_t imgWidth,
                               size_t fltHeight,
                               size_t fltWidth,
                               size_t strideRows,
                               size_t strideCols)
{

}

/____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::MaxPoolLayerBackward(std::vector<TCudaMatrix<AFloat>> & activationGradientsBackward,
                                         const std::vector<TCudaMatrix<AFloat>> & activationGradients,
                                         const std::vector<TCudaMatrix<AFloat>> & indexMatrix,
                                         size_t batchSize,
                                         size_t depth,
                                         size_t nLocalViews)
{

}

} // namespace DNN
} // namespace TMVA
