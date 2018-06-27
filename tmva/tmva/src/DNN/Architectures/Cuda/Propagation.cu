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
void TCuda<AFloat>::Copy(std::vector<TCudaMatrix<AFloat>> & B,
                             const std::vector<TCudaMatrix<AFloat>> & A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      Copy(B[i], A[i]);
   }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Im2col(TCudaMatrix<AFloat> &A,
                           const TCudaMatrix<AFloat> &B,
                           size_t imgHeight,
                           size_t imgWidth,
                           size_t fltHeight,
                           size_t fltWidth,
                           size_t strideRows,
                           size_t strideCols,
                           size_t zeroPaddingHeight,
                           size_t zeroPaddingWidth)
{
    size_t depth = B.GetNrows();

    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(A);
    cudaStream_t s = A.GetComputeStream();

    ::TMVA::DNN::Cuda::Im2Col<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(), depth, imgHeight, imgWidth,
                                                             fltHeight, fltWidth, strideRows, strideCols,
                                                             zeroPaddingHeight, zeroPaddingWidth);
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
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = B.GetComputeStream();

   ::TMVA::DNN::Cuda::RotateWeights<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(), filterDepth,
                                                                   filterHeight, filterWidth, numFilters);

}

template <typename AFloat>
void TCuda<AFloat>::ConvLayerForward(std::vector<TCudaMatrix<AFloat>> & output,
                                     std::vector<TCudaMatrix<AFloat>> & derivatives,
                                     const std::vector<TCudaMatrix<AFloat>> &input,
                                     const TCudaMatrix<AFloat> &weights, const TCudaMatrix<AFloat> & biases,
                                     size_t inputHeight, size_t inputWidth, size_t inputDepth, size_t fltHeight,
                                     size_t fltWidth, size_t numberFilters, size_t strideRows, size_t strideCols,
                                     size_t zeroPaddingHeight, size_t zeroPaddingWidth, EActivationFunction activFunc)
{

   // Issue with re-definition of `calculateDimension`. I need to solve this...
   size_t height = ((inputHeight - fltHeight + 2 * zeroPaddingHeight) / strideRows) + 1;
   size_t width = ((inputWidth- fltWidth + 2 * zeroPaddingWidth) / strideCols) + 1;
   size_t nLocalViews = height * width;
   size_t nLocalViewPixels = inputDepth * fltHeight * fltWidth;

   TCudaMatrix<AFloat> inputPrime(nLocalViews, nLocalViewPixels);
   for(size_t event = 0; event < input.size(); event++) {
      Im2col(inputPrime, input[event], inputHeight, inputWidth, fltHeight, fltWidth, strideRows, strideCols,
             zeroPaddingHeight, zeroPaddingWidth);

      MultiplyTranspose(output[event], weights, inputPrime);
      AddConvBiases(output[event], biases);

      evaluateDerivative<TCuda<AFloat>>(derivatives[event], activFunc, output[event]);
      evaluate<TCuda<AFloat>>(output[event], activFunc);
  }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ConvLayerBackward(std::vector<TCudaMatrix<AFloat>> & activationGradientsBackward,
                                      TCudaMatrix<AFloat> & weightGradients,
                                      TCudaMatrix<AFloat> & biasGradients,
                                      std::vector<TCudaMatrix<AFloat>> & df,
                                      const std::vector<TCudaMatrix<AFloat>> & activationGradients,
                                      const TCudaMatrix<AFloat> & weights,
                                      const std::vector<TCudaMatrix<AFloat>> & activationBackward,
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
    for (size_t i = 0; i < batchSize; i++) {
        // Compute element-wise product.
        Hadamard(df[i], activationGradients[i]);
    }

    // Calculate the activation gradients of the previous layer
    CalculateConvActivationGradients(activationGradientsBackward, df, weights, batchSize, inputHeight, inputWidth, depth,
                                     height, width, filterDepth, filterHeight, filterWidth);


    // Calculate the weight gradients
    CalculateConvWeightGradients(weightGradients, df, activationBackward, batchSize, inputHeight, inputWidth, depth,
                                 height, width, filterDepth, filterHeight, filterWidth, nLocalViews);

    // Calculate the bias gradients
    CalculateConvBiasGradients(biasGradients, df, batchSize, depth, nLocalViews);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvActivationGradients(
                                    std::vector<TCudaMatrix<AFloat>> & activationGradientsBackward,
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
    if (activationGradientsBackward.size() == 0) return;

    TCudaMatrix<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
    RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());

    // Calculate the zero paddings.
    size_t tempZeroPaddingHeight = (size_t)(floor((inputHeight - height + filterHeight - 1) / 2));
    size_t tempZeroPaddingWidth = (size_t)(floor((inputWidth - width + filterWidth - 1) / 2));

    // Calculate the number of local views and the number of pixels in each view.
    size_t tempNLocalViews = inputHeight * inputWidth;
    size_t tempNLocalViewPixels = depth * filterHeight * filterWidth;

    // Problem here. We need to generalize!
    size_t tempStrideRows = 1;
    size_t tempStrideCols = 1;

    // Convolution.
    TCudaMatrix<AFloat> dfPrime(tempNLocalViews, tempNLocalViewPixels);
    for(size_t event = 0; event < df.size(); event++) {
        Im2col(dfPrime, df[event], height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
               tempZeroPaddingHeight, tempZeroPaddingWidth);

        MultiplyTranspose(activationGradientsBackward[event], rotWeights, dfPrime);
    }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvWeightGradients(TCudaMatrix<AFloat> & weightGradients,
                                                 std::vector<TCudaMatrix<AFloat>> & df,
                                                 const std::vector<TCudaMatrix<AFloat>> & activationsBackward,
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
    // reinitialize the weight gradients to 0
    weightGradients.Zero();

    const size_t filterSize = filterHeight * filterWidth;
    const size_t nLocalViewPixels = filterDepth * filterSize;
    R__ASSERT( weightGradients.GetNcols() == nLocalViewPixels);

    const size_t tempStrideRows = 1;
    const size_t tempStrideCols = 1;

    // Calculate the zero paddings from the input height and width (assume stride = 1)
    const size_t tempZeroPaddingHeight = (height - inputHeight + filterHeight - 1) / 2;
    const size_t tempZeroPaddingWidth = (width - inputWidth + filterWidth - 1) / 2;

    std::vector< TCudaMatrix<AFloat> > vres;
    for (size_t i = 0; i < batchSize; i++) {
        vres.emplace_back(depth, nLocalViewPixels);
    }

    // Convolution.
    TCudaMatrix<AFloat> activationsPrime(nLocalViews, nLocalViewPixels);
    for(size_t event = 0; event < df.size(); event++) {
        Im2col(activationsPrime, activationsBackward[event], inputHeight, inputWidth, filterHeight, filterWidth,
               tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);

        Multiply(vres[event], df[event], activationsPrime);
    }

    R__ASSERT(vres.size() == batchSize);
    for (size_t i = 0; i < batchSize; i++) {
        for (size_t j = 0; j < depth; j++) {
            for (size_t k = 0; k < filterDepth; k++) {
                size_t kOffset = k * filterSize;
                for (size_t l = 0; l < filterSize; l++) {
                    weightGradients(j, kOffset + l) += vres[i](j,  kOffset + l);
                }
            }
        }
    }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvBiasGradients(TCudaMatrix<AFloat> & biasGradients,
                                               std::vector<TCudaMatrix<AFloat>> & df,
                                               size_t batchSize,
                                               size_t depth,
                                               size_t nLocalViews)
{
    biasGradients.Zero();
    TCudaMatrix<AFloat> temp(biasGradients.GetNrows(), biasGradients.GetNcols());
    for (size_t event = 0; event < batchSize; event++) {
        TCuda<AFloat>::SumRows(temp, df[event]);
        TCuda<AFloat>::ScaleAdd(biasGradients, temp);
    }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AddConvBiases(TCudaMatrix<AFloat> &output,
                                  const TCudaMatrix<AFloat> &biases)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(output);
    cudaStream_t s = output.GetComputeStream();
    ::TMVA::DNN::Cuda::AddBiases<<<gridDims, blockDims, 0, s>>>(
            output.GetDataPointer(),
            biases.GetDataPointer(),
            output.GetNrows(),
            output.GetNcols());
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

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::MaxPoolLayerBackward(std::vector<TCudaMatrix<AFloat>> & activationGradientsBackward,
                                         const std::vector<TCudaMatrix<AFloat>> & activationGradients,
                                         const std::vector<TCudaMatrix<AFloat>> & indexMatrix,
                                         size_t batchSize,
                                         size_t depth,
                                         size_t nLocalViews)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Reshape(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B)
{
   //TODO    
}

//______________________________________________________________________________
template <typename AReal>
void TCuda<AReal>::Rearrange(std::vector<TCudaMatrix<AReal>> &out, const std::vector<TCudaMatrix<AReal>> &in)
{
   // B x T x D out --- T x B x D in*/
   size_t B = out.size();
   size_t T = out[0].GetNrows();
   size_t D = out[0].GetNcols();
   if ((T != in.size()) || (B != in[0].GetNrows()) 
       || (D != in[0].GetNcols())) {
      std::cout << "Incompatible Dimensions\n"
         << in.size() << "x" << in[0].GetNrows() << "x" << in[0].GetNcols() 
         << " --> " << B << "x" << T << "x" << D << "\n";
      return;
   }
   for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < T; ++j) {
         for (size_t k = 0; k < D; ++k) {
            out[i](j, k) = in[j](i, k);
         }
      }
   }
   return;
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Flatten(TCudaMatrix<AFloat> &A,
                            const std::vector<TCudaMatrix<AFloat>> &B,
                            size_t size,
                            size_t nRows,
                            size_t nCols)
{

}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Deflatten(std::vector<TCudaMatrix<AFloat>> &A,
                              const TCudaMatrix<AFloat> &B,
                              size_t index,
                              size_t nRows,
                              size_t nCols)
{

}

} // namespace DNN
} // namespace TMVA
