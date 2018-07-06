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
#include <math.h>

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

inline bool isInteger(double x)
{
   return x == floor(x);
}

int calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   double dimension = ((imgDim - fltDim + 2 * padding) / stride) + 1;
   if (!isInteger(dimension)) {
      std::cout << "Not compatible hyper parameters" << std::endl;
      std::exit(EXIT_FAILURE);
   }

   return (size_t)dimension;
}

///////////////////////////////////////////////////////////////////////////////////
/// \brief A helper for image operations that rearranges image regions into
///        column vectors.
///
/// \param[out] A The output matrix. Each row corresponds to a receptive field.
/// \param[in] B The input matrix. Each row corresponds to a row in the image view.
/// \param[in] imgHeight The heigh of the input.
/// \param[in] imgWidth The output of the input.
/// \param[in] fltHeight Height of the kernel.
/// \param[in] fltWidth Width of the kernel.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
/// \param[in] zeroPaddingHeight The padding in the horizontal dimension.
/// \param[in] zeroPaddingWidth The padding in the vertical dimension.
///
/// This transformation allows us to express a 2D convolution as a matrix
/// multiplication. We can therefore harness the finely tuned GEMM
/// implementation of cuBLAS to achieve maximum performance. This function
/// can greatly speed-up propagation in TConvLayer.
///////////////////////////////////////////////////////////////////////////////////
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
//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Downsampling function used as the forward propagation step of a
///        Max-Pooling layer.
///
/// \param[out] A The output matrix. Each row corresponds to a slice and each element
///             is the max within a receptive field.
/// \param[out] B The winning indices matrix. Each element is the index of the max element.
/// \param[in] C The input matrix. Each row is a slice.
/// \param[in] imgHeight The heigh of the input.
/// \param[in] imgWidth The output of the input.
/// \param[in] fltHeight Height of the kernel.
/// \param[in] fltWidth Width of the kernel.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
///
/// Each output element is the maximum of the receptive field. We also save the winning
/// indices to facilitate back-propagation - we need to know which input element influenced
/// the output and only apply the derivative correction to this particular element.
/// The slicing process is the same as in a convolutional layer, however padding is set to 0.
///////////////////////////////////////////////////////////////////////////////////////////////
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
   size_t depth = C.GetNrows();

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Downsample<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(),
                                                                C.GetDataPointer(), depth, imgHeight, imgWidth,
                                                                fltHeight, fltWidth, strideRows, strideCols);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::MaxPoolLayerBackward(TCudaMatrix<AFloat> & activationGradientsBackward,
                                         const TCudaMatrix<AFloat> & activationGradients,
                                         const TCudaMatrix<AFloat> & indexMatrix,
                                         size_t imgHeight,
                                         size_t imgWidth,
                                         size_t fltHeight,
                                         size_t fltWidth,
                                         size_t strideRows,
                                         size_t strideCols,
                                         size_t /* nLocalViews */)
{
   size_t depth = activationGradientsBackward.GetNrows();

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(activationGradientsBackward);
   cudaStream_t s = activationGradientsBackward.GetComputeStream();

   ::TMVA::DNN::Cuda::MaxPoolBackward<<<gridDims, blockDims, 0, s>>>(activationGradientsBackward.GetDataPointer(),
                                                                     activationGradients.GetDataPointer(),
                                                                     indexMatrix.GetDataPointer(),
                                                                     depth, imgHeight, imgWidth, fltHeight, fltWidth,
                                                                     strideRows, strideCols);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Reshape(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(A);
    cudaStream_t s = A.GetComputeStream();

    ::TMVA::DNN::Cuda::Reshape<<<gridDims, blockDims>>>(A.GetDataPointer(), B.GetDataPointer(),
                                                        A.GetNrows(), A.GetNcols(), B.GetNrows(), B.GetNcols());
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
////////////////////////////////////////////////////////////////////////////////
/// \brief Flatten a vector of matrices into a single matrix.
///
/// \param[out] A Output matrix.
/// \param[in] B Input vector. Each element is a matrix to be concatenated.
/// \param[in] size Number of matrices in the input vector.
/// \param[in] nRows Number of rows in each matrix of the input vector.
/// \param[in] nCols Number of columns on each matrix of the input vector.
///
/// Each row in the output matrix is the concatenation of the same row in
/// each of the input matrices. Passing an std::vector to a CUDA kernel is
/// a non trivial task that requires manually allocating and copying to device
/// memory - details in comments within the function's body. Launching one
/// thread per output element.
//////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
void TCuda<AFloat>::Flatten(TCudaMatrix<AFloat> &A,
                            const std::vector<TCudaMatrix<AFloat>> &B,
                            size_t size,
                            size_t nRows,
                            size_t nCols)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();

   // Get raw pointers from a vector of matrices - this is more challenging than it sounds.
   //
   // Attention: While `TCudaMatrix.GetDataPointer() returns a pointer to device memory,
   //            std::vector (and its .data() raw pointer) resides on host memory. Therefore
   //            we need to manually copy these pointers to the device prior to invoking the kernel.

   const AFloat ** dB; // device pointer to device pointers.S
   const AFloat ** hB = new const AFloat * [size]; // host pointer to device pointers.

   cudaMalloc(&dB, sizeof(AFloat *) * size);

   for(size_t i = 0; i < size; ++i) {
      hB[i] = B[i].GetDataPointer();
   }

   cudaMemcpy(dB, hB, sizeof(AFloat *) * size, cudaMemcpyHostToDevice);

   // Launch the kernel using our device pointers.
   ::TMVA::DNN::Cuda::Flatten<<<gridDims, blockDims>>>(A.GetDataPointer(), dB, size, nRows, nCols);
}

//____________________________________________________________________________
////////////////////////////////////////////////////////////////////////////////
/// \brief Deflatten a matrix into a vector of matrices.
///
/// \param[out] A Output matrices. Each element will be a part of the input.
/// \param[in] B Input flat matrix.
/// \param[in] size Number of matrices in the output vector.
/// \param[in] nRows Number of rows in each matrix of the output vector.
/// \param[in] nCols Number of columns on each matrix of the output vector.
///
/// Each row in the input matrix is the concatenation of the same row in
/// each of the output matrices. Passing an std::vector to a CUDA kernel is
/// a non trivial task that requires manually allocating and copying to device
/// memory - details in comments within the function's body. Launching one
/// thread per input element.
//////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
void TCuda<AFloat>::Deflatten(std::vector<TCudaMatrix<AFloat>> &A,
                              const TCudaMatrix<AFloat> &B,
                              size_t size,
                              size_t nRows,
                              size_t nCols)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(B);
    cudaStream_t s = B.GetComputeStream();

    // Get raw pointers from a vector of matrices - this is more challenging than it sounds.
    //
    // Attention: While `TCudaMatrix.GetDataPointer() returns a pointer to device memory,
    //            std::vector (and its .data() raw pointer) resides on host memory. Therefore
    //            we need to manually copy these pointers to the device prior to invoking the kernel.

    AFloat ** dA; // device pointer to device pointers.
    AFloat ** hA = new AFloat * [size]; // host pointer to device pointers.

    cudaMalloc(&dA, sizeof(AFloat *) * size);

    for(size_t i = 0; i < size; ++i) {
        hA[i] = A[i].GetDataPointer();
    }

    cudaMemcpy(dA, hA, sizeof(AFloat *) * size, cudaMemcpyHostToDevice);

    // Launch the kernel using our device pointers.
    ::TMVA::DNN::Cuda::Deflatten<<<gridDims, blockDims>>>(dA, B.GetDataPointer(), size, nRows, nCols);
}

} // namespace DNN
} // namespace TMVA
