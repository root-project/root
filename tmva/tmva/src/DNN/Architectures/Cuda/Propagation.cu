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
   cudaStream_t s = output.GetComputeStream();
   cublasSetStream(input.GetCublasHandle(), s);
   cublasSgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                        // beta
               output.GetDataPointer(), m);   // *C, ldc
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
   cudaStream_t s = output.GetComputeStream();
   cublasSetStream(input.GetCublasHandle(), s);
   cublasDgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                        // beta
               output.GetDataPointer(), m);   // *C, ldc
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
void TCuda<AFloat>::Backward(TCudaTensor<AFloat> & activation_gradients_backward,
                             TCudaMatrix<AFloat> & weight_gradients,
                             TCudaMatrix<AFloat> & bias_gradients,
                             const TCudaTensor<AFloat> & df,
                             const TCudaTensor<AFloat> & activation_gradients,
                             const TCudaMatrix<AFloat> & weights,
                             const TCudaTensor<AFloat> & activation_backward)
{
   // Compute element-wise product.
   //Matrix_t df_m = df.GetMatrix(); 

   // df  is the output of ActivationBackward
   //TCuda<AFloat>::Hadamard(df, activation_gradients);
   //TCuda<AFloat>::Hadamard(df_m, activation_gradients.GetMatrix());

   Matrix_t df_m = df.GetMatrix(); 

   // Activation gradients.
   if (activation_gradients_backward.GetSize() > 0) {

      Matrix_t  activation_gradients_backward_m = activation_gradients_backward.GetMatrix(); 
      TCuda<AFloat>::Multiply(activation_gradients_backward_m, df_m, weights);
   }

   // Weight gradients.
   if (weight_gradients.GetNoElements() > 0) {
      TCuda<AFloat>::TransposeMultiply(weight_gradients, df_m, activation_backward.GetMatrix());
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      TCuda<AFloat>::SumColumns(bias_gradients, df_m);
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
void TCuda<AFloat>::Copy(TCudaTensor<AFloat> & B,
                             const TCudaTensor<AFloat> & A)
{
   size_t n = B.GetSize(); 
   //asssert (A.GetSize() >= B.GetSize());
   cudaMemcpyAsync(B.GetDataPointer(), A.GetDataPointer(),
      n * sizeof(AFloat), cudaMemcpyDeviceToDevice, 0);
}

//____________________________________________________________________________
template<typename AFloat>
size_t TCuda<AFloat>::calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   size_t temp = imgDim - fltDim + 2 * padding;
   if (temp % stride || temp + stride <= 0) {
      Fatal("calculateDimension", "Not compatible hyper parameters for layer - (imageDim, filterDim, padding, stride)"
            " %zu , %zu , %zu , %zu", imgDim, fltDim, padding, stride);
   }
   return temp / stride + 1;
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
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = B.GetComputeStream();

   ::TMVA::DNN::Cuda::RotateWeights<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(), filterDepth,
                                                                   filterHeight, filterWidth, numFilters);
}

#if 0
template <typename AFloat>
void TCuda<AFloat>::PrepareInternals(TCudaTensor<AFloat> & /* inputPrime */ )
{
   // non think this is needed when using tensor
   // for (size_t event = 0; event < inputPrime.size(); event++) {
   //    cudaStream_t s;
   //    cudaStreamCreate(&s);
   //    inputPrime[event].SetComputeStream(s);
   // }
}
#endif

template <typename AFloat>
void TCuda<AFloat>::ConvLayerForward(TCudaTensor<AFloat> & output,
                                     TCudaTensor<AFloat> & inputActivationFunc,
                                     const TCudaTensor<AFloat> &input,
                                     const TCudaMatrix<AFloat> &weights, const TCudaMatrix<AFloat> & biases,
                                     const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                     TCudaTensor<AFloat> & inputPrime,
                                     const ConvDescriptors_t & /*descriptors*/,
                                     ConvWorkspace_t & /*workspace*/)
{
   size_t height = calculateDimension(params.inputHeight, params.filterHeight, params.paddingHeight, params.strideRows);
   size_t width = calculateDimension(params.inputWidth, params.filterWidth, params.paddingWidth, params.strideCols);

   // for(size_t event = 0; event < input.size(); event++) {
   //    cudaStream_t s = inputPrime[event].GetComputeStream();
   //    output[event].SetComputeStream(s);
   //    derivatives[event].SetComputeStream(s);
   // }

   for(size_t event = 0; event < input.GetFirstSize(); event++) {
      Matrix_t inputPrime_m = inputPrime.At(event).GetMatrix();
      Matrix_t output_m = output.At(event).GetMatrix();

      Im2col(inputPrime_m, input.At(event).GetMatrix(), params.inputHeight, params.inputWidth, params.filterHeight, params.filterWidth,
             params.strideRows, params.strideCols, params.paddingHeight, params.paddingWidth);

      MultiplyTranspose(output_m, weights, inputPrime_m);
      AddConvBiases(output_m, biases);
   }

   //evaluateDerivative<TCuda<AFloat>>(derivatives, activFunc, output);
   //evaluate<TCuda<AFloat>>(output, activFunc);
   
   // save output of convolution before activation function evaluation
   Copy(inputActivationFunc, output);
   ActivationFunctionForward(output, activFunc, ActivationDescriptor_t() ); 
  
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ConvLayerBackward(TCudaTensor<AFloat> & activationGradientsBackward,
                                      TCudaMatrix<AFloat> & weightGradients,
                                      TCudaMatrix<AFloat> & biasGradients,
                                      TCudaTensor<AFloat> & inputActivationFunc,
                                      TCudaTensor<AFloat> & activationGradients,
                                      const TCudaMatrix<AFloat> & weights,
                                      const TCudaTensor<AFloat> & activationBackward,
                                      const Tensor_t & outputTensor,
                                      EActivationFunction activFunc,
                                      const ConvDescriptors_t & /*descriptors*/,
                                      ConvWorkspace_t & /*workspace*/,
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
    
   // Compute activation backward
  //Tensor_t df = activationGradients;   // this is a shallow copy
   Tensor_t df(activationGradients.GetShape() );  
   ActivationFunctionBackward(df, outputTensor, activationGradients, inputActivationFunc, 
                              activFunc, ActivationDescriptor_t() );


   //Hadamard(df, activationGradients);
   

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
                                    TCudaTensor<AFloat> & activationGradientsBackward,
                                    const TCudaTensor<AFloat> & df,
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
   if (activationGradientsBackward.GetSize() == 0) return;

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

   R__ASSERT( df.GetFirstSize() ==  batchSize);
   // Convolution.
   TCudaMatrix<AFloat> dfPrime(tempNLocalViews, tempNLocalViewPixels);
   for(size_t event = 0; event < batchSize; event++) {
      Im2col(dfPrime, df.At(event).GetMatrix(), height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);

      TCudaMatrix<AFloat> agb_m = activationGradientsBackward.At(event).GetMatrix();
      MultiplyTranspose(agb_m, rotWeights, dfPrime);
   }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvWeightGradients(TCudaMatrix<AFloat> & weightGradients,
                                                 const TCudaTensor<AFloat> & df,
                                                 const TCudaTensor<AFloat> & activationsBackward,
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
    R__ASSERT( weightGradients.GetNrows() == depth);
    R__ASSERT( df.GetFirstSize() ==  batchSize);



    const size_t tempStrideRows = 1;
    const size_t tempStrideCols = 1;

    // Calculate the zero paddings from the input height and width (assume stride = 1)
    const size_t tempZeroPaddingHeight = (height - inputHeight + filterHeight - 1) / 2;
    const size_t tempZeroPaddingWidth = (width - inputWidth + filterWidth - 1) / 2;

    // Convolution.
    TCudaMatrix<AFloat> activationsPrime(nLocalViews, nLocalViewPixels);
    TCudaMatrix<AFloat> resPrime(depth, nLocalViewPixels);
    for(size_t event = 0; event < batchSize; event++) {
        Im2col(activationsPrime, activationsBackward.At(event).GetMatrix(), inputHeight, inputWidth, filterHeight, filterWidth,
               tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);

        Multiply(resPrime, df.At(event).GetMatrix(), activationsPrime);

        TCuda<AFloat>::ScaleAdd(weightGradients, resPrime, 1.0); 
    }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::CalculateConvBiasGradients(TCudaMatrix<AFloat> & biasGradients,
                                               const TCudaTensor<AFloat> & df,
                                               size_t batchSize,
                                               size_t /* depth */,
                                               size_t /* nLocalViews */)
{
    biasGradients.Zero();
    TCudaMatrix<AFloat> temp(biasGradients.GetNrows(), biasGradients.GetNcols());
    for (size_t event = 0; event < batchSize; event++) {
        TCuda<AFloat>::SumRows(temp, df.At(event).GetMatrix());
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
void TCuda<AFloat>::Downsample(TCudaTensor<AFloat> &A,
                               TCudaTensor<AFloat> &B,
                               const TCudaTensor<AFloat> &C,
                               size_t imgHeight,
                               size_t imgWidth,
                               size_t fltHeight,
                               size_t fltWidth,
                               size_t strideRows,
                               size_t strideCols)
{
   size_t depth = C.GetCSize(); 
   size_t bsize = C.GetFirstSize(); 

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D( A.GetHSize(), A.GetWSize() );
   cudaStream_t s = A.GetComputeStream();

   for(size_t event = 0; event < bsize; event++) {
      // need to implement tensor kernel 
      // ::TMVA::DNN::Cuda::Downsample<<<gridDims, blockDims, 0, s>>>(mA.GetDataPointer(), mB.GetDataPointer(),
      //                                                           mC.GetDataPointer(), depth, imgHeight, imgWidth,
      //                                                           fltHeight, fltWidth, strideRows, strideCols);
      ::TMVA::DNN::Cuda::Downsample<<<gridDims, blockDims, 0, s>>>(A.GetDataPointerAt(event), B.GetDataPointerAt(event),
                                                                 C.GetDataPointerAt(event), depth, imgHeight, imgWidth,
                                                                 fltHeight, fltWidth, strideRows, strideCols);
   }
}
//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::MaxPoolLayerBackward(TCudaTensor<AFloat> & activationGradientsBackward,
                                         const TCudaTensor<AFloat> & activationGradients,
                                         const TCudaTensor<AFloat> & indexMatrix,
                                         size_t imgHeight,
                                         size_t imgWidth,
                                         size_t fltHeight,
                                         size_t fltWidth,
                                         size_t strideRows,
                                         size_t strideCols,
                                         size_t /* nLocalViews */)
{
   size_t depth = activationGradientsBackward.GetCSize();
   size_t bsize = activationGradients.GetFirstSize(); 

   dim3 blockDims = TDevice::BlockDims2D();
   // activationGradientsBackward.GetHSize() should be equal to depth
   dim3 gridDims  = TDevice::GridDims2D(activationGradientsBackward.GetHSize(), 
                    activationGradientsBackward.GetWSize());
   cudaStream_t s = activationGradientsBackward.GetComputeStream();

   for(size_t event = 0; event < bsize; event++) {

      ::TMVA::DNN::Cuda::MaxPoolBackward<<<gridDims, blockDims, 0, s>>>(activationGradientsBackward.GetDataPointerAt(event),
                                                                     activationGradients.GetDataPointerAt(event),
                                                                     indexMatrix.GetDataPointerAt(event),
                                                                     depth, imgHeight, imgWidth, fltHeight, fltWidth,
                                                                     strideRows, strideCols);
   }
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
void TCuda<AReal>::Rearrange(TCudaTensor<AReal> &out, const TCudaTensor<AReal> &in)
{
   // B x T x D out --- T x B x D in*/

   // need to implement (usa CPu impl). Needs by RNN
   out = in; 

   // size_t B = out.size();
   // size_t T = out[0].GetNrows();
   // size_t D = out[0].GetNcols();
   // if ((T != in.size()) || (B != in[0].GetNrows()) 
   //     || (D != in[0].GetNcols())) {
   //    std::cout << "Incompatible Dimensions\n"
   //       << in.size() << "x" << in[0].GetNrows() << "x" << in[0].GetNcols() 
   //       << " --> " << B << "x" << T << "x" << D << "\n";
   //    return;
   // }
   // for (size_t i = 0; i < B; ++i) {
   //    for (size_t j = 0; j < T; ++j) {
   //       for (size_t k = 0; k < D; ++k) {
   //          out[i](j, k) = in[j](i, k);
   //       }
   //    }
   // }
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
void TCuda<AFloat>::Flatten(TCudaTensor<AFloat> &A,
                            const TCudaTensor<AFloat> &B)
{
   // flatten B: ( B x C x HW ) in ( 1, B , CHW)
   size_t size = B.GetFirstSize();   // B size
   size_t nRows = B.GetCSize();      // C size
   size_t nCols = B.GetWSize();      // HW size

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A.GetHSize(), A.GetWSize());
   cudaStream_t s = A.GetComputeStream();

   // Get raw pointers from a vector of matrices - this is more challenging than it sounds.
   //
   // Attention: While `TCudaMatrix.GetDataPointer() returns a pointer to device memory,
   //            std::vector (and its .data() raw pointer) resides on host memory. Therefore
   //            we need to manually copy these pointers to the device prior to invoking the kernel.

   // const AFloat ** dB; // device pointer to device pointers.S
   // const AFloat ** hB = new const AFloat * [size]; // host pointer to device pointers.

   // cudaMalloc(&dB, sizeof(AFloat *) * size);
   // for(size_t i = 0; i < size; ++i) {
   //    hB[i] = B[i].GetDataPointer();
   // }

   // cudaMemcpy(dB, hB, sizeof(AFloat *) * size, cudaMemcpyHostToDevice);
   //std::cout << "flatten from : " << size << " , " << nRows << " , " << nCols << std::endl;

   
   // for (size_t i = 0; i < size; i++) {
   //    for (size_t j = 0; j < nRows; j++) {
   //       for (size_t k = 0; k < nCols; k++) {
   //          A( 0, i, j * nCols + k) = B(i, j, k);
   //       }  
   //    }
   // }

   //PrintTensor(A, "manual reshape");

   // to be fixed !!!
   // Launch the kernel using our device pointers.
   ::TMVA::DNN::Cuda::Flatten<<<gridDims, blockDims>>>(A.GetDataPointer(), B.GetDataPointer(), size, nRows, nCols);

   //PrintTensor(A, "kernel reshape");


   // delete [] hB;
   // cudaFree(dB);
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
void TCuda<AFloat>::Deflatten(TCudaTensor<AFloat> &A,
                              const TCudaTensor<AFloat> &B)
{
    size_t size = A.GetFirstSize();   // B size
    size_t nRows = A.GetCSize();      // C size
    size_t nCols = A.GetWSize();      // HW size

    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(B.GetHSize(), B.GetWSize());
    cudaStream_t s = B.GetComputeStream();

    //std::cout << "Deflatten to " << size << " , " << nRows << "  " << nCols << std::endl;

    // Get raw pointers from a vector of matrices - this is more challenging than it sounds.
    //
    // Attention: While `TCudaMatrix.GetDataPointer() returns a pointer to device memory,
    //            std::vector (and its .data() raw pointer) resides on host memory. Therefore
    //            we need to manually copy these pointers to the device prior to invoking the kernel.

   //  AFloat ** dA; // device pointer to device pointers.
   //  AFloat ** hA = new AFloat * [size]; // host pointer to device pointers.

   //  cudaMalloc(&dA, sizeof(AFloat *) * size);

   //  for(size_t i = 0; i < size; ++i) {
   //      hA[i] = A[i].GetDataPointer();
   //  }

   //  cudaMemcpy(dA, hA, sizeof(AFloat *) * size, cudaMemcpyHostToDevice);

    // Launch the kernel using our device pointers.
   ::TMVA::DNN::Cuda::Deflatten<<<gridDims, blockDims>>>(A.GetDataPointer(), B.GetDataPointer(), size, nRows, nCols);

   // assert (  B.GetFirstSize() == 1);
   // assert (  B.GetHSize() == size);
   // assert (  B.GetWSize() == nRows*nCols);
   // for (size_t i = 0; i < (size_t)size; i++) {
   //    for (size_t j = 0; j < (size_t)nRows; j++) {
   //       for (size_t k = 0; k < (size_t)nCols; k++) {
   //             A(i, j, k) = B(0, i, j * nCols + k);
   //       }
   //    }
   // }


   //  cudaFree(dA); 
   //  delete [] hA; 
}

} // namespace DNN
} // namespace TMVA
