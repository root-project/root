// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Implementation of the functions required for the forward and     //
// backward propagation of activations through a neural network for //
// the reference implementation.                                    //
//////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/DNN/Architectures/Cpu/Blas.h"

namespace TMVA
{
namespace DNN
{

template<typename AFloat>
void TCpu<AFloat>::MultiplyTranspose(TCpuMatrix<AFloat> &output,
                                     const TCpuMatrix<AFloat> &input,
                                     const TCpuMatrix<AFloat> &Weights)
{
    int m = (int) input.GetNrows();
    int k = (int) input.GetNcols();
    int n = (int) Weights.GetNrows();

    char transa = 'N';
    char transb = 'T';

    AFloat alpha = 1.0;
    AFloat beta  = 0.0;

    const AFloat *A = input.GetRawDataPointer();
    const AFloat *B = Weights.GetRawDataPointer();
          AFloat *C = output.GetRawDataPointer();

    ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha,
                            A, &m, B, &n, &beta, C, &m);
}

template<typename AFloat>
void TCpu<AFloat>::AddRowWise(
    TCpuMatrix<AFloat> &output,
    const TCpuMatrix<AFloat> &biases)
{
    int m = (int) output.GetNrows();
    int n = (int) output.GetNcols();

    int inc = 1.0;
    AFloat alpha = 1.0;

          AFloat * A = output.GetRawDataPointer();
    const AFloat * x = TCpuMatrix<AFloat>::GetOnePointer();
    const AFloat * y = biases.GetRawDataPointer();

    ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template<typename AFloat>
void TCpu<AFloat>::Backward(
    TCpuMatrix<AFloat> & activationGradientsBackward,
    TCpuMatrix<AFloat> & weightGradients,
    TCpuMatrix<AFloat> & biasGradients,
    TCpuMatrix<AFloat> & df,
    const TCpuMatrix<AFloat> & activationGradients,
    const TCpuMatrix<AFloat> & weights,
    const TCpuMatrix<AFloat> & activationsBackward)
{
   // Compute element-wise product.
   Hadamard(df, activationGradients);

   // Activation gradients.
   if (activationGradientsBackward.GetNElements() > 0)
       Multiply(activationGradientsBackward, df, weights);

   // Weight gradients.
   if (weightGradients.GetNElements() > 0)
       TransposeMultiply(weightGradients, df, activationsBackward);

   // Bias gradients.
   if (biasGradients.GetNElements() > 0)
       SumColumns(biasGradients, df);
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Im2col(TCpuMatrix<AFloat> &A, TCpuMatrix<AFloat> &B, size_t imgHeight, size_t imgWidth,
                          size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                          size_t zeroPaddingHeight, size_t zeroPaddingWidth)
{
      // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   // convolution centers
   for (int i = -zeroPaddingHeight + fltHeight / 2; i <= imgHeightBound; i += strideRows) {
      for (int j = -zeroPaddingWidth + fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;

         // within the local view
         for (int m = 0; m < B.GetNrows(); m++) {
            for (int k = i - fltHeight / 2; k <= i + (fltHeight - 1) / 2; k++) {
               for (int l = j - fltWidth / 2; l <= j + (fltWidth - 1) / 2; l++) {

                  // Check the boundaries
                  if (k < 0 || k >= imgHeight || l < 0 || l >= imgWidth)
                     A(currLocalView, currLocalViewPixel++) = 0;
                  else
                     A(currLocalView, currLocalViewPixel++) = B(m, k * imgWidth + l);
               }
            }
         }

         currLocalView++;
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::RotateWeights(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B, size_t filterDepth,
                                 size_t filterHeight, size_t filterWidth, size_t numFilters)
{
   size_t jump = filterHeight * filterWidth;
   for (size_t j = 0; j < filterDepth; j++) {
      for (size_t k = 0; k < numFilters; k++) {
         for (size_t i = 0; i < jump; i++) {
            A(j, k * jump + i) = B(k, ((j + 1) * jump - 1) - i);
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::AddConvBiases(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &biases)
{
   int m = (int)output.GetNrows();
   int n = (int)output.GetNcols();

   int inc = 1.0;
   AFloat alpha = 1.0;

   AFloat *A = output.GetRawDataPointer();
   const AFloat *x = biases.GetRawDataPointer();
   const AFloat *y = TCpuMatrix<AFloat>::GetOnePointer();

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::ConvLayerBackward(std::vector<TCpuMatrix<AFloat>> &activationGradientsBackward,
                                     TCpuMatrix<AFloat> &weightGradients, TCpuMatrix<AFloat> &biasGradients,
                                     std::vector<TCpuMatrix<AFloat>> &df,
                                     const std::vector<TCpuMatrix<AFloat>> &activationGradients,
                                     const TCpuMatrix<AFloat> &weights,
                                     const std::vector<TCpuMatrix<AFloat>> &activationsBackward, size_t batchSize,
                                     size_t inputHeight, size_t inputWidth, size_t depth, size_t height, size_t width,
                                     size_t filterDepth, size_t filterHeight, size_t filterWidth, size_t nLocalViews)
{
   // Update derivatives
   size_t m, n;
   m = activationGradients[0].GetNrows();
   n = activationGradients[0].GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      // Compute element-wise product.
      Hadamard(df[i], activationGradients[i]);
   }

   // Calculate the activation gradients of the previous layer
   CalculateConvActivationGradients(activationGradientsBackward, df, weights, batchSize, inputHeight, inputWidth,
                                    depth, height, width, filterDepth, filterHeight, filterWidth);

   // Calculate the weight gradients
   CalculateConvWeightGradients(weightGradients, df, activationsBackward, batchSize, inputHeight, inputWidth, depth,
                                height, width, filterDepth, filterHeight, filterWidth, nLocalViews);

   // Calculate the bias gradients
   CalculateConvBiasGradients(biasGradients, df, batchSize, depth, nLocalViews);
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvActivationGradients(std::vector<TCpuMatrix<AFloat>> &activationGradientsBackward,
                                                    std::vector<TCpuMatrix<AFloat>> &df,
                                                    const TCpuMatrix<AFloat> &weights, size_t batchSize,
                                                    size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                    size_t width, size_t filterDepth, size_t filterHeight,
                                                    size_t filterWidth)
{
   // Transform the weights
   TCpuMatrix<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());

   // Calculate the zero paddings
   size_t tempZeroPaddingHeight = (size_t)(floor((inputHeight - height + filterHeight - 1) / 2));
   size_t tempZeroPaddingWidth = (size_t)(floor((inputWidth - width + filterWidth - 1) / 2));

   // Calculate the number of local views and the number of pixles in each view
   size_t tempNLocalViews = inputHeight * inputWidth;
   size_t tempNLocalViewPixels = depth * filterHeight * filterWidth;

   size_t tempStrideRows = 1;
   size_t tempStrideCols = 1;

   // An entire convolution follows
   for (size_t i = 0; i < batchSize; i++) {
      TCpuMatrix<AFloat> dfTr(tempNLocalViews, tempNLocalViewPixels);
      Im2col(dfTr, df[i], inputHeight, inputWidth, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);

      TransposeMultiply(activationGradientsBackward[i], rotWeights, dfTr);
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvWeightGradients(TCpuMatrix<AFloat> &weightGradients,
                                                std::vector<TCpuMatrix<AFloat>> &df,
                                                const std::vector<TCpuMatrix<AFloat>> &activationsBackward,
                                                size_t batchSize, size_t inputHeight, size_t inputWidth, size_t depth,
                                                size_t height, size_t width, size_t filterDepth, size_t filterHeight,
                                                size_t filterWidth, size_t nLocalViews)
{
   // reinitialize the weight gradients to 0
   for (size_t i = 0; i < depth; i++) {
      for (size_t j = 0; j < nLocalViews; j++) {
         weightGradients(i, j) = 0;
      }
   }

   for (size_t i = 0; i < batchSize; i++) {
      // Calculate the zero paddings
      size_t tempZeroPaddingHeight = (filterHeight - height + inputHeight - 1) / 2;
      size_t tempZeroPaddingWidth = (filterWidth - width + inputWidth - 1) / 2;

      size_t tempNLocalViews = filterHeight * filterWidth;
      size_t tempNLocalViewPixels = inputHeight * inputWidth;

      size_t tempStrideRows = 1;
      size_t tempStrideCols = 1;

      for (size_t j = 0; j < depth; j++) {

         // row matrix
         TCpuMatrix<AFloat> rowDelta(1, nLocalViews);
         for (size_t k = 0; k < nLocalViews; k++) {
            rowDelta(0, k) = df[i](j, k);
         }

         // convolution
         TCpuMatrix<AFloat> res(filterDepth, filterHeight * filterWidth);

         TCpuMatrix<AFloat> rowDeltaTr(tempNLocalViews, tempNLocalViewPixels);
         Im2col(rowDeltaTr, rowDelta, height, width, inputHeight, inputWidth, tempStrideRows, tempStrideCols,
                tempZeroPaddingHeight, tempZeroPaddingWidth);

        TransposeMultiply(res, activationsBackward[i], rowDeltaTr);

         for (size_t k = 0; k < filterDepth; k++) {
            for (size_t l = 0; l < filterHeight * filterWidth; l++) {
               weightGradients(j, k * filterDepth + l) += res(k, (tempNLocalViews - 1) - l);
            }
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvBiasGradients(TCpuMatrix<AFloat> &biasGradients, std::vector<TCpuMatrix<AFloat>> &df,
                                              size_t batchSize, size_t depth, size_t nLocalViews)
{
   for (size_t i = 0; i < depth; i++) {
      AFloat sum = 0;
      for (size_t j = 0; j < nLocalViews; j++) {
         for (size_t k = 0; k < batchSize; k++) {
            sum += df[k](i, j);
         }
      }
      biasGradients(i, 0) = sum;
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Downsample(TCpuMatrix<AFloat> &A, TCpuMatrix<AFloat> &B, const TCpuMatrix<AFloat> &C,
                              size_t imgHeight, size_t imgWidth, size_t fltHeight, size_t fltWidth, size_t strideRows,
                              size_t strideCols)
{
   // image boudaries
   int imgHeightBound = imgHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   // centers
   for (int i = fltHeight / 2; i <= imgHeightBound; i += strideRows) {
      for (int j = fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         // within local views
         for (int m = 0; m < C.GetNrows(); m++) {
            AFloat value = -std::numeric_limits<AFloat>::max();

            for (int k = i - fltHeight / 2; k <= i + (fltHeight - 1) / 2; k++) {
               for (int l = j - fltWidth / 2; l <= j + (fltWidth - 1) / 2; l++) {
                  if (C(m, k * imgWidth + l) > value) {
                     value = C(m, k * imgWidth + l);
                     B(m, currLocalView) = k * imgWidth + l;
                  }
               }
            }
            A(m, currLocalView) = value;
         }
         currLocalView++;
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::MaxPoolLayerBackward(std::vector<TCpuMatrix<AFloat>> &activationGradientsBackward,
                                        const std::vector<TCpuMatrix<AFloat>> &activationGradients,
                                        const std::vector<TCpuMatrix<AFloat>> &indexMatrix, size_t batchSize,
                                        size_t depth, size_t nLocalViews)
{
   for (size_t i = 0; i < batchSize; i++) {
      for (size_t j = 0; j < depth; j++) {

         // initialize to zeros
         for (size_t t = 0; t < (size_t)activationGradientsBackward[i].GetNcols(); t++) {
            activationGradientsBackward[i](j, t) = 0;
         }

         // set values
         for (size_t k = 0; k < nLocalViews; k++) {
            AFloat grad = activationGradients[i](j, k);
            size_t winningIdx = indexMatrix[i](j, k);
            activationGradientsBackward[i](j, winningIdx) = grad;
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Reshape(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B)
{
   size_t nColsA = A.GetNcols();
   size_t nColsB = B.GetNcols();

   for (size_t i = 0; i < A.GetNrows(); i++) {
      for (size_t j = 0; j < A.GetNcols(); j++) {
         size_t nElem = i * nColsA + j;
         A(i, j) = B(nElem / nColsB, (nElem - 1) % nColsB);
      }
   }
}

} // namespace DNN
} // namespace TMVA
