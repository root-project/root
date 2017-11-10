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

namespace TMVA {
namespace DNN {

template <typename AFloat>
void TCpu<AFloat>::MultiplyTranspose(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &input,
                                     const TCpuMatrix<AFloat> &Weights)
{

   int m = (int)input.GetNrows();
   int k = (int)input.GetNcols();
   int n = (int)Weights.GetNrows();

   R__ASSERT((int) output.GetNrows() == m);
   R__ASSERT((int) output.GetNcols() == n);
   R__ASSERT((int) Weights.GetNcols() == k); 

   char transa = 'N';
   char transb = 'T';

   AFloat alpha = 1.0;
   AFloat beta = 0.0;

   const AFloat *A = input.GetRawDataPointer();
   const AFloat *B = Weights.GetRawDataPointer();
   AFloat *C = output.GetRawDataPointer();

   ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha, A, &m, B, &n, &beta, C, &m);
}

template <typename AFloat>
void TCpu<AFloat>::AddRowWise(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &biases)
{
   int m = (int)output.GetNrows();
   int n = (int)output.GetNcols();

   int inc = 1.0;
   AFloat alpha = 1.0;

   AFloat *A = output.GetRawDataPointer();
   const AFloat *x = TCpuMatrix<AFloat>::GetOnePointer();
   const AFloat *y = biases.GetRawDataPointer();

   R__ASSERT(m <= (int)TCpuMatrix<AFloat>::GetOnePointerSize()); 
   R__ASSERT(n <= (int)biases.GetNcols()*biases.GetNrows()); 

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template <typename AFloat>
void TCpu<AFloat>::Backward(TCpuMatrix<AFloat> &activationGradientsBackward, TCpuMatrix<AFloat> &weightGradients,
                            TCpuMatrix<AFloat> &biasGradients, TCpuMatrix<AFloat> &df,
                            const TCpuMatrix<AFloat> &activationGradients, const TCpuMatrix<AFloat> &weights,
                            const TCpuMatrix<AFloat> &activationsBackward)
{
   // Compute element-wise product.
   Hadamard(df, activationGradients);

   // Activation gradients.
   if (activationGradientsBackward.GetNElements() > 0) Multiply(activationGradientsBackward, df, weights);

   // Weight gradients.
   if (weightGradients.GetNElements() > 0) TransposeMultiply(weightGradients, df, activationsBackward);

   // Bias gradients.
   if (biasGradients.GetNElements() > 0) SumColumns(biasGradients, df);
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
         for (int m = 0; m < (Int_t)B.GetNrows(); m++) {
            for (int k = i - fltHeight / 2; k <= Int_t(i + (fltHeight - 1) / 2); k++) {
               for (int l = j - fltWidth / 2; l <= Int_t(j + (fltWidth - 1) / 2); l++) {

                  // Check the boundaries
                  R__ASSERT(currLocalView < A.GetNrows() );
                  R__ASSERT(currLocalViewPixel < A.GetNcols() );
                  //R__ASSERT(k * imgWidth + l < B.GetNcols());
                  if (k < 0 || k >= (Int_t)imgHeight || l < 0 || l >= (Int_t)imgWidth || k * imgWidth + l >=  B.GetNcols())
                     A(currLocalView, currLocalViewPixel++) = 0;
                  else
                     A(currLocalView, currLocalViewPixel++) = B(m, k * imgWidth + l);
               }
            }
         }
         //std::cout << " i " << i << "  " << j << " increment currLocalView " << currLocalView << std::endl;
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
            //A(j, k * jump + i) = B(k, j * jump + i);
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
   //    size_t m, n;
   //    m = activationGradients[0].GetNrows();
   //    n = activationGradients[0].GetNcols();

   for (size_t i = 0; i < batchSize; i++) {
      // Compute element-wise product.
      Hadamard(df[i], activationGradients[i]);
   }

   // Calculate the activation gradients of the previous layer
   CalculateConvActivationGradients(activationGradientsBackward, df, weights, batchSize, inputHeight, inputWidth, depth,
                                                                                         height, width, filterDepth, filterHeight, filterWidth);

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
   if (activationGradientsBackward.size() == 0) return;

   
   // Transform the weights

   PrintMatrix(weights,"weights");
   // filter depth must be same as input depth
   TCpuMatrix<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());
   PrintMatrix(rotWeights,"rot-weights");

   // Calculate the zero paddings
   size_t tempZeroPaddingHeight = (size_t)(floor((inputHeight - height + filterHeight - 1) / 2));
   size_t tempZeroPaddingWidth = (size_t)(floor((inputWidth - width + filterWidth - 1) / 2));

   // size_t tempZeroPaddingHeight = 1;
   // size_t tempZeroPaddingWidth = 1;
   
   // Calculate the number of local views and the number of pixles in each view
   size_t tempNLocalViews = inputHeight * inputWidth;
   size_t tempNLocalViewPixels = depth * filterHeight * filterWidth;

   size_t tempStrideRows = 1;
   size_t tempStrideCols = 1;

   // An entire convolution follows
   for (size_t i = 0; i < batchSize; i++) {
      TCpuMatrix<AFloat> dfTr(tempNLocalViews, tempNLocalViewPixels);
      for (int j = 0; j < dfTr.GetNrows(); ++j) {
         for (int k = 0; k < dfTr.GetNcols(); ++k) {
            dfTr(j,k)  = 0;
         }
      }
      Im2col(dfTr, df[i], height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);

      PrintMatrix(df[i],"df[i]");
      PrintMatrix(dfTr,"dfTr");

      MultiplyTranspose(activationGradientsBackward[i], rotWeights, dfTr);

      PrintMatrix(activationGradientsBackward[i],"activGrad-result");

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
   for (size_t i = 0; i < weightGradients.GetNrows(); i++) {
      for (size_t j = 0; j < weightGradients.GetNcols(); j++) {
         weightGradients(i, j) = 0;
      }
   }

   size_t nLocalViewPixels = filterDepth * filterHeight * filterWidth;
   R__ASSERT( weightGradients.GetNcols() == filterDepth * filterHeight * filterWidth);

   // convolution
   TCpuMatrix<AFloat> res(depth, nLocalViewPixels);
   //std::cout << "do back-propagation in conv layer - compute weight gradient" << std::endl;
   for (size_t i = 0; i < batchSize; i++) {

      size_t tempStrideRows = 1;
      size_t tempStrideCols = 1;
      
      // Calculate the zero paddings from the input height and width (assume stride =1 )      
      size_t tempZeroPaddingHeight = (height - inputHeight + filterHeight - 1) / 2;
      size_t tempZeroPaddingWidth = (width - inputWidth + filterWidth - 1) / 2;

      //PrintMatrix(df[i],"df-i");

      //computing t he gradient is equivalent of doing a convolution of the input using as conv kernel the delta's (the df[] values) 
      //N.B. only stride values=1 are now supported
 
      TCpuMatrix<AFloat> xTr(nLocalViews, nLocalViewPixels);
      Im2col(xTr, const_cast<TCpuMatrix<AFloat> &>(activationsBackward[i]), inputHeight, inputWidth, filterHeight , filterWidth,
             tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);


      //PrintMatrix(xTr,"xTr-i");
      //PrintMatrix(activationsBackward[i],"actbackward-i");
      Multiply(res, df[i], xTr);
      //PrintMatrix(res,"res_ofMT");

      for (size_t j = 0; j < depth; j++) {
         for (size_t k = 0; k < filterDepth; k++) {
            for (size_t l = 0; l < filterHeight * filterWidth; l++) {
               //weightGradients(j, k * (filterHeight * filterWidth) + l) += res(k, (tempNLocalViews - 1) - l);
               weightGradients(j, k * (filterHeight * filterWidth) + l) += res(j,  k * (filterHeight * filterWidth) + l);
            }
         }
      }
  
   }
   //PrintMatrix(weightGradients,"W-Grad");
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
         for (int m = 0; m < (Int_t)C.GetNrows(); m++) {
            AFloat value = -std::numeric_limits<AFloat>::max();

            for (int k = i - fltHeight / 2; k <= Int_t(i + (fltHeight - 1) / 2); k++) {
               for (int l = j - fltWidth / 2; l <= Int_t(j + (fltWidth - 1) / 2); l++) {
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
            activationGradientsBackward[i](j, winningIdx) += grad;
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

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Flatten(TCpuMatrix<AFloat> &A, const std::vector<TCpuMatrix<AFloat>> &B, size_t size, size_t nRows,
                           size_t nCols)
{
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
            A(i, j * nCols + k) = B[i](j, k);
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Deflatten(std::vector<TCpuMatrix<AFloat>> &A, const TCpuMatrix<AFloat> &B, size_t size, size_t nRows,
                             size_t nCols)
{
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
            A[i](j, k) = B(i, j * nCols + k);
         }
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TCpu<AReal>::Rearrange(std::vector<TCpuMatrix<AReal>> &out, const std::vector<TCpuMatrix<AReal>> &in)
{
   // B x T x D out --- T x B x D in*/
   size_t B = out.size();
   size_t T = out[0].GetNrows();
   size_t D = out[0].GetNcols();
   if ((T != in.size()) || (B != in[0].GetNrows()) || (D != in[0].GetNcols())) {
      std::cout << "Incompatible Dimensions\n"
                << in.size() << "x" << in[0].GetNrows() << "x" << in[0].GetNcols() << " --> " << B << "x" << T << "x"
                << D << "\n";
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

} // namespace DNN
} // namespace TMVA
