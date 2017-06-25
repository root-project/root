// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski 31/05/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the Convolution functions for the reference //
 // implementation.                                              //
 //////////////////////////////////////////////////////////////////


#include "TMVA/DNN/Architectures/Reference.h"
#include <limits>
#include <math.h>

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Im2col(TMatrixT<Real_t> &A,
                                TMatrixT<Real_t> &B,
                                size_t imgHeight,
                                size_t imgWidth,
                                size_t fltHeight,
                                size_t fltWidth,
                                size_t strideRows,
                                size_t strideCols,
                                size_t zeroPaddingHeight,
                                size_t zeroPaddingWidth)
{
   // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;
    
   // convolution centers
   for(int i = -zeroPaddingHeight + fltHeight / 2; i <= imgHeightBound; i += strideRows) {
      for(int j = -zeroPaddingWidth + fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;
         
         // within the local view
         for(size_t m = 0; m < (size_t) B.GetNrows(); m++){
            for(size_t k = i - fltHeight / 2; k <= i + (fltHeight - 1) / 2; k++) {
               for(size_t l = j - fltWidth / 2; l <= j + (fltWidth - 1) / 2; l++) {
                  
                  // Check the boundaries
                  if(k < 0 || k >= imgHeight || l < 0 || l >= imgWidth)
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

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::RotateWeights(TMatrixT<Real_t> &A,
                                       const TMatrixT<Real_t> &B,
                                       size_t filterDepth,
                                       size_t filterHeight,
                                       size_t filterWidth,
                                       size_t numFilters)
{
   size_t jump = filterHeight * filterWidth;
   for(size_t j = 0; j < filterDepth; j++) {
      for(size_t k = 0; k < numFilters; k++) {
         for(size_t i = 0; i < jump; i++) {
            A(j, k * jump + i) = B(k, ((j + 1) * jump - 1) - i);
         }
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Flatten(TMatrixT<Real_t> &A,
                                 const std::vector<TMatrixT<Real_t>> B,
                                 size_t size,
                                 size_t nRows,
                                 size_t nCols)
{
   for(size_t i = 0; i < (size_t) size; i++) {
      for(size_t j = 0; j < (size_t) nRows; j++) {
         for(size_t k = 0; k < (size_t) nCols; k++) {
            A(i, j * nCols + k) = B[i](j, k);
         }
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Deflatten(std::vector<TMatrixT<Real_t>> A,
                                   const TMatrixT<Real_t> &B,
                                   size_t size,
                                   size_t nRows,
                                   size_t nCols)
{
   for(size_t i = 0; i < (size_t) size; i++){
      for(size_t j = 0; j < (size_t) nRows; j++){
         for(size_t k = 0; k < (size_t) nCols; k++){
            A[i](j, k) = B(i, j * nCols + k);
         }
      }
   }
}

//______________________________________________________________________________
template<typename Scalar_t>
void TReference<Scalar_t>::ConvLayerBackward(std::vector<TMatrixT<Scalar_t>> &activation_gradients_backward,
                                             TMatrixT<Scalar_t> & weight_gradients,
                                             TMatrixT<Scalar_t> & bias_gradients,
                                             std::vector<TMatrixT<Scalar_t>> &df,
                                             const std::vector<TMatrixT<Scalar_t>> &activation_gradients,
                                             const TMatrixT<Scalar_t> & weights,
                                             const std::vector<TMatrixT<Scalar_t>> &activations_backward,
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
        
   // Update derivatives
   size_t m, n;
   m = activation_gradients[0].GetNrows();
   n = activation_gradients[0].GetNcols();
        
        
   for(size_t i = 0; i < batchSize; i++) {
      for(size_t j = 0; j < (size_t) m; j++) {
         for(size_t k = 0; k < (size_t) n; k++) {
            df[i](j, k) *= activation_gradients[i](j, k);
         }
      }
   }
        
   // Calculate the activation gradients of the previous layer
   CalculateConvActivationGradients(activation_gradients_backward, df, weights,
                                    batchSize, inputHeight, inputWidth, depth,
                                    height, width, filterDepth, filterHeight, filterWidth);
        
   // Calculate the weight gradients
   CalculateConvWeightGradients(weight_gradients, df, activations_backward, batchSize,
                                inputHeight, inputWidth, depth, height, width,
                                filterDepth, filterHeight, filterWidth, nLocalViews);
        
   // Calculate the bias gradients
   CalculateConvBiasGradients(bias_gradients, df, batchSize, depth, nLocalViews);
}

//______________________________________________________________________________
template<typename Scalar_t>
void TReference<Scalar_t>::CalculateConvActivationGradients(
                                    std::vector<TMatrixT<Scalar_t>> &activation_gradients_backward,
                                    std::vector<TMatrixT<Scalar_t>> &df,
                                    const TMatrixT<Scalar_t> & weights,
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
        
   // Transform the weights
   TMatrixT<Scalar_t> rotWeights(filterDepth, depth * filterHeight * filterWidth);
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
   for(size_t i = 0; i < batchSize; i++) {
      TMatrixT<Scalar_t> dfTr(tempNLocalViews, tempNLocalViewPixels);
      Im2col(dfTr, df[i], inputHeight, inputWidth, filterHeight, filterWidth,
             tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
            
      activation_gradients_backward[i].MultT(rotWeights, dfTr);
   }
}

//______________________________________________________________________________
template<typename Scalar_t>
void TReference<Scalar_t>::CalculateConvWeightGradients(TMatrixT<Scalar_t> & weight_gradients,
                                    std::vector<TMatrixT<Scalar_t>> &df,
                                    const std::vector<TMatrixT<Scalar_t>> &activations_backward,
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
   for(size_t i = 0; i < depth; i++){
      for(size_t j = 0; j < nLocalViews; j++){
         weight_gradients(i, j) = 0;
      }
   }
        
   for(size_t i = 0; i < batchSize; i++) {
      // Calculate the zero paddings
      size_t tempZeroPaddingHeight = (filterHeight - height + inputHeight - 1) / 2;
      size_t tempZeroPaddingWidth = (filterWidth - width + inputWidth - 1) / 2;
            
      size_t tempNLocalViews = filterHeight * filterWidth;
      size_t tempNLocalViewPixels = inputHeight * inputWidth;
            
      size_t tempStrideRows = 1;
      size_t tempStrideCols = 1;
            
      for(size_t j = 0; j < depth; j++) {
                
         // row matrix
         TMatrixT<Scalar_t> rowDelta(1, nLocalViews);
         for(size_t k = 0; k < nLocalViews; k++){
            rowDelta(0, k) = df[i](j, k);
         }
                
         // convolution
         TMatrixT<Scalar_t> res(filterDepth, filterHeight * filterWidth);
                
         TMatrixT<Scalar_t> rowDeltaTr(tempNLocalViews, tempNLocalViewPixels);
         Im2col(rowDeltaTr, rowDelta, height, width, inputHeight, inputWidth,
                tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
                
         res.MultT(activations_backward[i], rowDeltaTr);
                
         for(size_t k = 0; k < filterDepth; k++) {
            for(size_t l = 0; l < filterHeight * filterWidth; l++) {
               weight_gradients(j, k * filterDepth + l) += res(k, (tempNLocalViews - 1) - l);
            }
         }
      }
   }
}

//______________________________________________________________________________
template<typename Scalar_t>
void TReference<Scalar_t>::CalculateConvBiasGradients(TMatrixT<Scalar_t> & bias_gradients,
                                                      std::vector<TMatrixT<Scalar_t>> &df,
                                                      size_t batchSize,
                                                      size_t depth,
                                                      size_t nLocalViews)
{
   for(size_t i = 0; i < depth; i++) {
      Scalar_t sum = 0;
      for(size_t j = 0; j < nLocalViews; j++) {
         for(size_t k = 0; k < batchSize; k++) {
            sum += df[k](i, j);
         }
      }
      bias_gradients(i, 0) = sum;
   }
}

//______________________________________________________________________________
template<typename Scalar_t>
void TReference<Scalar_t>::AddConvBiases(TMatrixT<Scalar_t> &output,
                                         const TMatrixT<Scalar_t> &biases)
{
   for (size_t i = 0; i < (size_t) output.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) output.GetNcols(); j++) {
         output(i,j) += biases(i,0);
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Downsample(TMatrixT<Real_t> &A,
                                    TMatrixT<Real_t> &B,
                                    const TMatrixT<Real_t> &C,
                                    size_t imgHeight,
                                    size_t imgWidth,
                                    size_t fltHeight,
                                    size_t fltWidth,
                                    size_t strideRows,
                                    size_t strideCols)
{
   // image boudaries
   int imgHeightBound = imgHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;
        
   // centers
   for(int i = fltHeight / 2; i <= imgHeightBound; i+= strideRows) {
      for(int j = fltWidth / 2; j <= imgWidthBound; j += strideCols) {
         // within local views
         for(size_t m = 0; m < (size_t) C.GetNrows(); m++){
            Real_t value = -std::numeric_limits<Real_t>::max();
                
            for(size_t k = i - fltHeight / 2; k <= i + (fltHeight - 1) / 2; k++) {
            for(size_t l = j - fltWidth / 2; l <= j + (fltWidth - 1) / 2; l++) {
               if(C(m, k * imgWidth + l) > value) {
                  value = C(m, k * imgWidth + l);
                  B(m, currLocalView) = k * imgWidth  + l;
               }
            }
         }
         A(m, currLocalView) = value;
      }
      currLocalView++;
      }
   }
}

//______________________________________________________________________________
template<typename Scalar_t>
void TReference<Scalar_t>::PoolLayerBackward(
                            std::vector<TMatrixT<Scalar_t>> & activationGradientsBackward,
                            const std::vector<TMatrixT<Scalar_t>> & activationGradients,
                            const std::vector<TMatrixT<Scalar_t>> & indexMatrix,
                            size_t batchSize,
                            size_t depth,
                            size_t nLocalViews)
{
   for(size_t i = 0; i < batchSize; i++) {
      for(size_t j = 0; j < depth; j++) {
            
         // initialize to zeros
         for(size_t t  = 0; t < (size_t) activationGradientsBackward[i].GetNcols(); t++){
            activationGradientsBackward[i][j][t] = 0;
         }
            
         // set values
         for(size_t k = 0; k < nLocalViews; k++) {
            Scalar_t grad = activationGradients[i][j][k];
            size_t winningIdx = indexMatrix[i][j][k];
            activationGradientsBackward[i][j][winningIdx] = grad;
         }
      }
   }
}
    
} // namespace DNN
} // namespace TMVA
