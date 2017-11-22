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
void TCpu<AFloat>::Im2col(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B, size_t imgHeight, size_t imgWidth,
                          size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                          size_t zeroPaddingHeight, size_t zeroPaddingWidth)
{

   // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   const int halfFltHeight =  fltHeight / 2;
   const int halfFltWidth =  fltWidth / 2;
   const int halfFltHeightM1 = (fltHeight - 1) / 2;
   const int halfFltWidthM1 = (fltWidth - 1) / 2;
   const int nRowsInput = B.GetNrows();
   const int nColsInput = B.GetNcols(); 
   const int nRowsOutput = A.GetNrows();
   const int nColsOutput = A.GetNcols(); 

   // convolution centers
   for (int i = halfFltHeight -zeroPaddingHeight; i <= imgHeightBound; i += strideRows) {
      for (int j = halfFltWidth -zeroPaddingWidth ; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;

         // within the local view
         R__ASSERT((int) currLocalView < nRowsOutput );

         for (int m = 0; m < nRowsInput; m++) {
            for (int k = i - halfFltHeight  ; k <= Int_t(i + halfFltHeightM1 ); k++) {
               int kstep = k * imgWidth;
               for (int l = j - halfFltWidth ; l <= Int_t(j + halfFltWidthM1); l++) {

                  // Check the boundaries
                  R__ASSERT(currLocalViewPixel < nColsOutput );
                  //R__ASSERT(k * imgWidth + l < B.GetNcols());
                  if (k < 0 || k >= (Int_t)imgHeight || l < 0 || l >= (Int_t)imgWidth || kstep + l >=  nColsInput)
                     A(currLocalView, currLocalViewPixel++) = 0;
                  else
                     A(currLocalView, currLocalViewPixel++) = B(m, kstep + l);
               }
            }
         }
         //std::cout << " i " << i << "  " << j << " increment currLocalView " << currLocalView << std::endl;
         currLocalView++;
      }
   }
   //PrintMatrix(A,"FromIm2Col"); 
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Im2colIndices(std::vector<int> &V, const TCpuMatrix<AFloat> &B, size_t nLocalViews, size_t imgHeight, size_t imgWidth,
                          size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols,
                           size_t zeroPaddingHeight, size_t zeroPaddingWidth)
{

   // image boudaries
   int imgHeightBound = imgHeight + zeroPaddingHeight - (fltHeight - 1) / 2 - 1;
   int imgWidthBound = imgWidth + zeroPaddingWidth - (fltWidth - 1) / 2 - 1;
   size_t currLocalView = 0;

   const int halfFltHeight =  fltHeight / 2;
   const int halfFltWidth =  fltWidth / 2;
   const int halfFltHeightM1 = (fltHeight - 1) / 2;
   const int halfFltWidthM1 = (fltWidth - 1) / 2;
   const int nRowsInput = B.GetNrows();
   const int nColsInput = B.GetNcols();
   const size_t nSizeOutput = V.size();
   const int npixels =  nRowsInput * fltHeight * fltWidth;
   // const int nRowsOutput = A.GetNrows();
   // const int nColsOutput = A.GetNcols(); 

   // convolution centers
   for (int i = halfFltHeight -zeroPaddingHeight; i <= imgHeightBound; i += strideRows) {
      for (int j = halfFltWidth -zeroPaddingWidth ; j <= imgWidthBound; j += strideCols) {
         size_t currLocalViewPixel = 0;

         // within the local view
         //R__ASSERT((int) currLocalView < nRowsOutput );

         for (int m = 0; m < nRowsInput; m++) {
            for (int k = i - halfFltHeight  ; k <= Int_t(i + halfFltHeightM1 ); k++) {
               int kstep = k * imgWidth;
               for (int l = j - halfFltWidth ; l <= Int_t(j + halfFltWidthM1); l++) {

                  // Check the boundaries
                  //R__ASSERT(currLocalViewPixel < nColsOutput );
                  R__ASSERT(currLocalView * npixels + currLocalViewPixel < nSizeOutput ); 
                  if (k < 0 || k >= (Int_t)imgHeight || l < 0 || l >= (Int_t)imgWidth || kstep + l >=  nColsInput)
                     //V[currLocalView * npixels + currLocalViewPixel]=-1;
                     V[currLocalViewPixel * nLocalViews + currLocalView] = -1;
                  else
                     V[currLocalViewPixel * nLocalViews + currLocalView]= ( kstep + l) * nRowsInput + m;
                  
                  currLocalViewPixel++;
               }
            }
         }
         currLocalView++;
      }
   }
}
template <typename AFloat>
void TCpu<AFloat>::Im2colFast(TCpuMatrix<AFloat> &A, const TCpuMatrix<AFloat> &B, const std::vector<int> &V) 
{
   size_t  n = V.size(); 
   R__ASSERT( n == A.GetNcols() * A.GetNrows() );
   AFloat *  a = A.GetRawDataPointer();
   const AFloat *  b = B.GetRawDataPointer();

//#define DL_USE_MTE  
   // parallel execution
#ifdef DL_USE_MTE
   const size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(n);

   auto f = [&](UInt_t workerID)
   {
      for (size_t j = 0; j < nsteps; ++j) {
         size_t ii = workerID+j;
         if (ii >= n) break;
         int idx = V[ii]; 
         if (idx >= 0) a[ii] = b[idx];
         else a[ii] = 0;
      }
      return 0;
   };

   A.GetThreadExecutor().Foreach(f, ROOT::TSeqI(0,n,nsteps) );

#else
   //serial execution
   for (size_t ii = 0; ii < n; ++ii) {
      int idx = V[ii]; 
      if (idx >= 0) a[ii] = b[idx];
      else a[ii] = 0;
   }

#endif
   // PrintMatrix(A,"FromFastIm2Col");
   // PrintMatrix(B,"input to Im2Col");
   // std::cout << "V vector " << V.size() << std::endl;
   // for ( int i = 0; i < n; ++i) {
   //    std::cout << V[i] << "  ";
   // }
   // std::cout << std::endl;
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
void TCpu<AFloat>::ConvLayerForward(std::vector<TCpuMatrix<AFloat>> & output, std::vector<TCpuMatrix<AFloat>> & derivatives,
                                    const std::vector<TCpuMatrix<AFloat>> &input,
                                    const TCpuMatrix<AFloat> & weights, const TCpuMatrix<AFloat> & biases,
                                    EActivationFunction activFunc, const std::vector<int> & vIndices,
                                    size_t nlocalViews, size_t nlocalViewPixels,
                                    AFloat dropoutProbability, bool applyDropout)
{

   //TCpuMatrix<AFloat> inputTr(this->GetNLocalViews(), this->GetNLocalViewPixels());
   auto f = [&] (UInt_t i)
   {
       // if (applyDropout && (dropoutProbability != 1.0)) {
       //    Dropout(input[i], dropoutProbability);
       // }

       TCpuMatrix<AFloat> inputTr(nlocalViews, nlocalViewPixels);
       //inputTr.Zero();   // this is not thread safe

       Im2colFast(inputTr, input[i], vIndices);

       MultiplyTranspose(output[i], weights, inputTr);
       AddConvBiases(output[i], biases);

       evaluateDerivative<TCpu<AFloat>>(derivatives[i], activFunc, output[i]);
       evaluate<TCpu<AFloat>>(output[i], activFunc);
       
   };

   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI(input.size() ) );
   
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
                                                    const std::vector<TCpuMatrix<AFloat>> &df,
                                                    const TCpuMatrix<AFloat> &weights, size_t batchSize,
                                                    size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                    size_t width, size_t filterDepth, size_t filterHeight,
                                                    size_t filterWidth)
{
   if (activationGradientsBackward.size() == 0) return;

   
   // Transform the weights

   //PrintMatrix(weights,"weights");
   // filter depth must be same as input depth
   TCpuMatrix<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());
   //PrintMatrix(rotWeights,"rot-weights");

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

    std::vector<int> vIndices( tempNLocalViews * tempNLocalViewPixels );
    Im2colIndices(vIndices, df[0], tempNLocalViews, height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);


    //for (size_t i = 0; i < batchSize; i++) {
    R__ASSERT(batchSize == df.size() );
    R__ASSERT(batchSize == activationGradientsBackward.size() );
    auto f = [&] (UInt_t i)
   {
   
       // Im2col(dfTr, df[i], height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
       //       tempZeroPaddingHeight, tempZeroPaddingWidth);

      TCpuMatrix<AFloat> dfTr(tempNLocalViews, tempNLocalViewPixels);
      
      Im2colFast(dfTr, df[i], vIndices); 

       //PrintMatrix(df[i],"df[i]");
       //PrintMatrix(dfTr,"dfTr");

       MultiplyTranspose(activationGradientsBackward[i], rotWeights, dfTr);

       //PrintMatrix(activationGradientsBackward[i],"activGrad-result");

   };

    TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI( batchSize ) );
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvWeightGradients(TCpuMatrix<AFloat> &weightGradients,
                                                const std::vector<TCpuMatrix<AFloat>> &df,
                                                const std::vector<TCpuMatrix<AFloat>> &activationsBackward,
                                               size_t batchSize, size_t inputHeight, size_t inputWidth, size_t depth,
                                                size_t height, size_t width, size_t filterDepth, size_t filterHeight,
                                                size_t filterWidth, size_t nLocalViews)
{
   // reinitialize the weight gradients to 0
   weightGradients.Zero();

   const size_t filterSize = filterHeight * filterWidth;
   const size_t nLocalViewPixels = filterDepth * filterHeight * filterWidth;
   R__ASSERT( weightGradients.GetNcols() == filterDepth * filterHeight * filterWidth);

   const size_t tempStrideRows = 1;
   const size_t tempStrideCols = 1;
      
      // Calculate the zero paddings from the input height and width (assume stride =1 )      
   const size_t tempZeroPaddingHeight = (height - inputHeight + filterHeight - 1) / 2;
   const size_t tempZeroPaddingWidth = (width - inputWidth + filterWidth - 1) / 2;


   // convolution
   
   

   std::vector<int> vIndices(nLocalViews * nLocalViewPixels );
   Im2colIndices(vIndices, activationsBackward[0], nLocalViews, inputHeight, inputWidth, filterHeight , filterWidth,
             tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
   
   //std::cout << "do back-propagation in conv layer - compute weight gradient" << std::endl;

   std::vector< TCpuMatrix<AFloat> > vres;//(batchSize); 
   for (size_t i = 0; i < batchSize; i++) {
      vres.emplace_back(depth, nLocalViewPixels);
      //PrintMatrix(df[i],"df");
      //PrintMatrix(activationsBackward[i],"df");
      
   }
   
   auto fmap = [&](int i) { 
 
      //PrintMatrix(df[i],"df-i");
      TCpuMatrix<AFloat> xTr(nLocalViews, nLocalViewPixels);
      TCpuMatrix<AFloat> res(depth, nLocalViewPixels);

      //computing t he gradient is equivalent of doing a convolution of the input using as conv kernel the delta's (the df[] values) 
      //N.B. only stride values=1 are now supported
 
      //xTr.Zero(); 
      // Im2col(xTr, const_cast<TCpuMatrix<AFloat> &>(activationsBackward[i]), inputHeight, inputWidth, filterHeight , filterWidth,
      //        tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
      Im2colFast(xTr, activationsBackward[i], vIndices);

      //std::cout << "doing im2colfast" << std::endl;
      //PrintMatrix(xTr,"xTr-i");
      //PrintMatrix(activationsBackward[i],"actbackward-i");
      Multiply(vres[i], df[i], xTr);
      //PrintMatrix(vres[i],"res_ofMT");

      return;
      //return res;
   };

   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(fmap, ROOT::TSeqI( batchSize ) );

//   auto freduce = [&](const std::vector<TCpuMatrix<AFloat>> & vres) { 
      R__ASSERT(vres.size() == batchSize); 
      for (size_t i = 0; i < batchSize; i++) {
         //PrintMatrix(vres[i],"res");
         for (size_t j = 0; j < depth; j++) {
            for (size_t k = 0; k < filterDepth; k++) {
               size_t kOffset = k * filterSize; 
               for (size_t l = 0; l < filterSize; l++) {
                  //weightGradients(j, k * (filterHeight * filterWidth) + l) += res(k, (tempNLocalViews - 1) - l);
                  weightGradients(j, kOffset + l) += vres[i](j,  kOffset + l);
               }
            }
         }
         // PrintMatrix(weightGradients,"weights_i");
      }
      //  };
  
   //TCpuMatrix<AFloat>::GetThreadExecutor().MapReduce(fmap, ROOT::TSeqI( batchSize ) , freduce);
   //PrintMatrix(weightGradients,"W-Grad");
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvBiasGradients(TCpuMatrix<AFloat> &biasGradients, const std::vector<TCpuMatrix<AFloat>> &df,
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
