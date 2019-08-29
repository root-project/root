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


// template <typename AFloat>
// void TCpu<AFloat>::MultiplyTranspose(Tensor_t &output, const Tensor_t &input,
//                                      const TCpuMatrix<AFloat> &Weights)
// {
//    // apply  multiply transpose. Assume tensor has size == 2  and is column major ordering
//    //int m = (int)input.GetNrows();
//    //int k = (int)input.GetNcols();
    
//    int n = (int)Weights.GetNrows();
//    int m = (int)input.GetShape()[0];
//    int k = (int)input.GetShape()[1];

//    if ((int)output.GetShape()[0] != m) {
//       Error("MultiplyTranspose","Invalid input - output  rows  - input:  %d != output : %d",m, (int) output.GetNrows());
//       R__ASSERT((int) GetShape()[0] == m);
//    }
//    if ((int)output.GetShape()[1] != n) {
//       Error("MultiplyTranspose","Invalid output cols or weight  rows  - output cols:  %d != weight rows : %d",(int) output.GetNcols(),n);
//       R__ASSERT((int) GetShape()[1] == n);
//    }
//    if ((int)Weights.GetNcols() != k) {
//       Error("MultiplyTranspose","Invalid input cols or weight cols  - input cols:  %d != weight cols : %d", k, (int) Weights.GetNcols());
//       R__ASSERT((int) Weights.GetNcols() == k); 
//    }

//    char transa = 'N';
//    char transb = 'T';

//    AFloat alpha = 1.0;
//    AFloat beta = 0.0;

//    const AFloat *A = input.GetData();
//    const AFloat *B = Weights.GetRawDataPointer();
//    AFloat *C = output.GetData();

//    ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha, A, &m, B, &n, &beta, C, &m);
// }

// template <typename AFloat>
// void TCpu<AFloat>::AddRowWise(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &biases)
// {
//    int m = (int)output.GetShape()[0];
//    int n = (int)output.GetShape()[1];

//    int inc = 1.0;
//    AFloat alpha = 1.0;

//    AFloat *A = output.GetData();
//    const AFloat *x = TCpuMatrix<AFloat>::GetOnePointer();
//    const AFloat *y = biases.GetRawDataPointer();

//    R__ASSERT(m <= (int)TCpuMatrix<AFloat>::GetOnePointerSize()); 
//    R__ASSERT(n <= (int)(biases.GetNcols()*biases.GetNrows())); 

//    ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
// }

template <typename AFloat>
void TCpu<AFloat>::MultiplyTranspose(TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &input,
                                     const TCpuMatrix<AFloat> &Weights)
{

   int m = (int)input.GetNrows();
   int k = (int)input.GetNcols();
   int n = (int)Weights.GetNrows();

   if ((int)output.GetNrows() != m) {
      Error("MultiplyTranspose","Invalid input - output  rows  - input:  %d != output : %d",m, (int) output.GetNrows());
      R__ASSERT((int) output.GetNrows() == m);
   }
   if ((int)output.GetNcols() != n) {
      Error("MultiplyTranspose","Invalid output cols or weight  rows  - output cols:  %d != weight rows : %d",(int) output.GetNcols(),n);
      R__ASSERT((int) output.GetNcols() == n);
   }
   if ((int)Weights.GetNcols() != k) {
      Error("MultiplyTranspose","Invalid input cols or weight cols  - input cols:  %d != weight cols : %d", k, (int) Weights.GetNcols());
      R__ASSERT((int) Weights.GetNcols() == k); 
   }

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
   R__ASSERT(n <= (int)(biases.GetNcols()*biases.GetNrows())); 

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template <typename AFloat>
void TCpu<AFloat>::Backward(TCpuTensor<AFloat> &activationGradientsBackward, TCpuMatrix<AFloat> &weightGradients,
                            TCpuMatrix<AFloat> &biasGradients, const TCpuTensor<AFloat> &df,
                            const TCpuTensor<AFloat> &activationGradients, const TCpuMatrix<AFloat> &weights,
                            const TCpuTensor<AFloat> &activationsBackward)
{
   // Compute element-wise product.
   //Hadamard(df, activationGradients);

   Matrix_t df_m = df.GetMatrix(); 

   // Activation gradients (exclude if it is first layer)
   if (activationGradientsBackward.GetSize() > 0 ) { 
   
      Matrix_t  activationGradientsBackward_m = activationGradientsBackward.GetMatrix(); 
     
      Multiply(activationGradientsBackward_m, df_m, weights);
   }

   // Weight gradients.
   if (weightGradients.GetNoElements() > 0) TransposeMultiply(weightGradients, df_m, activationsBackward.GetMatrix());

   // Bias gradients.
   if (biasGradients.GetNoElements() > 0) SumColumns(biasGradients, df_m);
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
                  R__ASSERT((int) currLocalViewPixel < nColsOutput );
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
   //TMVA_DNN_PrintTCpuMatrix(A,"FromIm2Col"); 
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
   // TMVA_DNN_PrintTCpuMatrix(A,"FromFastIm2Col");
   // TMVA_DNN_PrintTCpuMatrix(B,"input to Im2Col");
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

   R__ASSERT(m <= (int)biases.GetNoElements() ); 
   R__ASSERT(n <= (int)TCpuMatrix<AFloat>::GetOnePointerSize() ); 

   ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template<typename AFloat>
size_t TCpu<AFloat>::calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   size_t temp = imgDim - fltDim + 2 * padding;
   if (temp % stride || temp + stride <= 0) {
      Fatal("calculateDimension", "Not compatible hyper parameters for layer - (imageDim, filterDim, padding, stride) "
            "%zu, %zu, %zu, %zu", imgDim, fltDim, padding, stride);
   }
   return temp / stride + 1;
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::ConvLayerForward(TCpuTensor<AFloat> & output,
                                    TCpuTensor<AFloat> & inputActivationFunc,
                                    const TCpuTensor<AFloat> &input,
                                    const TCpuMatrix<AFloat> &weights, const TCpuMatrix<AFloat> & biases,
                                    const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                    TCpuTensor<AFloat> & /*  */,
                                    const ConvDescriptors_t & /*descriptors*/,
                                    ConvWorkspace_t & /*workspace*/)
{
   size_t height = calculateDimension(params.inputHeight, params.filterHeight, params.paddingHeight, params.strideRows);
   size_t width = calculateDimension(params.inputWidth, params.filterWidth, params.paddingWidth, params.strideCols);
   size_t nLocalViews = height * width;
   size_t nLocalViewPixels = params.inputDepth * params.filterHeight * params.filterWidth;

   R__ASSERT( input.GetSize() > 0);
   std::vector<int> forwardIndices(nLocalViews * nLocalViewPixels);
   Im2colIndices(forwardIndices, input.At(0).GetMatrix(), nLocalViews, params.inputHeight, params.inputWidth, params.filterHeight,
                 params.filterWidth, params.strideRows, params.strideCols, params.paddingHeight, params.paddingWidth);

   //this should fix multi-thread inizializations of arrays
   TCpuMatrix<AFloat>::InitializeOneVector(nLocalViews);
   TCpuMatrix<AFloat>::InitializeOneVector(output.GetWSize());   // since it is used in AddCOnvBiases


   auto f = [&] (UInt_t i)
   {
       // dropout not yet implemented for CNN
       // if (applyDropout && (dropoutProbability != 1.0)) {
       //    Dropout(input[i], dropoutProbability);
       // }

       TCpuMatrix<AFloat> inputTr(nLocalViews, nLocalViewPixels);
       //inputTr.Zero();   // this is not thread safe

       Im2colFast(inputTr, input.At(i).GetMatrix(), forwardIndices);

       Matrix_t output_m = output.At(i).GetMatrix();
       MultiplyTranspose(output_m, weights, inputTr);
       AddConvBiases(output_m, biases);

   };

   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI(input.GetFirstSize()));

   //evaluateDerivative<TCpu<AFloat>>(derivatives, activFunc, output);
   // need to save output of convolution (input to activation function)
   Copy(inputActivationFunc, output);

   //evaluate<TCpu<AFloat>>(output, activFunc);
   ActivationFunctionForward(output, activFunc, DummyActivationDescriptor());
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::ConvLayerBackward(TCpuTensor<AFloat> &activationGradientsBackward,
                                     TCpuMatrix<AFloat> &weightGradients, TCpuMatrix<AFloat> &biasGradients,
                                     TCpuTensor<AFloat> &inputActivationFunc,
                                     TCpuTensor<AFloat> &activationGradients,
                                     const TCpuMatrix<AFloat> &weights,
                                     const TCpuTensor<AFloat> &activationsBackward,
                                     const Tensor_t & outputTensor,
                                     EActivationFunction activFunc,
                                     const ConvDescriptors_t & /*descriptors*/,
                                     ConvWorkspace_t & /*workspace*/,
                                     size_t batchSize,   size_t inputHeight, 
                                     size_t inputWidth,  size_t depth, 
                                     size_t height,      size_t width,
                                     size_t filterDepth, size_t filterHeight, 
                                     size_t filterWidth, size_t nLocalViews)
{
   // Update derivatives
   //    size_t m, n;
   //    m = activationGradients[0].GetNrows();
   //    n = activationGradients[0].GetNcols();

 
   // Compute activation backward pass  dx = f'(x) * dy  
   //  put resulting dx of activation in activationgradients
   Tensor_t df(activationGradients.GetShape() );   // this is a deep copy, could be put as data member of class
   ActivationFunctionBackward(df, outputTensor, activationGradients, inputActivationFunc, 
                              activFunc, DummyActivationDescriptor() );

   // Hadamard(df, activationGradients);
   

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
void TCpu<AFloat>::CalculateConvActivationGradients(TCpuTensor<AFloat> &activationGradientsBackward,
                                                    const TCpuTensor<AFloat> &df,
                                                    const TCpuMatrix<AFloat> &weights, size_t batchSize,
                                                    size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                    size_t width, size_t filterDepth, size_t filterHeight,
                                                    size_t filterWidth)
{
   if (activationGradientsBackward.GetSize() == 0) return;

  
   activationGradientsBackward.Zero();
   

   // Transform the weights

   //TMVA_DNN_PrintTCpuMatrix(weights,"weights");
   // filter depth must be same as input depth
   TCpuMatrix<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());
   //TMVA_DNN_PrintTCpuMatrix(rotWeights,"rot-weights");

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
    Im2colIndices(vIndices, df.At(0).GetMatrix(), tempNLocalViews, height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);


    //for (size_t i = 0; i < batchSize; i++) {
    R__ASSERT(batchSize == df.GetFirstSize() );
    R__ASSERT(batchSize == activationGradientsBackward.GetFirstSize() );
    auto f = [&] (UInt_t i)
   {
   
       // Im2col(dfTr, df[i], height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
       //       tempZeroPaddingHeight, tempZeroPaddingWidth);

      TCpuMatrix<AFloat> dfTr(tempNLocalViews, tempNLocalViewPixels);
      
      Im2colFast(dfTr, df.At(i).GetMatrix(), vIndices); 

       //TMVA_DNN_PrintTCpuMatrix(df[i],"df[i]");
       //TMVA_DNN_PrintTCpuMatrix(dfTr,"dfTr");

      Matrix_t agb_m = activationGradientsBackward.At(i).GetMatrix();
      MultiplyTranspose(agb_m, rotWeights, dfTr);

       //TMVA_DNN_PrintTCpuMatrix(activationGradientsBackward[i],"activGrad-result");

   };

    TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI( batchSize ) );
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvWeightGradients(TCpuMatrix<AFloat> &weightGradients,
                                                const TCpuTensor<AFloat> &df,
                                                const TCpuTensor<AFloat> &activationsBackward,
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
   Im2colIndices(vIndices, activationsBackward.At(0).GetMatrix(), nLocalViews, inputHeight, inputWidth, filterHeight , filterWidth,
             tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
   
   //std::cout << "do back-propagation in conv layer - compute weight gradient" << std::endl;

   // std::vector< TCpuMatrix<AFloat> > vres;//(batchSize); 
   // for (size_t i = 0; i < batchSize; i++) {
   //    vres.emplace_back(depth, nLocalViewPixels);
   //    //TMVA_DNN_PrintTCpuMatrix(df[i],"df");
   //    //TMVA_DNN_PrintTCpuMatrix(activationsBackward[i],"df");
      
   //}
   //TCpuTensor<AFloat> vres( { batchSize, depth, nLocalViewPIxels} );
   TCpuTensor<AFloat> vres( batchSize, depth, nLocalViewPixels);

   auto fmap = [&](int i) { 
 
      //TMVA_DNN_PrintTCpuMatrix(df[i],"df-i");
      TCpuMatrix<AFloat> xTr(nLocalViews, nLocalViewPixels);
      TCpuMatrix<AFloat> res(depth, nLocalViewPixels);

      //computing t he gradient is equivalent of doing a convolution of the input using as conv kernel the delta's (the df[] values) 
      //N.B. only stride values=1 are now supported
 
      //xTr.Zero(); 
      // Im2col(xTr, const_cast<TCpuMatrix<AFloat> &>(activationsBackward[i]), inputHeight, inputWidth, filterHeight , filterWidth,
      //        tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);
      Im2colFast(xTr, activationsBackward.At(i).GetMatrix(), vIndices);

      //std::cout << "doing im2colfast" << std::endl;
      //TMVA_DNN_PrintTCpuMatrix(xTr,"xTr-i");
      //TMVA_DNN_PrintTCpuMatrix(activationsBackward[i],"actbackward-i");
      Matrix_t mres = vres.At(i).GetMatrix();
      Multiply( mres, df.At(i).GetMatrix(), xTr);
      //TMVA_DNN_PrintTCpuMatrix(vres[i],"res_ofMT");

      return;
      //return res;
   };

   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(fmap, ROOT::TSeqI( batchSize ) );

//   auto freduce = [&](const TCpuTensor<AFloat> & vres) { 
      R__ASSERT(vres.GetFirstSize() == batchSize); 
      for (size_t i = 0; i < batchSize; i++) {
         //TMVA_DNN_PrintTCpuMatrix(vres[i],"res");
         Matrix_t vres_m = vres.At(i).GetMatrix(); 
         for (size_t j = 0; j < depth; j++) {
            for (size_t k = 0; k < filterDepth; k++) {
               size_t kOffset = k * filterSize; 
               for (size_t l = 0; l < filterSize; l++) {
                  //weightGradients(j, k * (filterHeight * filterWidth) + l) += res(k, (tempNLocalViews - 1) - l);
                  weightGradients(j, kOffset + l) += vres_m(j,  kOffset + l);
               }
            }
         }
         // TMVA_DNN_PrintTCpuMatrix(weightGradients,"weights_i");
      }
      //  };
  
   //TCpuMatrix<AFloat>::GetThreadExecutor().MapReduce(fmap, ROOT::TSeqI( batchSize ) , freduce);
   //TMVA_DNN_PrintTCpuMatrix(weightGradients,"W-Grad");
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CalculateConvBiasGradients(TCpuMatrix<AFloat> &biasGradients, const TCpuTensor<AFloat> &df,
                                              size_t batchSize, size_t depth, size_t nLocalViews)
{
   biasGradients.Zero();
   for (size_t i = 0; i < depth; i++) {
      AFloat sum = 0;
      for (size_t j = 0; j < nLocalViews; j++) {
         for (size_t k = 0; k < batchSize; k++) {
            sum += df(k,i,j);
            //sum += df[k](i, j);
         }
      }
      biasGradients(i, 0) = sum;
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Downsample(TCpuTensor<AFloat> &tA, TCpuTensor<AFloat> &tB, const TCpuTensor<AFloat> &tC,
                              const PoolingDescriptors_t & /*descriptors*/,
                              PoolingWorkspace_t & /*workspace*/,
                              size_t imgHeight, size_t imgWidth, size_t fltHeight, size_t fltWidth, size_t strideRows,
                              size_t strideCols)
{
   // A is output , B is a cached index tensor used for backward pass and C is the input

   assert( tA.GetFirstSize() == tC.GetFirstSize());
   for (size_t ifirst = 0; ifirst < tC.GetFirstSize(); ++ifirst) {

      Matrix_t A = tA.At(ifirst).GetMatrix();
      Matrix_t B = tB.At(ifirst).GetMatrix();
      Matrix_t C = tC.At(ifirst).GetMatrix();

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
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::MaxPoolLayerBackward(TCpuTensor<AFloat> &activationGradientsBackward,
                                        const TCpuTensor<AFloat> &activationGradients,
                                        const TCpuTensor<AFloat> &indexMatrix,
                                        const TCpuTensor<AFloat> & /*inputActivation*/,
                                        const TCpuTensor<AFloat> & /*outputTensor*/,
                                        const PoolingDescriptors_t & /*descriptors*/,
                                        PoolingWorkspace_t & /*workspace*/,
                                        size_t /* imgHeight */,
                                        size_t /* imgWidth */,
                                        size_t /* fltHeight */,
                                        size_t /* fltWidth */,
                                        size_t /* strideRows */,
                                        size_t /* strideCols */,
                                        size_t nLocalViews)
{

   assert( activationGradientsBackward.GetFirstSize() == activationGradients.GetFirstSize());
   for (size_t l = 0; l < activationGradients.GetFirstSize(); ++l) {

      Matrix_t activationGradientsBackward_m = activationGradientsBackward.At(l).GetMatrix(); 
      Matrix_t activationGradients_m = activationGradients.At(l).GetMatrix(); 
      Matrix_t indexMatrix_m = indexMatrix.At(l).GetMatrix(); 

      size_t depth = activationGradientsBackward_m.GetNrows();

      for (size_t j = 0; j < depth; j++) {
         // initialize to zeros
         for (size_t t = 0; t < (size_t)activationGradientsBackward_m.GetNcols(); t++) {
            activationGradientsBackward_m(j, t) = 0;
         }

         // set values
         for (size_t k = 0; k < nLocalViews; k++) {
            AFloat grad = activationGradients_m(j, k);
            size_t winningIdx = indexMatrix_m(j, k);
            activationGradientsBackward_m(j, winningIdx) += grad;
         }
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::BatchNormLayerForwardTraining(Matrix_t input,
                                                 Matrix_t & gamma,
                                                 Matrix_t & beta,
                                                 Matrix_t outputActivation,
                                                 Matrix_t & Xmu,
                                                 Matrix_t & output,
                                                 Matrix_t & Variance,
                                                 Matrix_t & IVariance,
                                                 # if 0
                                                 const BNormDescriptors_t & /*descriptors*/,
                                                 BNormWorkspace_t & /*workspace*/,
                                                 # endif
                                                 std::vector<Scalar_t> & RunningMeans,
                                                 std::vector<Scalar_t> & RunningVars,
                                                 Scalar_t nTrainedBatches,
                                                 Scalar_t momentum,
                                                 Scalar_t epsilon) 
{
   int n = input.GetNrows();
   int d = input.GetNcols();

   for (int k = 0; k < d; ++k) {

      double mean = 0;
      for (int i = 0; i < n; i++) {
         mean = mean + input(i, k);
      }
      mean = mean / n;

      for (int i = 0; i < n; i++) {
         Xmu(i, k) = input(i, k) - mean;
      }
      double sq = 0;
      for (int i = 0; i < n; i++) {
         sq = sq + (Xmu(i, k) * Xmu(i, k));
      }
      Variance(0, k) = sq / n;
      // fVar(0,k) = fVar(0,k) + epsilon;
      // sqrtvar(0,k) =
      IVariance(0, k) = 1. / std::sqrt(Variance(0, k) + epsilon);
      for (int i = 0; i < n; i++) {
         output(i, k) = Xmu(i, k) * IVariance(0, k);
         outputActivation(i, k) = gamma(0, k) * output(i, k) + beta(0, k);
      }

      // fVar(0,k) -= epsilon;

      if (nTrainedBatches == 0) {
         RunningMeans[k] = mean;
         RunningVars[k] = Variance(0, k) * (n) / (Scalar_t(n - 1) + epsilon);
      } else {
         Scalar_t decay = momentum; 
         if (momentum < 0) decay = nTrainedBatches/Scalar_t(nTrainedBatches+1);
         RunningMeans[k] = decay * RunningMeans[k] + (1. - decay) * mean;
         RunningVars[k] = decay * RunningVars[k] + (1.-decay) * Variance(0, k) * (n) / (Scalar_t(n - 1) + epsilon);
      }

   } // end loop on k
   nTrainedBatches++;

   // fVar.Print();
   // std::cout << " training batch " << fTrainedBatches << " mu var0" << fMu_Training[0] << std::endl;
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::BatchNormLayerForwardInference(Matrix_t input,
                                                  Matrix_t & gamma,
                                                  Matrix_t & beta,
                                                  Matrix_t outputActivation,
                                                  # if 0
                                                  const BNormDescriptors_t & /*descriptors*/,
                                                  BNormWorkspace_t & /*workspace*/,
                                                  # endif
                                                  std::vector<Scalar_t> & RunningMeans,
                                                  std::vector<Scalar_t> & RunningVars,
                                                  Scalar_t nTrainedBatches,
                                                  Scalar_t epsilon) 
{
   int n = input.GetNrows();
   int d = input.GetNcols();

   for (int k = 0; k < d; ++k) {
      // during inference just use stored mu and variance
      for (int i = 0; i < n; i++) {
         outputActivation(i, k) =
            gamma(0, k) * ((input(i, k) - RunningMeans[k]) / (sqrt(RunningVars[k] + epsilon))) + beta(0, k);
      }
   }

   nTrainedBatches = 0;

   // std::cout << " testing batch  " << fTrainedBatches << " mu var0" << fMu_Training[0] << std::endl;
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::BatchNormLayerForwardBackward(const Matrix_t & outputGrad,
                                                 const Matrix_t & gamma,
                                                 Matrix_t &dgamma,
                                                 Matrix_t &dbeta,
                                                 Matrix_t dx,
                                                 Matrix_t & output,
                                                 Matrix_t & Xmu,
                                                 Matrix_t & IVariance,
                                                 Matrix_t & Variance,
                                                 # if 0
                                                 const BNormDescriptors_t & /*descriptors*/,
                                                 BNormWorkspace_t & /*workspace*/,
                                                 # endif
                                                 Scalar_t epsilon) 
{
   //const Matrix_t &x = activations_backward[0];
   int d = outputGrad.GetNcols();
   int n = outputGrad.GetNrows();

   // compute first gradients for gamma and beta
   for (int k = 0; k < d; k++) {
      dgamma(0, k) = 0;
      dbeta(0, k) = 0;
      for (int i = 0; i < n; i++) {
         dbeta(0, k) += outputGrad(i, k);
         dgamma(0, k) += outputGrad(i, k) * output(i, k);
         // dxhat(i,k) = dout(i,k) * gamma(0,k);
      }
   }

   // compute gradients with respect to input
   double npSumDy = 0;
   double npSumDyHMu = 0;

   for (int k = 0; k < d; k++) {
      for (int i = 0; i < n; i++) {
         npSumDy += outputGrad(i, k);
         npSumDyHMu += outputGrad(i, k) * Xmu(i, k);
      }
      for (int i = 0; i < n; i++) {
         dx(i, k) = (1. / double(n) * gamma(0, k) * IVariance(0, k)) *
                    (n * outputGrad(i, k) - npSumDy - Xmu(i, k) / (Variance(0, k) + epsilon) * npSumDyHMu);
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
         A(i, j) = B(nElem / nColsB, nElem % nColsB);
      }
   }
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Flatten(TCpuTensor<AFloat> &A, const TCpuTensor<AFloat> &B )
{

   //printf ( "input tensor %f \n",B(0,0,0));
   
   assert( B.GetShape().size() == 3  );
   assert( A.GetShape().size() == 3  );


   size_t bsize = B.GetFirstSize(); 
   size_t nRows = B.GetHSize();
   size_t nCols = B.GetWSize();

   assert (  A.GetFirstSize() == 1);
   assert (  A.GetHSize() == bsize);
   assert (  A.GetWSize() == nRows*nCols);

   for (size_t i = 0; i < bsize; i++) {
      for (size_t j = 0; j < nRows; j++) {
         for (size_t k = 0; k < nCols; k++) {
            A( 0, i, j * nCols + k) = B(i, j, k);
         }  
      }
   }

   // size_t bsize = B.GetFirstSize();
   // size_t n = B.GetSize()/bsize; 
   // if (B.GetLayout() == TCpuTensor<AFloat>::MemoryLayout::ColumnMajor ) { 
       
   // }
   // A = B.Reshape(bsize, n)
}

//____________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Deflatten(TCpuTensor<AFloat> &A, const TCpuTensor<AFloat> &B )
{

   assert( B.GetShape().size() == 3  );
   assert( A.GetShape().size() == 3  );

   size_t size = A.GetFirstSize(); 
   size_t nRows = A.GetHSize();
   size_t nCols = A.GetWSize();

   assert (  B.GetFirstSize() == 1);
   assert (  B.GetHSize() == size);
   assert (  B.GetWSize() == nRows*nCols);
   for (size_t i = 0; i < (size_t)size; i++) {
      for (size_t j = 0; j < (size_t)nRows; j++) {
         for (size_t k = 0; k < (size_t)nCols; k++) {
               A(i, j, k) = B(0, i, j * nCols + k);
         }
      }
   }
}

//______________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::Rearrange(Tensor_t &out, const Tensor_t &in)
{
   // B x T x D out --- T x B x D in*/
   assert ( out.GetShape().size() == 3 && in.GetShape().size() == 3);
  

   size_t B = out.GetFirstSize(); 
   size_t T = out.GetHSize();  //1 for row-major
   size_t D = out.GetWSize();  // 2 for row-major
   if ((T != in.GetFirstSize()) || (B != in.GetHSize()) || (D != in.GetWSize()) ) {
      std::cout << "Incompatible Dimensions\n"
                << in.GetFirstSize() << "x" << in.GetHSize() << "x" << in.GetWSize() << " --> " << B << "x" << T << "x"
                << D << "\n";
      assert(false);
      return;
   }
   for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < T; ++j) {
         for (size_t k = 0; k < D; ++k) {
            out( i, j, k ) = in( j, i, k);
         }
      }
   }
   return;
}

} // namespace DNN
} // namespace TMVA
