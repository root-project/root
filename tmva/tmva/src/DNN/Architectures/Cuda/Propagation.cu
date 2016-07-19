// @(#)root/tmva/tmva/dnn:$Id$ // Author: Simon Pfreundschuh 13/07/16

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
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"

namespace TMVA {
namespace DNN  {

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::MultiplyTranspose(TCudaMatrix &output,
                            const TCudaMatrix &input,
                            const TCudaMatrix &Weights)
{
   int m, n, k;
   k = input.GetNcols();
   m = input.GetNrows();
   n = Weights.GetNrows();
   CudaDouble_t alpha = 1.0, beta = 0.0;

   tick();
   // Compute C = beta * C + alpha * (A * B^T)
   cublasDgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                           // beta
               output.GetDataPointer(), m);   // *C, ldc
   tock(fTimings.TimeMultiplyTranspose);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::AddRowWise(TCudaMatrix &Weights,
                      const TCudaMatrix &theta)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(Weights);
   cudaStream_t s = Weights.GetComputeStream();

   tick();
   add_row_wise<<<gridDims, blockDims, 0, s>>>(
       Weights.GetDataPointer(),
       theta.GetDataPointer(),
       Weights.GetNrows(),
       Weights.GetNcols());
   tock(fTimings.TimeAddRowWise);
}

//______________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Backward(TCudaMatrix & activation_gradients_backward,
                                  TCudaMatrix & weight_gradients,
                                  TCudaMatrix & bias_gradients,
                                  TCudaMatrix & df,
                                  const TCudaMatrix & activation_gradients,
                                  const TCudaMatrix & weights,
                                  const TCudaMatrix & activation_backward)
{
   // Compute element-wise product.
   Hadamard(df, activation_gradients);

   tick();

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0)
      Multiply(activation_gradients_backward, df, weights);

   // Weight gradients.
   if (weight_gradients.GetNoElements() > 0)
      TransposeMultiply(weight_gradients, df, activation_backward);

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0)
      SumColumns(bias_gradients, df);

   tock(fTimings.TimeBackward);
}

} // namespace DNN
} // namespace TMVA
