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

} // namespace DNN
} // namespace TMVA
