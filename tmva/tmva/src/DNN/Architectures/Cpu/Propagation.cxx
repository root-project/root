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

template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::MultiplyTranspose(TCpuMatrix<Real_t> &output,
                                                  const TCpuMatrix<Real_t> &input,
                                                  const TCpuMatrix<Real_t> &Weights)
{
    int m = (int) input.GetNrows();
    int k = (int) input.GetNcols();
    int n = (int) Weights.GetNrows();

    char transa = 'N';
    char transb = 'T';

    Real_t alpha = 1.0;
    Real_t beta  = 0.0;

    const Real_t *A = input.GetRawDataPointer();
    const Real_t *B = Weights.GetRawDataPointer();
          Real_t *C = output.GetRawDataPointer();

    ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha,
                            A, &m, B, &n, &beta, C, &m);
}

template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::AddRowWise(
    TCpuMatrix<Real_t> &output,
    const TCpuMatrix<Real_t> &biases)
{
    int m = (int) output.GetNrows();
    int n = (int) output.GetNcols();

    int inc = 1.0;
    Real_t alpha = 1.0;

          Real_t * A = output.GetRawDataPointer();
    const Real_t * x = TCpuMatrix<Real_t>::GetOnePointer();
    const Real_t * y = biases.GetRawDataPointer();

    ::TMVA::DNN::Blas::Ger(&m, &n, &alpha, x, &inc, y, &inc, A, &m);
}

template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::Backward(
    TCpuMatrix<Real_t> & activationGradientsBackward,
    TCpuMatrix<Real_t> & weightGradients,
    TCpuMatrix<Real_t> & biasGradients,
    TCpuMatrix<Real_t> & df,
    const TCpuMatrix<Real_t> & activationGradients,
    const TCpuMatrix<Real_t> & weights,
    const TCpuMatrix<Real_t> & activationsBackward)
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
