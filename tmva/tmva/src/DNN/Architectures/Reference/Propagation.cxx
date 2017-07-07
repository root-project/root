// @(#)root/tmva/tmva/dnn:$Id$ // Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Implementation of the functions required for the forward and    //
// backward propagation of activations through a neural network in //
// the reference implementation.                                   //
/////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

template <typename AReal>
void TReference<AReal>::MultiplyTranspose(TMatrixT<AReal> &output, const TMatrixT<AReal> &input,
                                          const TMatrixT<AReal> &weights)
{
    output.MultT(input, weights);
}

template <typename AReal>
void TReference<AReal>::AddRowWise(TMatrixT<AReal> &output, const TMatrixT<AReal> &biases)
{
   for (size_t i = 0; i < (size_t) output.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) output.GetNcols(); j++) {
         output(i,j) += biases(j,0);
      }
   }
}

template <typename AReal>
void TReference<AReal>::Backward(TMatrixT<AReal> &activation_gradients_backward, TMatrixT<AReal> &weight_gradients,
                                 TMatrixT<AReal> &bias_gradients, TMatrixT<AReal> &df,
                                 const TMatrixT<AReal> &activation_gradients, const TMatrixT<AReal> &weights,
                                 const TMatrixT<AReal> &activations_backward)
{

   // Compute element-wise product.
   for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
         df(i,j) *= activation_gradients(i,j);
      }
   }

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0) {
       activation_gradients_backward.Mult(df, weights);
   }

   // Weights gradients.
   if (weight_gradients.GetNoElements() > 0) {
      weight_gradients.TMult(df, activations_backward);
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
         AReal sum = 0.0;
         for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
            sum += df(i,j);
         }
         bias_gradients(j,0) = sum;
      }
   }
}

template <typename AReal>
void TReference<AReal>::ScaleAdd(TMatrixT<AReal> &A, const TMatrixT<AReal> &B, AReal beta)
{
   for (size_t i = 0; i < (size_t) A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) A.GetNcols(); j++) {
         A(i,j) += beta * B(i,j);
      }
   }
}

template <typename AReal>
void TReference<AReal>::Copy(TMatrixT<AReal> &A, const TMatrixT<AReal> &B)
{
   A = B;
}

template <typename AReal>
void TReference<AReal>::SumColumns(TMatrixT<AReal> &B, const TMatrixT<AReal> &A)
{
   B = 0.0;
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         B(0, j) += A(i, j);
      }
   }
}

} // namespace DNN
} // namespace TMVA
