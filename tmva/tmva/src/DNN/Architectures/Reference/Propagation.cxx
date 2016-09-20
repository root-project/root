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

template<typename Scalar_t>
void TReference<Scalar_t>::MultiplyTranspose(TMatrixT<Scalar_t> &output,
                                            const TMatrixT<Scalar_t> &input,
                                            const TMatrixT<Scalar_t> &weights)
{
    output.MultT(input, weights);
}

template<typename Scalar_t>
void TReference<Scalar_t>::AddRowWise(TMatrixT<Scalar_t> &output,
                                     const TMatrixT<Scalar_t> &biases)
{
   for (size_t i = 0; i < (size_t) output.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) output.GetNcols(); j++) {
         output(i,j) += biases(j,0);
      }
   }
}

template<typename Scalar_t>
void TReference<Scalar_t>::Backward(TMatrixT<Scalar_t> & activation_gradients_backward,
                                   TMatrixT<Scalar_t> & weight_gradients,
                                   TMatrixT<Scalar_t> & bias_gradients,
                                   TMatrixT<Scalar_t> & df,
                                   const TMatrixT<Scalar_t> & activation_gradients,
                                   const TMatrixT<Scalar_t> & weights,
                                   const TMatrixT<Scalar_t> & activations_backward)
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
         Scalar_t sum = 0.0;
         for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
            sum += df(i,j);
         }
         bias_gradients(j,0) = sum;
      }
   }
}

template<typename Scalar_t>
void TReference<Scalar_t>::ScaleAdd(TMatrixT<Scalar_t> & A,
                                   const TMatrixT<Scalar_t> & B,
                                   Scalar_t beta)
{
   for (size_t i = 0; i < (size_t) A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) A.GetNcols(); j++) {
         A(i,j) += beta * B(i,j);
      }
   }
}

template<typename Scalar_t>
void TReference<Scalar_t>::Copy(TMatrixT<Scalar_t> & A,
                                const TMatrixT<Scalar_t> & B)
{
   A = B;
}

} // namespace DNN
} // namespace TMVA
