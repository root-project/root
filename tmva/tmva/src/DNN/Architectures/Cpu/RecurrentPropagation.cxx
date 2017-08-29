// @(#)root/tmva/tmva/dnn:$Id$ 
// Author: Saurav Shekhar 23/06/17

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Implementation of the functions required for the forward and    //
// backward propagation of activations through a recurrent neural  //
// network in the TCpu architecture                                //
/////////////////////////////////////////////////////////////////////


#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/DNN/Architectures/Cpu/Blas.h"

namespace TMVA
{
namespace DNN
{
  
template<typename AFloat>
auto TCpu<AFloat>::RecurrentLayerBackward(TCpuMatrix<AFloat> & state_gradients_backward, // BxH
                                          TCpuMatrix<AFloat> & input_weight_gradients,
                                          TCpuMatrix<AFloat> & state_weight_gradients,
                                          TCpuMatrix<AFloat> & bias_gradients,
                                          TCpuMatrix<AFloat> & df, //DxH
                                          const TCpuMatrix<AFloat> & state, // BxH
                                          const TCpuMatrix<AFloat> & weights_input, // HxD 
                                          const TCpuMatrix<AFloat> & weights_state, // HxH
                                          const TCpuMatrix<AFloat> & input,  // BxD
                                          TCpuMatrix<AFloat> & input_gradient)
-> TCpuMatrix<AFloat> &
{
   // Compute element-wise product.
   Hadamard(df, state_gradients_backward);  // B x H 
   
   // Input gradients.
   if (input_gradient.GetNElements() > 0) Multiply(input_gradient, df, weights_input);

   // State gradients.
   if (state_gradients_backward.GetNElements() > 0) Multiply(state_gradients_backward, df, weights_state);

   // Weights gradients
   if (input_weight_gradients.GetNElements() > 0) {
      TCpuMatrix<AFloat> tmp(input_weight_gradients);
      TransposeMultiply(input_weight_gradients, df, input); // H x B . B x D
      ScaleAdd(input_weight_gradients, tmp, 1);
   }
   if (state_weight_gradients.GetNElements() > 0) {
      TCpuMatrix<AFloat> tmp(state_weight_gradients);
      TransposeMultiply(state_weight_gradients, df, state); // H x B . B x H
      ScaleAdd(state_weight_gradients, tmp, 1);
   }

   // Bias gradients.
   if (bias_gradients.GetNElements() > 0) SumColumns(bias_gradients, df);
   return input_gradient;
}

} // namespace DNN
} // namespace TMVA
