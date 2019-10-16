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

namespace TMVA
{
namespace DNN
{

template<typename AFloat>
auto TCpu<AFloat>::RecurrentLayerBackward(TCpuMatrix<AFloat> & state_gradients_backward, // BxH
                                          TCpuMatrix<AFloat> & input_weight_gradients,
                                          TCpuMatrix<AFloat> & state_weight_gradients,
                                          TCpuMatrix<AFloat> & bias_gradients,
                                          TCpuMatrix<AFloat> & df, //BxH
                                          const TCpuMatrix<AFloat> & state, // BxH
                                          const TCpuMatrix<AFloat> & weights_input, // HxD
                                          const TCpuMatrix<AFloat> & weights_state, // HxH
                                          const TCpuMatrix<AFloat> & input,  // BxD
                                          TCpuMatrix<AFloat> & input_gradient)
-> TCpuMatrix<AFloat> &
{

   // std::cout << "Recurrent Propo" << std::endl;
   // TMVA_DNN_PrintTCpuMatrix(df,"DF");
   // TMVA_DNN_PrintTCpuMatrix(state_gradients_backward,"State grad");
   // TMVA_DNN_PrintTCpuMatrix(input_weight_gradients,"input w grad");
   // TMVA_DNN_PrintTCpuMatrix(state,"state");
   // TMVA_DNN_PrintTCpuMatrix(input,"input");

   // Compute element-wise product.
   //Hadamard(df, state_gradients_backward);  // B x H

   // Input gradients.
   if (input_gradient.GetNoElements() > 0) Multiply(input_gradient, df, weights_input);

   // State gradients.
   if (state_gradients_backward.GetNoElements() > 0) Multiply(state_gradients_backward, df, weights_state);

   // compute the gradients
   // Perform the operation in place by readding the result on the same gradient matrix
   // e.g. W += D * X

   // Weights gradients
   if (input_weight_gradients.GetNoElements() > 0) {
      TransposeMultiply(input_weight_gradients, df, input, 1. , 1.); // H x B . B x D
   }
   if (state_weight_gradients.GetNoElements() > 0) {
      TransposeMultiply(state_weight_gradients, df, state, 1. , 1. ); // H x B . B x H
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      SumColumns(bias_gradients, df, 1., 1.);  // could be probably do all here
   }

   //std::cout << "RecurrentPropo: end " << std::endl;

   // TMVA_DNN_PrintTCpuMatrix(state_gradients_backward,"State grad");
   // TMVA_DNN_PrintTCpuMatrix(input_weight_gradients,"input w grad");
   // TMVA_DNN_PrintTCpuMatrix(bias_gradients,"bias grad");
   // TMVA_DNN_PrintTCpuMatrix(input_gradient,"input grad");

   return input_gradient;
}

} // namespace DNN
} // namespace TMVA
