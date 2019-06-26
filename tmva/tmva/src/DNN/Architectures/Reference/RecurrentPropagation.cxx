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
// network in the reference implementation.                        //
/////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA {
namespace DNN  {

  
//______________________________________________________________________________
template<typename Scalar_t>
auto TReference<Scalar_t>::RecurrentLayerBackward(TMatrixT<Scalar_t> & state_gradients_backward, // BxH
                                                  TMatrixT<Scalar_t> & input_weight_gradients,
                                                  TMatrixT<Scalar_t> & state_weight_gradients,
                                                  TMatrixT<Scalar_t> & bias_gradients,
                                                  TMatrixT<Scalar_t> & df, //BxH
                                                  const TMatrixT<Scalar_t> & state, // BxH
                                                  const TMatrixT<Scalar_t> & weights_input, // HxD 
                                                  const TMatrixT<Scalar_t> & weights_state, // HxH
                                                  const TMatrixT<Scalar_t> & input,  // BxD
                                                  TMatrixT<Scalar_t> & input_gradient)
-> Matrix_t &
{

   // std::cout << "Reference Recurrent Propo" << std::endl;
   // std::cout << "df\n";
   // df.Print();
   // std::cout << "state gradient\n";
   // state_gradients_backward.Print();
   // std::cout << "inputw gradient\n";
   // input_weight_gradients.Print(); 
   // std::cout << "state\n";
   // state.Print();
   // std::cout << "input\n";
   // input.Print();
   
   // Compute element-wise product.
   for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
         df(i,j) *= state_gradients_backward(i,j);      // B x H
      }
   }
   
   // Input gradients.
   if (input_gradient.GetNoElements() > 0) {
      input_gradient.Mult(df, weights_input);     // B x H . H x D = B x D
   }
   // State gradients
   if (state_gradients_backward.GetNoElements() > 0) {
      state_gradients_backward.Mult(df, weights_state);  // B x H . H x H = B x H
   }
   
   // Weights gradients.
   if (input_weight_gradients.GetNoElements() > 0) {
      TMatrixT<Scalar_t> tmp(input_weight_gradients);
      input_weight_gradients.TMult(df, input);             // H x B . B x D
      input_weight_gradients += tmp;
   }
   if (state_weight_gradients.GetNoElements() > 0) {
      TMatrixT<Scalar_t> tmp(state_weight_gradients);
      state_weight_gradients.TMult(df, state);             // H x B . B x H
      state_weight_gradients += tmp;
   }
   
   // Bias gradients. B x H -> H x 1
   if (bias_gradients.GetNoElements() > 0) {
      // this loops on state size
      for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
         Scalar_t sum = 0.0;
         // this loops on batch size summing all gradient contributions in a batch
         for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
            sum += df(i,j);
         }
         bias_gradients(j,0) += sum;
      }
   }

   // std::cout << "RecurrentPropo: end " << std::endl;

   // std::cout << "state gradient\n";
   // state_gradients_backward.Print();
   // std::cout << "inputw gradient\n";
   // input_weight_gradients.Print(); 
   // std::cout << "bias gradient\n";
   // bias_gradients.Print(); 
   // std::cout << "input gradient\n";
   // input_gradient.Print(); 

   
   return input_gradient;
}

 
//______________________________________________________________________________
template <typename Scalar_t>
auto TReference<Scalar_t>::LSTMLayerBackward(TMatrixT<Scalar_t> & state_gradients_backward,
                                             TMatrixT<Scalar_t> & cell_gradients_backward,
                                             TMatrixT<Scalar_t> & input_weight_gradients,
                                             TMatrixT<Scalar_t> & forget_weight_gradients,
                                             TMatrixT<Scalar_t> & candidate_weight_gradients,
                                             TMatrixT<Scalar_t> & output_weight_gradients,
                                             TMatrixT<Scalar_t> & input_state_weight_gradients,
                                             TMatrixT<Scalar_t> & forget_state_weight_gradients,
                                             TMatrixT<Scalar_t> & candidate_state_weight_gradients,
                                             TMatrixT<Scalar_t> & output_state_weight_gradients,
                                             TMatrixT<Scalar_t> & input_bias_gradients,
                                             TMatrixT<Scalar_t> & forget_bias_gradients,
                                             TMatrixT<Scalar_t> & candidate_bias_gradients,
                                             TMatrixT<Scalar_t> & output_bias_gradients,
                                             TMatrixT<Scalar_t> & di,
                                             TMatrixT<Scalar_t> & df,
                                             TMatrixT<Scalar_t> & dc,
                                             TMatrixT<Scalar_t> & dout,
                                             const TMatrixT<Scalar_t> & precStateActivations,
                                             //const TMatrixT<Scalar_t> & precCelleActivations,
                                             //const TMatrixT<Scalar_t> & fCell,
                                             //const TMatrixT<Scalar_t> & fInput,
                                             //const TMatrixT<Scalar_t> & fForget,
                                             //const TMatrixT<Scalar_t> & fCandidate,
                                             //const TMatrixT<Scalar_t> & fOutput,
                                             const TMatrixT<Scalar_t> & weights_input,
                                             const TMatrixT<Scalar_t> & weights_forget,
                                             const TMatrixT<Scalar_t> & weights_candidate,
                                             const TMatrixT<Scalar_t> & weights_output,
                                             const TMatrixT<Scalar_t> & weights_input_state,
                                             const TMatrixT<Scalar_t> & weights_forget_state,
                                             const TMatrixT<Scalar_t> & weights_candidate_state,
                                             const TMatrixT<Scalar_t> & weights_output_state,
                                             const TMatrixT<Scalar_t> & input,
                                             TMatrixT<Scalar_t> & input_gradient)
-> Matrix_t & 
{
   /* TODO: Update all gate values during backward pass using required equations.
    * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 */

}


} // namespace DNN
} // namespace TMVA
