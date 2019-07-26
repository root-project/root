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
                                             const TMatrixT<Scalar_t> & precCellActivations,
                                             const TMatrixT<Scalar_t> & fInput,
                                             const TMatrixT<Scalar_t> & fForget,
                                             const TMatrixT<Scalar_t> & fCandidate,
                                             const TMatrixT<Scalar_t> & fOutput,
                                             const TMatrixT<Scalar_t> & weights_input,
                                             const TMatrixT<Scalar_t> & weights_forget,
                                             const TMatrixT<Scalar_t> & weights_candidate,
                                             const TMatrixT<Scalar_t> & weights_output,
                                             const TMatrixT<Scalar_t> & weights_input_state,
                                             const TMatrixT<Scalar_t> & weights_forget_state,
                                             const TMatrixT<Scalar_t> & weights_candidate_state,
                                             const TMatrixT<Scalar_t> & weights_output_state,
                                             const TMatrixT<Scalar_t> & input,
                                             TMatrixT<Scalar_t> & input_gradient,
                                             TMatrixT<Scalar_t> & cell_gradient,
                                             TMatrixT<Scalar_t> & cell_tanh)
-> Matrix_t & 
{
   /* TODO: Update all gate values during backward pass using required equations.
    * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 */

   Hadamard(cell_gradient, fOutput); 
   Hadamard(cell_gradient, state_gradients_backward);
   cell_gradient += cell_gradients_backward;
   cell_gradients_backward = cell_gradient;
   Hadamard(cell_gradients_backward, fForget);

   TMatrixT<Scalar_t> candidate_gradient(cell_gradient);
   Hadamard(candidate_gradient, fInput);
   Hadamard(candidate_gradient, dc);
  
   TMatrixT<Scalar_t> input_gate_gradient(cell_gradient);
   Hadamard(input_gate_gradient, fCandidate);
   Hadamard(input_gate_gradient, di);

   TMatrixT<Scalar_t> forget_gradient(cell_gradient);
   Hadamard(forget_gradient, precCellActivations);
   Hadamard(forget_gradient, df);
  
   TMatrixT<Scalar_t> output_gradient(cell_tanh);
   Hadamard(output_gradient, state_gradients_backward);
   Hadamard(output_gradient, dout);

   TMatrixT<Scalar_t> tmpInp(input_gradient);
   tmpInp.Mult(input_gate_gradient, weights_input);
   input_gradient = tmpInp;
   tmpInp.Mult(forget_gradient, weights_forget);
   input_gradient += tmpInp;
   tmpInp.Mult(candidate_gradient, weights_candidate);
   input_gradient += tmpInp;
   tmpInp.Mult(output_gradient, weights_output);
   input_gradient += tmpInp;

   TMatrixT<Scalar_t> tmpState(state_gradients_backward);
   tmpState.Mult(input_gate_gradient, weights_input_state);
   state_gradients_backward = tmpState;
   tmpState.Mult(forget_gradient, weights_forget_state);
   state_gradients_backward += tmpState;
   tmpState.Mult(candidate_gradient, weights_candidate_state);
   state_gradients_backward += tmpState;
   tmpState.Mult(output_gradient, weights_output_state);
   state_gradients_backward += tmpState;

   TMatrixT<Scalar_t> tmp(input_weight_gradients);
   input_weight_gradients.TMult(input_gate_gradient, input);             
   input_weight_gradients += tmp;
   tmp = forget_weight_gradients;
   forget_weight_gradients.TMult(forget_gradient, input);
   forget_weight_gradients += tmp;
   tmp = candidate_weight_gradients;
   candidate_weight_gradients.TMult(candidate_gradient, input);
   candidate_weight_gradients += tmp;
   tmp = output_weight_gradients;
   output_weight_gradients.TMult(output_gradient, input);
   output_weight_gradients += tmp;

   TMatrixT<Scalar_t> tmp1(input_state_weight_gradients);
   input_state_weight_gradients.TMult(input_gate_gradient, precStateActivations);        
   input_state_weight_gradients += tmp1;   
   tmp1 = forget_state_weight_gradients;
   forget_state_weight_gradients.TMult(forget_gradient, precStateActivations);
   forget_state_weight_gradients += tmp1;
   tmp1 = candidate_state_weight_gradients;
   candidate_state_weight_gradients.TMult(candidate_gradient, precStateActivations);
   candidate_state_weight_gradients += tmp1;
   tmp1 = output_state_weight_gradients;
   output_state_weight_gradients.TMult(output_gradient, precStateActivations);
   output_state_weight_gradients += tmp1;

   for (size_t j = 0; j < (size_t) df.GetNcols(); j++) {
      Scalar_t sum_inp = 0.0, sum_forget = 0.0, sum_candidate = 0.0, sum_out = 0.0;
      // this loops on batch size summing all gradient contributions in a batch
      for (size_t i = 0; i < (size_t) df.GetNrows(); i++) {
         sum_inp += input_gate_gradient(i,j);
         sum_forget += forget_gradient(i,j);
         sum_candidate += candidate_gradient(i,j);
         sum_out += output_gradient(i,j);
      }
      input_bias_gradients(j,0) += sum_inp;
      forget_bias_gradients(j,0) += sum_forget;
      candidate_bias_gradients(j,0) += sum_candidate;
      output_bias_gradients(j,0) += sum_out;
   }
   
   return input_gradient;
}



//______________________________________________________________________________
template <typename Scalar_t>
auto TReference<Scalar_t>::GRULayerBackward(TMatrixT<Scalar_t> & state_gradients_backward,
                                     TMatrixT<Scalar_t> & reset_weight_gradients,
                                     TMatrixT<Scalar_t> & update_weight_gradients,
                                     TMatrixT<Scalar_t> & candidate_weight_gradients,
                                     TMatrixT<Scalar_t> & reset_state_weight_gradients,
                                     TMatrixT<Scalar_t> & update_state_weight_gradients,
                                     TMatrixT<Scalar_t> & candidate_state_weight_gradients,
                                     TMatrixT<Scalar_t> & reset_bias_gradients,
                                     TMatrixT<Scalar_t> & update_bias_gradients,
                                     TMatrixT<Scalar_t> & candidate_bias_gradients,
                                     TMatrixT<Scalar_t> & dr,
                                     TMatrixT<Scalar_t> & du,
                                     TMatrixT<Scalar_t> & dc,
                                     const TMatrixT<Scalar_t> & precStateActivations,
                                     const TMatrixT<Scalar_t> & fReset,
                                     const TMatrixT<Scalar_t> & fUpdate,
                                     const TMatrixT<Scalar_t> & fCandidate,
                                     const TMatrixT<Scalar_t> & weights_reset,
                                     const TMatrixT<Scalar_t> & weights_update,
                                     const TMatrixT<Scalar_t> & weights_candidate,
                                     const TMatrixT<Scalar_t> & weights_reset_state,
                                     const TMatrixT<Scalar_t> & weights_update_state,
                                     const TMatrixT<Scalar_t> & weights_candidate_state,
                                     const TMatrixT<Scalar_t> & input,
                                     TMatrixT<Scalar_t> & input_gradient)
-> Matrix_t &
{

   //to do multiply by delh_t
   TMatrixT<Scalar_t> reset_gradient(fUpdate);
   TMatrixT<Scalar_t> tmpMul(precStateActivations);
   tmpMul.Mult(dr, weights_candidate_state);
   Hadamard(reset_gradient, tmpMul);
   Hadamard(reset_gradient, dc);
   Hadamard(reset_gradient, precStateActivations);

   TMatrixT<Scalar_t> update_gradient(precStateActivations); // H X 1
   for (size_t j = 0; j < (size_t) tmp.GetNcols(); j++) {
      for (size_t i = 0; i < (size_t) tmp.GetNrows(); i++) {
         tmp(i,j) = fCandidate(i,j) - tmp(i,j);
      }
   }
   Hadamard(update_gradient, du);

   TMatrixT<Scalar_t> candidate_gradient(fUpdate);
   Hadamard(candidate_gradient, dc);
   
   

   return input_gradient;
}





} // namespace DNN
} // namespace TMVA
