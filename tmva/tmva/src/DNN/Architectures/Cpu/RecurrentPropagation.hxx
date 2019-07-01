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
   if (input_gradient.GetNoElements() > 0) {
      Multiply(input_gradient, df, weights_input);
   }

   // State gradients.
   if (state_gradients_backward.GetNoElements() > 0) {
      Multiply(state_gradients_backward, df, weights_state);
   }

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

//______________________________________________________________________________
template <typename AFloat>
auto inline TCpu<AFloat>::LSTMLayerBackward(TCpuMatrix<AFloat> & state_gradients_backward,
												 TCpuMatrix<AFloat> & cell_gradients_backward,
												 TCpuMatrix<AFloat> & input_weight_gradients,
												 TCpuMatrix<AFloat> & forget_weight_gradients,
												 TCpuMatrix<AFloat> & candidate_weight_gradients,
												 TCpuMatrix<AFloat> & output_weight_gradients,
												 TCpuMatrix<AFloat> & input_state_weight_gradients,
												 TCpuMatrix<AFloat> & forget_state_weight_gradients,
												 TCpuMatrix<AFloat> & candidate_state_weight_gradients,
												 TCpuMatrix<AFloat> & output_state_weight_gradients,
												 TCpuMatrix<AFloat> & input_bias_gradients,
												 TCpuMatrix<AFloat> & forget_bias_gradients,
												 TCpuMatrix<AFloat> & candidate_bias_gradients,
												 TCpuMatrix<AFloat> & output_bias_gradients,
												 TCpuMatrix<AFloat> & di,
												 TCpuMatrix<AFloat> & df,
												 TCpuMatrix<AFloat> & dc,
												 TCpuMatrix<AFloat> & dout,
												 const TCpuMatrix<AFloat> & precStateActivations,
												 const TCpuMatrix<AFloat> & precCellActivations,
												 const TCpuMatrix<AFloat> & fInput,
												 const TCpuMatrix<AFloat> & fForget,
												 const TCpuMatrix<AFloat> & fCandidate,
												 const TCpuMatrix<AFloat> & fOutput,
												 const TCpuMatrix<AFloat> & weights_input,
												 const TCpuMatrix<AFloat> & weights_forget,
												 const TCpuMatrix<AFloat> & weights_candidate,
												 const TCpuMatrix<AFloat> & weights_output,
												 const TCpuMatrix<AFloat> & weights_input_state,
												 const TCpuMatrix<AFloat> & weights_forget_state,
												 const TCpuMatrix<AFloat> & weights_candidate_state,
												 const TCpuMatrix<AFloat> & weights_output_state,
												 const TCpuMatrix<AFloat> & input,
												 TCpuMatrix<AFloat> & input_gradient,
                                     TCpuMatrix<AFloat> & cell_gradient,
                                     TCpuMatrix<AFloat> & cell_tanh)
-> TCpuMatrix<AFloat> &
{
   /* TODO: Update all gate values during backward pass using required equations.
    * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 */

   //some temporary varibales used later
   TCpuMatrix<AFloat> cache(fCell.GetNrows(), fCell.GetNcols());
   TCpuMatrix<AFloat> tmpInp(input_gradient.GetNrows(), input_gradient.GetNcols());
   TCpuMatrix<AFloat> tmpState(state_gradients_backward.GetNrows(), state_gradients_backward.GetNcols());

   TCpuMatrix<AFloat> input_gate_gradient(fInput.GetNrows(), fInput.GetNcols());
   TCpuMatrix<AFloat> forget_gradient(fForget.GetNrows(), fForget.GetNcols());
   TCpuMatrix<AFloat> candidate_gradient(fCandidate.GetNrows(), fCandidate.GetNcols());
   TCpuMatrix<AFloat> output_gradient(fOutput.GetNrows(), fOutput.GetNcols());
   
   Hadamard(cell_gradient, fOutput); 
   Hadamard(cell_gradient, state_gradients_backward);
   ScaleAdd(cell_gradient, cell_gradients_backward);
   Copy(cell_gradient, cell_gradients_backward);
   Hadamard(cell_gradients_backward, fForget);

   Copy(cell_gradient, candidate_gradient);
   Hadamard(candidate_gradient, fInput);
   Hadamard(candidate_gradient, dc);

   Copy(cell_gradient, input_gate_gradient);
   Hadamard(input_gate_gradient, fCandidate);
   Hadamard(input_gate_gradient, di);

   Copy(cell_gradient, forget_gradient);
   Hadamard(forget_gradient, precCellActivations);
   Hadamard(forget_gradient, df);

   Copy(cell_gradient, output_gradient);
   Hadamard(output_gradient, state_gradients_backward);
   Hadamard(output_gradient, cell_tanh);
   Hadamard(output_gradient, dout);

   Multiply(tmpInp, input_gate_gradient, weights_input);
   Copy(tmpInp, input_gradient);
   Multiply(tmpInp, forget_gradient, weights_forget);
   ScaleAdd(input_gradient, tmpInp);
   Multiply(tmpInp, candidate_gradient, weights_candidate);
   ScaleAdd(input_gradient, tmpInp);
   Multiply(tmpInp, output_gradient, weights_output);
   ScaleAdd(input_gradient, tmpInp);

   Multiply(tmpState, input_gate_gradient, weights_input_state);
   Copy(tmpState, state_gradients_backward);
   Multiply(tmpState, forget_gradient, weights_forget_state);
   ScaleAdd(state_gradients_backward, tmpState);
   Multiply(tmpState, candidate_gradient, weights_candidate_state);
   ScaleAdd(state_gradients_backward, tmpState);
   Multiply(tmpState, output_gradient, weights_output_state);
   ScaleAdd(state_gradients_backward, tmpState);

   TransposeMultiply(input_weight_gradients, input_gate_gradient, input, 1. , 1.); // H x B . B x D
   TransposeMultiply(forget_weight_gradients, forget_gradient, input, 1. , 1.); 
   TransposeMultiply(candidate_weight_gradients, candidate_gradient, input, 1. , 1.); 
   TransposeMultiply(output_weight_gradients, output_gradient, input, 1. , 1.); 
   
   TransposeMultiply(input_state_weight_gradients, input_gate_gradient, precStateActivations, 1. , 1. ); // H x B . B x H
   TransposeMultiply(forget_state_weight_gradients, forget_gradient, precStateActivations, 1. , 1. );
   TransposeMultiply(candidate_state_weight_gradients, candidate_gradient, precStateActivations, 1. , 1. );
   TransposeMultiply(output_state_weight_gradients, output_gradient, precStateActivations, 1. , 1. );

   SumColumns(input_bias_gradients, input_gate_gradient, 1., 1.);  // could be probably do all here
   SumColumns(forget_bias_gradients, forget_gradient, 1., 1.);  
   SumColumns(candidate_bias_gradients, candidate_gradient, 1., 1.);  
   SumColumns(output_bias_gradients, output_gradient, 1., 1.);  
   
   return input_gradient;
}
} // namespace DNN
} // namespace TMVA
