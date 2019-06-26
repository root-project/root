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
												 //const TCpuMatrix<AFloat> & precCelleActivations,
												 //const TCpuMatrix<AFloat> & fCell,
												 //const TCpuMatrix<AFloat> & fInput,
												 //const TCpuMatrix<AFloat> & fForget,
												 //const TCpuMatrix<AFloat> & fCandidate,
												 //const TCpuMatrix<AFloat> & fOutput,
												 const TCpuMatrix<AFloat> & weights_input,
												 const TCpuMatrix<AFloat> & weights_forget,
												 const TCpuMatrix<AFloat> & weights_candidate,
												 const TCpuMatrix<AFloat> & weights_output,
												 const TCpuMatrix<AFloat> & weights_input_state,
												 const TCpuMatrix<AFloat> & weights_forget_state,
												 const TCpuMatrix<AFloat> & weights_candidate_state,
												 const TCpuMatrix<AFloat> & weights_output_state,
												 const TCpuMatrix<AFloat> & input,
												 TCpuMatrix<AFloat> & input_gradient)
-> TCpuMatrix<AFloat> &
{
   /* TODO: Update all gate values during backward pass using required equations.
    * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 */

   //cell_gradient
   //candidate_grad (init 1)
   //inp
   //for
   //op

   /*
   DNN::evaluateDerivative<Architecture_t>(cell_gradient, DNN::EActivationFunction::kTanh, fCell);
   Hadamard(cell_gradient, fOuptput); 
   Hadamard(cell_gradient, state_gradients_backward);
   ScaleAdd(cell_gradient, cell_gradients_backward);
  
   Hadamard(candidate_gradient, cell_gradient);
   Hadamard(candidate_gradient, fInput);
   Hadamard(candidate_gradient, dc);
   Hadamard(input_gradient, cell_gradient);
   Hadamard(input_gradient, fCandidate);
   Hadamard(input_gradient, dci);
   Hadamard(forget_gradient, cell_gradient);
   Hadamard(forget_gradient, fInput);
   Hadamard(candidate_gradient, dc);
   Hadamard(cell_gradient, fForget);
   Copy(cell_gradient, cell_gradients_backward);
  */

	// Input gradients.
	if (input_gradient.GetNElements() > 0) {
      Multiply(input_gradient, di, weights_input);
   }

	// Forget gradients.
	if (forget_gradient.GetNElements() > 0) {
      Multiply(forget_gradient, df, weights_forget);
   }

	// Candidate gradients
	if (candidate_gradient.GetNElements() > 0) {
      Multiply(candidate_gradient, dc, weights_candidate);
   }

	// Output gradients
	if (output_gradient.GetNElements() > 0) {
      Multiply(output_gradient, dout, weights_output);
   }

	// ______________________________________________________
	// Weight gradients calculation.
	// Total there are 8 different weight matrices.

	// For input gate.
	if (input_weight_gradients.GetNElements() > 0) {
	   TransposeMultiply(input_weight_gradients, di, input, 1.0, 1.0);
	}

	if (input_state_weight_gradients.GetNElements() > 0) {
	   TransposeMultiply(input_state_weight_gradients, di, output_state, 1.0, 1.0);
	}

	// For forget gate.
	if (forget_weight_gradients.GetNElements() > 0) {
      TransposeMultiply(forget_weight_gradients, df, input, 1.0, 1.0);
	}
	 
   if (forget_state_weight_gradients.GetNElements() > 0) {
	   TransposeMultiply(forget_state_weight_gradients, df, output_state, 1.0, 1.0);
	}

	// For candidate gate.
	if (candidate_weight_gradients.GetNElements() > 0) {
	   TransposeMultiply(candidate_weight_gradients, dc, input, 1.0, 1.0);
	}
	 
   if (candidate_state_weight_gradients.GetNElements() > 0) {
	   TransposeMultiply(candidate_state_weight_gradients, dc, output_state, 1.0, 1.0);
	}

	// For output gate.
	if (output_weight_gradients.GetNElements() > 0) {
      TransposeMultiply(output_weight_gradients, dout, input, 1.0, 1.0);
	}
	 
   if (output_state_weight_gradients.GetNElements() > 0) {
      TransposeMultiply(output_state_weight_gradients, dout, output_state, 1.0, 1.0);
	}

	// We've 4 bias vectors.
   if (input_bias_gradients.GetNElements() > 0) {
      SumColumns(input_bias_gradients, di, 1.0, 1.0);
   }
	 
   if (forget_bias_gradients.GetNElements() > 0) {
      SumColumns(forget_bias_gradients, df, 1.0, 1.0);
   }
	 
   if (candidate_bias_gradients.GetNElements() > 0) {
      SumColumns(candidate_bias_gradients, dc, 1.0, 1.0);
   }
	 
   if (output_bias_gradients.GetNElements() > 0) {
      SumColumns(output_bias_gradients, dout, 1.0, 1.0);
   }

   return input_gradient;
}
} // namespace DNN
} // namespace TMVA
