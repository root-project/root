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
  // TODO
}

} // namespace DNN
} // namespace TMVA
