// @(#)root/tmva/tmva/dnn:$Id$
// Author: Saurav Shekhar 23/06/17 

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the functions required for the forward and //
 // backward propagation of activations through a neural network //
 // for CUDA architectures.                                      //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

namespace TMVA 
{
namespace DNN  
{

//____________________________________________________________________________
template<typename AFloat>
auto RecurrentLayerBackward(TCudaMatrix<AFloat> & state_gradients_backward, // BxH
                            TCudaMatrix<AFloat> & input_weight_gradients,
                            TCudaMatrix<AFloat> & state_weight_gradients,
                            TCudaMatrix<AFloat> & bias_gradients,
                            TCudaMatrix<AFloat> & df, //DxH
                            const TCudaMatrix<AFloat> & state, // BxH
                            const TCudaMatrix<AFloat> & weights_input, // HxD 
                            const TCudaMatrix<AFloat> & weights_state, // HxH
                            const TCudaMatrix<AFloat> & input,  // BxD
                            TCudaMatrix<AFloat> & input_gradient);
-> Matrix_t &
{
   //TODO
}

} // namespace DNN
} // namespace TMVA

