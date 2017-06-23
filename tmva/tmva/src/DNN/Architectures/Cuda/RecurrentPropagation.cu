// @(#)root/tmva/tmva/dnn:$Id$
// Author: Saurav Shekhar 23/06/17 

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar
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
void TCuda<AFloat>::RecurrentBackward(TCudaMatrix<AFloat> & activation_gradients_backward,
                             TCudaMatrix<AFloat> & weight_gradients,
                             TCudaMatrix<AFloat> & bias_gradients,
                             TCudaMatrix<AFloat> & df,
                             const TCudaMatrix<AFloat> & activation_gradients,
                             const TCudaMatrix<AFloat> & weights,
                             const TCudaMatrix<AFloat> & activation_backward)
{
  //TODO
}

} // namespace DNN
} // namespace TMVA

