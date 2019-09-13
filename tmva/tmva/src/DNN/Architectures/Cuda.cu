// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////
// Explicit instantiation of the TCuda architecture class with //
// for Double_t and Real_t floating point types.               //
/////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "Cuda/Propagation.cu"
#include "Cuda/Arithmetic.cu"
#include "Cuda/ActivationFunctions.cu"
#include "Cuda/OutputFunctions.cu"
#include "Cuda/LossFunctions.cu"
#include "Cuda/Regularization.cu"
#include "Cuda/Initialization.cu"
#include "Cuda/Dropout.cu"
#include "Cuda/RecurrentPropagation.cu"

namespace TMVA {
namespace DNN  {

template class TCuda<Real_t>;
template class TCuda<Double_t>;


#ifndef R__HAS_TMVAGPU
   // if R__HAS_TMVAGPU is not defined this file should not be compiled 
   static_assert(false,"GPU/CUDA architecture is not enabled"); 
#endif


} // namespace tmva
} // namespace dnn
