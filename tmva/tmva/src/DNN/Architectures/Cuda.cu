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
// and without profiling.                                      //
/////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"

#include "Cuda/Propagation.cu"
#include "Cuda/Arithmetic.cu"
#include "Cuda/ActivationFunctions.cu"
#include "Cuda/OutputFunctions.cu"
#include "Cuda/LossFunctions.cu"
#include "Cuda/Regularization.cu"
#include "Cuda/Initialization.cu"
#include "Cuda/Dropout.cu"

namespace TMVA {
namespace DNN  {
template class TCuda<false>;
} // namespace TMVA
} // namespace DNN
