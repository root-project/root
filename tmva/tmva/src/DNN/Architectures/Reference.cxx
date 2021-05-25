// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////
// Explicit instantiation of the TReference architecture class //
// template for Double_t scalar types.                        //
////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"

#include "Reference/Propagation.hxx"
#include "Reference/ActivationFunctions.hxx"
#include "Reference/Arithmetic.hxx"
#include "Reference/OutputFunctions.hxx"
#include "Reference/LossFunctions.hxx"
#include "Reference/Regularization.hxx"
#include "Reference/Initialization.hxx"
#include "Reference/Dropout.hxx"
#include "Reference/DenoisePropagation.hxx"
#include "Reference/RecurrentPropagation.hxx"

namespace TMVA {
namespace DNN  {
   template class TReference<Real_t>;
   template class TReference<Double_t>;
} // namespace TMVA
} // namespace DNN
