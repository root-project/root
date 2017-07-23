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

#include "Reference/Propagation.cxx"
#include "Reference/ActivationFunctions.cxx"
#include "Reference/OutputFunctions.cxx"
#include "Reference/LossFunctions.cxx"
#include "Reference/Regularization.cxx"
#include "Reference/Initialization.cxx"
#include "Reference/Dropout.cxx"
#include "Reference/DenoisePropagation.cxx"

namespace TMVA {
namespace DNN  {

template class TReference<Real_t>;
template class TReference<Double_t>;
} // namespace TMVA
} // namespace DNN
