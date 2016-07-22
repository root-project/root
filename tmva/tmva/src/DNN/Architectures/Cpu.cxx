// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////
// Explicit instantiation of the CPU architecture class. //
///////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"

#include "Cpu/Propagation.cxx"
#include "Cpu/Arithmetic.cxx"
#include "Cpu/ActivationFunctions.cxx"
#include "Cpu/OutputFunctions.cxx"
#include "Cpu/LossFunctions.cxx"
#include "Cpu/Regularization.cxx"
#include "Cpu/Initialization.cxx"
#include "Cpu/Dropout.cxx"

namespace TMVA {
namespace DNN  {
template class TCpu<Double_t, false>;
} // namespace TMVA
} // namespace DNN
