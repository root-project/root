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

#include "TMVA/DNN/Architectures/Cpu.h"

#include "Cpu/ActivationFunctions.cxx"
#include "Cpu/Arithmetic.cxx"
#include "Cpu/Dropout.cxx"
#include "Cpu/Initialization.cxx"
#include "Cpu/LossFunctions.cxx"
#include "Cpu/OutputFunctions.cxx"
#include "Cpu/Propagation.cxx"
#include "Cpu/Regularization.cxx"
#include "Cpu/RecurrentPropagation.cxx"

namespace TMVA {
namespace DNN  {
template class TCpu<Double_t>;
template class TCpu<Real_t>;
} // namespace TMVA
} // namespace DNN
