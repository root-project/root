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

#include "Cpu/ActivationFunctions.hxx"
#include "Cpu/Arithmetic.hxx"
#include "Cpu/Dropout.hxx"
#include "Cpu/Initialization.hxx"
#include "Cpu/LossFunctions.hxx"
#include "Cpu/OutputFunctions.hxx"
#include "Cpu/Propagation.hxx"
#include "Cpu/Regularization.hxx"
#include "Cpu/RecurrentPropagation.hxx"

namespace TMVA {
namespace DNN  {
   template class TCpu<Double_t>;
   template class TCpu<Float_t>;

// #ifndef R__HAS_TMVACPU
//    // if R__HAS_TMVACPU is not defined this file should not be compiled
//    static_assert(false,"CPU architecture is not enabled");
// #endif

} // namespace TMVA
} // namespace DNN
