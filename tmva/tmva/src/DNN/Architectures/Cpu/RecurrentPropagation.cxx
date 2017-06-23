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
void TCpu<AFloat>::RecurrentBackward(
    TCpuMatrix<AFloat> & activationGradientsBackward,
    TCpuMatrix<AFloat> & weightGradients,
    TCpuMatrix<AFloat> & biasGradients,
    TCpuMatrix<AFloat> & df,
    const TCpuMatrix<AFloat> & activationGradients,
    const TCpuMatrix<AFloat> & weights,
    const TCpuMatrix<AFloat> & activationsBackward)
{
  // TODO
}

} // namespace DNN
} // namespace TMVA
