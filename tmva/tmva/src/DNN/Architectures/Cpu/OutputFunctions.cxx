// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////
// Implementation of output functions for multi-threaded CPU //
// architectures.                                            //
///////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"

namespace TMVA
{
namespace DNN
{

template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::Sigmoid(TCpuMatrix<Real_t> & B,
                                        const TCpuMatrix<Real_t> & A)
{
   auto f = [](Real_t x) {return 1.0 / (1.0 + exp(-x));};
   B.MapFrom(f, A);
}

} // namespace DNN
} // namespace TMVA
