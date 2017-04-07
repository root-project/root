// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TRandom.h"

/////////////////////////////////////////////////////////////////////
// Implementation of Dropout for multi-threaded CPU architectures. //
/////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Dropout(TCpuMatrix<AFloat> &A,
                           AFloat dropoutProbability)
{
   AFloat *data = A.GetRawDataPointer();

   auto f = [&data, dropoutProbability](UInt_t workerID)
   {
      TRandom rand(time(nullptr) + workerID);
      AFloat r = rand.Uniform();
      data[workerID] = (r > dropoutProbability) ? 0.0 : data[workerID] / dropoutProbability;
      return 0;
   };

   A.GetThreadExecutor().Map(f, ROOT::TSeqI(A.GetNElements()));
}

} // namespace DNN
} // namespace TMVA
