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
   AFloat __restrict__ *data = A.GetRawDataPointer();

   auto fRange = [&data, dropoutProbability](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      TRandom rand(time(nullptr) + rangeBegin);

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
          AFloat r = rand.Uniform();
          data[i] = (r > dropoutProbability) ? 0.0 : data[i] / dropoutProbability;
      }
   };

   tbb::blocked_range<size_t> range(0, A.GetNElements());
   parallel_for(range, fRange);
}

} // namespace DNN
} // namespace TMVA
