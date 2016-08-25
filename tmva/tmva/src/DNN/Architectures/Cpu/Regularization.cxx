// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Implementation of the regularization functionals and gradients //
// for the multi-threaded CPU implementation using tbb.           //
////////////////////////////////////////////////////////////////////

#include "tbb/tbb.h"
#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename AFloat>
AFloat TCpu<AFloat>::L1Regularization(const TCpuMatrix<AFloat> &Weights)
{
   const AFloat __restrict__ *data = Weights.GetRawDataPointer();

   auto f = [&data](const tbb::blocked_range<size_t> & range,
                    AFloat partialSum)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      AFloat sum = partialSum;
      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         sum += fabs(data[i]);
      }
      return sum;
   };

   auto reduction = [](AFloat sum1, AFloat sum2)
   {
      return sum1 + sum2;
   };

   tbb::blocked_range<size_t> range(0, Weights.GetNElements());
   return parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::AddL1RegularizationGradients(
    TCpuMatrix<AFloat> & B,
    const TCpuMatrix<AFloat> & A,
    AFloat weightDecay)
{

         AFloat __restrict__ *dataB     =  B.GetRawDataPointer();
   const AFloat __restrict__ *dataA      = A.GetRawDataPointer();

   auto f = [&dataA, &dataB, weightDecay](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         AFloat sign = (dataA[i] < 0.0) ? -1.0 : 1.0;
         dataB[i] += weightDecay * sign;
      }
   };

   tbb::blocked_range<size_t> range(0, A.GetNElements());
   parallel_for(range, f);
}

//______________________________________________________________________________
template<typename AFloat>
AFloat TCpu<AFloat>::L2Regularization(const TCpuMatrix<AFloat> &Weights)
{
   const AFloat __restrict__ *data = Weights.GetRawDataPointer();

   auto f = [&data](const tbb::blocked_range<size_t> & range,
                    AFloat partialSum)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      AFloat sum = partialSum;
      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
          sum += data[i] * data[i];
      }
      return sum;
   };

   auto reduction = [](AFloat sum1, AFloat sum2)
   {
      return sum1 + sum2;
   };

   tbb::blocked_range<size_t> range(0, Weights.GetNElements());
   return parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::AddL2RegularizationGradients(
    TCpuMatrix<AFloat> & B,
    const TCpuMatrix<AFloat> & A,
    AFloat weightDecay)
{

         AFloat __restrict__ *dataB     =  B.GetRawDataPointer();
   const AFloat __restrict__ *dataA      = A.GetRawDataPointer();

   auto f = [&dataA, &dataB, weightDecay](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         dataB[i] += 2.0 * weightDecay * dataA[i];
      }
   };

   tbb::blocked_range<size_t> range(0, A.GetNElements());
   parallel_for(range, f);
}

} // namespace DNN
} // namespace TMVA
