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
template<typename Real_t, bool doProfiling>
Real_t TCpu<Real_t, doProfiling>::L1Regularization(const TCpuMatrix<Real_t> &Weights)
{
   const Real_t __restrict__ *data = Weights.GetRawDataPointer();

   auto f = [&data](const tbb::blocked_range<size_t> & range,
                    Real_t partialSum)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      Real_t sum = partialSum;
      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         sum += fabs(data[i]);
      }
      return sum;
   };

   auto reduction = [](Real_t sum1, Real_t sum2)
   {
      return sum1 + sum2;
   };

<<<<<<< HEAD
   tbb::blocked_range<size_t> range(0, Weights.GetNElements());
=======
   auto & elements = Weights.GetElements();
   tbb::blocked_range<size_t> range(0, elements.size());
>>>>>>> 055354f6262b9c10847d24ce0f683235cca9d892
   return parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::AddL1RegularizationGradients(
    TCpuMatrix<Real_t> & B,
    const TCpuMatrix<Real_t> & A,
    Real_t weightDecay)
{

         Real_t __restrict__ *dataB     =  B.GetRawDataPointer();
   const Real_t __restrict__ *dataA      = A.GetRawDataPointer();

   auto f = [&dataA, &dataB, weightDecay](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         Real_t sign = (dataA[i] < 0.0) ? -1.0 : 1.0;
         dataB[i] += weightDecay * sign;
      }
   };

<<<<<<< HEAD
   tbb::blocked_range<size_t> range(0, A.GetNElements());
=======
   auto & elements = A.GetElements();
   tbb::blocked_range<size_t> range(0, elements.size());
>>>>>>> 055354f6262b9c10847d24ce0f683235cca9d892
   parallel_for(range, f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
Real_t TCpu<Real_t, doProfiling>::L2Regularization(const TCpuMatrix<Real_t> &Weights)
{
   const Real_t __restrict__ *data = Weights.GetRawDataPointer();

   auto f = [&data](const tbb::blocked_range<size_t> & range,
                    Real_t partialSum)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      Real_t sum = partialSum;
      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
          sum += data[i] * data[i];
      }
      return sum;
   };

   auto reduction = [](Real_t sum1, Real_t sum2)
   {
      return sum1 + sum2;
   };

<<<<<<< HEAD
   tbb::blocked_range<size_t> range(0, Weights.GetNElements());
=======
   auto & elements = Weights.GetElements();
   tbb::blocked_range<size_t> range(0, elements.size());
>>>>>>> 055354f6262b9c10847d24ce0f683235cca9d892
   return parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::AddL2RegularizationGradients(
    TCpuMatrix<Real_t> & B,
    const TCpuMatrix<Real_t> & A,
    Real_t weightDecay)
{

         Real_t __restrict__ *dataB     =  B.GetRawDataPointer();
   const Real_t __restrict__ *dataA      = A.GetRawDataPointer();

   auto f = [&dataA, &dataB, weightDecay](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         dataB[i] += 2.0 * weightDecay * dataA[i];
      }
   };

<<<<<<< HEAD
   tbb::blocked_range<size_t> range(0, A.GetNElements());
=======
   auto & elements = A.GetElements();
   tbb::blocked_range<size_t> range(0, elements.size());
>>>>>>> 055354f6262b9c10847d24ce0f683235cca9d892
   parallel_for(range, f);
}

} // namespace DNN
} // namespace TMVA
