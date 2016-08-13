// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 /////////////////////////////////////////////////////////////////////
 // Implementation of the loss functions for the multi-threaded CPU //
 // implementation using tbb and BLAS.                              //
 /////////////////////////////////////////////////////////////////////

#include "tbb/tbb.h"
#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
Real_t TCpu<Real_t, doProfiling>::MeanSquaredError(const TCpuMatrix<Real_t> &Y,
                                                   const TCpuMatrix<Real_t> &output)
{
   const Real_t __restrict__ *dataY      = Y.GetRawDataPointer();
   const Real_t __restrict__ *dataOutput = output.GetRawDataPointer();

   auto f = [&dataY, &dataOutput](const tbb::blocked_range<size_t> & range,
                                  Real_t partialSum)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      Real_t sum = partialSum;
      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
          Real_t error = dataY[i] - dataOutput[i];
          sum += error * error;
      }

      return sum;
   };

   auto reduction = [](Real_t sum1, Real_t sum2)
   {
      return sum1 + sum2;
   };

   Real_t norm = 1.0 / ((Real_t) Y.GetNcols() * Y.GetNrows());
   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   return norm * parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::MeanSquaredErrorGradients(
    TCpuMatrix<Real_t> & dY,
    const TCpuMatrix<Real_t> & Y,
    const TCpuMatrix<Real_t> & output)
{

         Real_t __restrict__ *dataDY     = dY.GetRawDataPointer();
   const Real_t __restrict__ *dataY      = Y.GetRawDataPointer();
   const Real_t __restrict__ *dataOutput = output.GetRawDataPointer();
   Real_t norm = 1.0 / ((Real_t) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataDY, &dataY, &dataOutput, norm](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         dataDY[i] = - 2.0 * norm * (dataY[i] - dataOutput[i]);
      }
   };

   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   parallel_for(range, f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
Real_t TCpu<Real_t, doProfiling>::CrossEntropy(const TCpuMatrix<Real_t> &Y,
                                               const TCpuMatrix<Real_t> &output)
{
   const Real_t __restrict__ *dataY      = Y.GetRawDataPointer();
   const Real_t __restrict__ *dataOutput = output.GetRawDataPointer();

   auto f = [&dataY, &dataOutput](const tbb::blocked_range<size_t> & range,
                                  Real_t partialSum)
   {
      size_t rangeBegin = range.begin();
         size_t rangeEnd   = range.end();

         Real_t sum = partialSum;
         for (size_t i = rangeBegin; i != rangeEnd; ++i) {
            Real_t y   = dataY[i];
            Real_t sig = 1.0 / (1.0 + exp(- dataOutput[i]));
            sum += y * log(sig) + (1.0 - y) * log(1.0 - sig);
         }
         return sum;
   };

   auto reduction = [](Real_t sum1, Real_t sum2)
   {
      return sum1 + sum2;
   };

   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   Real_t norm = 1.0 / ((Real_t) Y.GetNcols() * Y.GetNrows());
   return - norm * parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::CrossEntropyGradients(
    TCpuMatrix<Real_t> & dY,
    const TCpuMatrix<Real_t> & Y,
    const TCpuMatrix<Real_t> & output)
{
         Real_t __restrict__ *dataDY     = dY.GetRawDataPointer();
   const Real_t __restrict__ *dataY      = Y.GetRawDataPointer();
   const Real_t __restrict__ *dataOutput = output.GetRawDataPointer();
   Real_t norm = 1.0 / ((Real_t) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataDY, &dataY, &dataOutput, norm](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         Real_t y   = dataY[i];
         Real_t sig = 1.0 / (1.0 + exp(- dataOutput[i]));
         dataDY[i] = norm * (sig - y);
      }
   };

   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   parallel_for(range, f);
}

} // namespace DNN
} // namespace TMVA
