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
template<typename AFloat>
AFloat TCpu<AFloat>::MeanSquaredError(const TCpuMatrix<AFloat> &Y,
                                      const TCpuMatrix<AFloat> &output)
{
   const AFloat __restrict__ *dataY      = Y.GetRawDataPointer();
   const AFloat __restrict__ *dataOutput = output.GetRawDataPointer();

   auto f = [&dataY, &dataOutput](const tbb::blocked_range<size_t> & range,
                                  AFloat partialSum)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      AFloat sum = partialSum;
      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
          AFloat error = dataY[i] - dataOutput[i];
          sum += error * error;
      }

      return sum;
   };

   auto reduction = [](AFloat sum1, AFloat sum2)
   {
      return sum1 + sum2;
   };

   AFloat norm = 1.0 / ((AFloat) Y.GetNcols() * Y.GetNrows());
   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   return norm * parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::MeanSquaredErrorGradients(
    TCpuMatrix<AFloat> & dY,
    const TCpuMatrix<AFloat> & Y,
    const TCpuMatrix<AFloat> & output)
{

         AFloat __restrict__ *dataDY     = dY.GetRawDataPointer();
   const AFloat __restrict__ *dataY      = Y.GetRawDataPointer();
   const AFloat __restrict__ *dataOutput = output.GetRawDataPointer();
   AFloat norm = 1.0 / ((AFloat) Y.GetNrows() * Y.GetNcols());

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
template<typename AFloat>
AFloat TCpu<AFloat>::CrossEntropy(const TCpuMatrix<AFloat> &Y,
                                               const TCpuMatrix<AFloat> &output)
{
   const AFloat __restrict__ *dataY      = Y.GetRawDataPointer();
   const AFloat __restrict__ *dataOutput = output.GetRawDataPointer();

   auto f = [&dataY, &dataOutput](const tbb::blocked_range<size_t> & range,
                                  AFloat partialSum)
   {
      size_t rangeBegin = range.begin();
         size_t rangeEnd   = range.end();

         AFloat sum = partialSum;
         for (size_t i = rangeBegin; i != rangeEnd; ++i) {
            AFloat y   = dataY[i];
            AFloat sig = 1.0 / (1.0 + exp(- dataOutput[i]));
            sum += y * log(sig) + (1.0 - y) * log(1.0 - sig);
         }
         return sum;
   };

   auto reduction = [](AFloat sum1, AFloat sum2)
   {
      return sum1 + sum2;
   };

   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   AFloat norm = 1.0 / ((AFloat) Y.GetNcols() * Y.GetNrows());
   return - norm * parallel_reduce(range, 0.0, f, reduction);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::CrossEntropyGradients(
    TCpuMatrix<AFloat> & dY,
    const TCpuMatrix<AFloat> & Y,
    const TCpuMatrix<AFloat> & output)
{
         AFloat __restrict__ *dataDY     = dY.GetRawDataPointer();
   const AFloat __restrict__ *dataY      = Y.GetRawDataPointer();
   const AFloat __restrict__ *dataOutput = output.GetRawDataPointer();
   AFloat norm = 1.0 / ((AFloat) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataDY, &dataY, &dataOutput, norm](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         AFloat y   = dataY[i];
         AFloat sig = 1.0 / (1.0 + exp(- dataOutput[i]));
         dataDY[i] = norm * (sig - y);
      }
   };

   tbb::blocked_range<size_t> range(0, Y.GetNElements());
   parallel_for(range, f);
}

} // namespace DNN
} // namespace TMVA
