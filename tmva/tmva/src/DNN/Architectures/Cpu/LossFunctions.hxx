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
 // implementation using Roots TThreadExecutor and BLAS.                 //
 /////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Architectures/Cpu.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template <typename AFloat>
AFloat TCpu<AFloat>::MeanSquaredError(const TCpuMatrix<AFloat> &Y, const TCpuMatrix<AFloat> &output,
                                      const TCpuMatrix<AFloat> &weights)
{
   const AFloat *dataY = Y.GetRawDataPointer();
   const AFloat *dataOutput = output.GetRawDataPointer();
   const AFloat *dataWeights = weights.GetRawDataPointer();
   std::vector<AFloat> temp(Y.GetNoElements());
   size_t m = Y.GetNrows();
   AFloat norm = 1.0 / ((AFloat) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataY, &dataOutput, &dataWeights, &temp, m](UInt_t workerID) {
      AFloat dy = dataY[workerID] - dataOutput[workerID];
      temp[workerID] = dataWeights[workerID % m] * dy * dy;
      return 0;
   };

   auto reduction = [](const std::vector<AFloat> & v )
   {
      return std::accumulate(v.begin(),v.end(),AFloat{});
   };

   Y.GetThreadExecutor().Map(f, ROOT::TSeqI(Y.GetNoElements()));
   return norm * Y.GetThreadExecutor().Reduce(temp, reduction);
}

//______________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::MeanSquaredErrorGradients(TCpuMatrix<AFloat> &dY, const TCpuMatrix<AFloat> &Y,
                                             const TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &weights)
{

         AFloat  *dataDY     = dY.GetRawDataPointer();
   const AFloat  *dataY      = Y.GetRawDataPointer();
   const AFloat  *dataOutput = output.GetRawDataPointer();
   const AFloat *dataWeights = weights.GetRawDataPointer();

   size_t m = Y.GetNrows();
   AFloat norm = 1.0 / ((AFloat) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataDY, &dataY, &dataOutput, &dataWeights, m, norm](UInt_t workerID) {
      dataDY[workerID] = -2.0 * norm * (dataY[workerID] - dataOutput[workerID]);
      dataDY[workerID] *= dataWeights[workerID % m];
      return 0;
   };

   Y.GetThreadExecutor().Map(f, ROOT::TSeqI(Y.GetNoElements()));
}

//______________________________________________________________________________
template <typename AFloat>
AFloat TCpu<AFloat>::CrossEntropy(const TCpuMatrix<AFloat> &Y, const TCpuMatrix<AFloat> &output,
                                  const TCpuMatrix<AFloat> &weights)
{
   const AFloat *dataY = Y.GetRawDataPointer();
   const AFloat *dataOutput = output.GetRawDataPointer();
   const AFloat *dataWeights = weights.GetRawDataPointer();
   std::vector<AFloat> temp(Y.GetNoElements());

   size_t m = Y.GetNrows();
   AFloat norm = 1.0 / ((AFloat) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataY, &dataOutput, &dataWeights, &temp, m](UInt_t workerID) {
      AFloat y   = dataY[workerID];
      // AFloat sig = 1.0 / (1.0 + exp(- dataOutput[workerID]));
      // Use more robust formula to compute log(sig) and log(1-sig) where sig= 1./(1+exp(-x))
      // when sig is close to zero or to 1
      AFloat x = dataOutput[workerID];
      AFloat lr = 0;
      if (x < -75.)
         lr = -x ;
      else if (x > 75)
         lr = exp(-x);
      else
         lr = std::log(1. + exp(-x));

      //temp[workerID] = - (y * log(sig) + (1.0 - y) * log(1.0 - sig));
      temp[workerID] =  y * lr + (1.0 - y) * (x +lr);

      temp[workerID] *= dataWeights[workerID % m];
      return 0;
   };

   auto reduction = [](const std::vector<AFloat> & v )
   {
      return std::accumulate(v.begin(),v.end(),AFloat{});
   };

   Y.GetThreadExecutor().Map(f, ROOT::TSeqI(Y.GetNoElements()));
   return norm * Y.GetThreadExecutor().Reduce(temp, reduction);
}

//______________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::CrossEntropyGradients(TCpuMatrix<AFloat> &dY, const TCpuMatrix<AFloat> &Y,
                                         const TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &weights)
{
         AFloat  *dataDY     = dY.GetRawDataPointer();
   const AFloat  *dataY      = Y.GetRawDataPointer();
   const AFloat  *dataOutput = output.GetRawDataPointer();
   const AFloat *dataWeights = weights.GetRawDataPointer();

   size_t m = Y.GetNrows();
   AFloat norm = 1.0 / ((AFloat) Y.GetNrows() * Y.GetNcols());

   auto f = [&dataDY, &dataY, &dataOutput, &dataWeights, m, norm](UInt_t workerID) {
      AFloat y   = dataY[workerID];
      AFloat sig = 1.0 / (1.0 + exp(- dataOutput[workerID]));
      dataDY[workerID] = norm * (sig - y);
      dataDY[workerID] *= dataWeights[workerID % m];
      return 0;
   };

   Y.GetThreadExecutor().Map(f, ROOT::TSeqI(Y.GetNoElements()));
}

//______________________________________________________________________________
template <typename AFloat>
AFloat TCpu<AFloat>::SoftmaxCrossEntropy(const TCpuMatrix<AFloat> &Y, const TCpuMatrix<AFloat> &output,
                                         const TCpuMatrix<AFloat> &weights)
{
   const AFloat  *dataY      = Y.GetRawDataPointer();
   const AFloat  *dataOutput = output.GetRawDataPointer();
   const AFloat *dataWeights = weights.GetRawDataPointer();

   std::vector<AFloat> temp(Y.GetNrows());
   size_t m = Y.GetNrows();
   size_t n = Y.GetNcols();
   AFloat norm = 1.0 / ((AFloat) m);

   auto f = [&dataY, &dataOutput, &dataWeights, &temp, n, m](UInt_t workerID) {
      AFloat sum = 0.0;
      for (size_t j = 0; j < n; j++) {
         sum += exp(dataOutput[workerID + j * m]);
      }
      for (size_t j = 0; j < n; j++) {
         temp[workerID] -=
            dataY[workerID + j * m] * log(exp(dataOutput[workerID + j * m]) / sum);
      }
      temp[workerID] *= dataWeights[workerID];
      return 0;
   };

   auto reduction = [](const std::vector<AFloat> & v )
   {
      return std::accumulate(v.begin(),v.end(),AFloat{});
   };

   Y.GetThreadExecutor().Map(f, ROOT::TSeqI(Y.GetNrows()));
   return norm * Y.GetThreadExecutor().Reduce(temp, reduction);
}

//______________________________________________________________________________
template <typename AFloat>
void TCpu<AFloat>::SoftmaxCrossEntropyGradients(TCpuMatrix<AFloat> &dY, const TCpuMatrix<AFloat> &Y,
                                                const TCpuMatrix<AFloat> &output, const TCpuMatrix<AFloat> &weights)
{
         AFloat  *dataDY     = dY.GetRawDataPointer();
   const AFloat  *dataY      = Y.GetRawDataPointer();
   const AFloat  *dataOutput = output.GetRawDataPointer();
   const AFloat *dataWeights = weights.GetRawDataPointer();

   size_t m = Y.GetNrows();
   size_t n = Y.GetNcols();
   AFloat norm = 1.0 / ((AFloat) m);

   auto f = [&dataDY, &dataY, &dataOutput, &dataWeights, norm, n, m](UInt_t workerID) {
      AFloat sum  = 0.0;
      AFloat sumY = 0.0;
      AFloat weight = dataWeights[workerID];
      for (size_t j = 0; j < n; j++) {
         sum  += exp(dataOutput[workerID + j * m]);
         sumY += dataY[workerID + j * m];
      }
      for (size_t j = 0; j < n; j++) {
         dataDY[workerID + j * m] =
            norm * (exp(dataOutput[workerID + j * m]) / sum * sumY - dataY[workerID + j * m]);
         dataDY[workerID + j * m] *= weight;
      }
      return 0;
   };

   Y.GetThreadExecutor().Map(f, ROOT::TSeqI(Y.GetNrows()));
}

} // namespace DNN
} // namespace TMVA
