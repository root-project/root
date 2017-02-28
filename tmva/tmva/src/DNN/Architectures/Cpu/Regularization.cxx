// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Implementation of the regularization functionals and gradients    //
// for the multi-threaded CPU implementation using Roots TThreadExecutor. //
///////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename AFloat>
AFloat TCpu<AFloat>::L1Regularization(const TCpuMatrix<AFloat> &Weights)
{
   const AFloat  *data = Weights.GetRawDataPointer();
   std::vector<AFloat> temp(Weights.GetNElements());

   auto f = [&data, &temp](UInt_t workerID)
   {
      temp[workerID] = fabs(data[workerID]);
      return 0;
   };

   auto reduction = [](const std::vector<AFloat> & v )
   {
      return std::accumulate(v.begin(),v.end(),AFloat{});
   };
   // auto reduction = [](AFloat sum1, AFloat sum2)
   // {
   //    return sum1 + sum2;
   // };

   Weights.GetThreadExecutor().Map(f, ROOT::TSeqI(Weights.GetNElements()));
   return Weights.GetThreadExecutor().Reduce(temp, reduction);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::AddL1RegularizationGradients(
    TCpuMatrix<AFloat> & B,
    const TCpuMatrix<AFloat> & A,
    AFloat weightDecay)
{
         AFloat  *dataB     =  B.GetRawDataPointer();
   const AFloat  *dataA      = A.GetRawDataPointer();

   auto f = [&dataA, &dataB, weightDecay](UInt_t workerID)
   {
      AFloat sign = (dataA[workerID] < 0.0) ? -1.0 : 1.0;
      dataB[workerID] += weightDecay * sign;
      return 0;
   };

   B.GetThreadExecutor().Map(f, ROOT::TSeqI(B.GetNElements()));
}

//______________________________________________________________________________
template<typename AFloat>
AFloat TCpu<AFloat>::L2Regularization(const TCpuMatrix<AFloat> &Weights)
{
   const AFloat  *data = Weights.GetRawDataPointer();
   std::vector<AFloat> temp(Weights.GetNElements());

   auto f = [&data, &temp](UInt_t workerID)
   {
      temp[workerID] = data[workerID] * data[workerID];
      return 0;
   };

   auto reduction = [](const std::vector<AFloat> & v )
   {
      return std::accumulate(v.begin(),v.end(),AFloat{});
   };
   // auto reduction = [](AFloat sum1, AFloat sum2)
   // {
   //    return sum1 + sum2;
   // };

   Weights.GetThreadExecutor().Map(f, ROOT::TSeqI(Weights.GetNElements()));
   return Weights.GetThreadExecutor().Reduce(temp, reduction);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::AddL2RegularizationGradients(
    TCpuMatrix<AFloat> & B,
    const TCpuMatrix<AFloat> & A,
    AFloat weightDecay)
{
         AFloat  *dataB     =  B.GetRawDataPointer();
   const AFloat  *dataA      = A.GetRawDataPointer();

   auto f = [&dataA, &dataB, weightDecay](UInt_t workerID)
   {
      dataB[workerID] += 2.0 * weightDecay * dataA[workerID];
      return 0;
   };

   B.GetThreadExecutor().Map(f, ROOT::TSeqI(B.GetNElements()));
}

} // namespace DNN
} // namespace TMVA
