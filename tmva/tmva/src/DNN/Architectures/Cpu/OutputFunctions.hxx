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

template<typename AFloat>
void TCpu<AFloat>::Sigmoid(TCpuMatrix<AFloat> & B,
                           const TCpuMatrix<AFloat> & A)
{
   auto f = [](AFloat x) {return 1.0 / (1.0 + exp(-x));};
   B.MapFrom(f, A);
}

template<typename AFloat>
void TCpu<AFloat>::Softmax(TCpuMatrix<AFloat> & B,
                           const TCpuMatrix<AFloat> & A)
{
   const AFloat  *dataA = A.GetRawDataPointer();
         AFloat  *dataB = B.GetRawDataPointer();
   size_t n = A.GetNcols();
   size_t m = A.GetNrows();

   auto f = [&dataA, &dataB, n, m](UInt_t workerID)
   {
      AFloat sum = 0.0;
      for (size_t i = 0; i < n; i++) {
         sum += exp(dataA[workerID + i * m]);
      }
      for (size_t i = 0; i < n; i++) {
         dataB[workerID + i * m] = exp(dataA[workerID + i * m]) / sum;
      }
      return 0;
   };

   B.GetThreadExecutor().Map(f, ROOT::TSeqI(A.GetNrows()));
}

} // namespace DNN
} // namespace TMVA
