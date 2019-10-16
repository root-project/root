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
//#if 0
//____________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::DropoutForward(TCpuTensor<AFloat> & A, 
                                  TDescriptors * /*descriptors*/,
                                  TWorkspace   * /*workspace*/, 
                                  AFloat dropoutProbability)
{
   AFloat *data = A.GetData();

   TRandom & dlRand = TCpu<AFloat>::GetRandomGenerator();
   size_t seed = dlRand.Integer(4294967295);   // use 2^32-1

   size_t nElements =  A.GetSize();
   const size_t nSteps = TCpuMatrix<AFloat>::GetNWorkItems(nElements);

   // apply droput. The probability is actually the probability to keep the node
   // (i.e. 1 - dropout_prob)
   auto f = [&data, dropoutProbability, &nSteps, &nElements, &seed](UInt_t workerID)
   {
      TRandom rand(seed+workerID);
      size_t iMax = std::min(workerID+nSteps,nElements); 
      for (size_t i = workerID; i < iMax; ++i) { 
         AFloat r = rand.Uniform();
         data[i] = (r > dropoutProbability) ? 0.0 : data[i] / dropoutProbability;
      }
      return 0;
   };

#ifdef DL_USE_MTE
   TCpuMatrix<AFloat>::GetThreadExecutor().Foreach(f, ROOT::TSeqI(0,nElements,nSteps));
#else
   for (size_t i = 0;  i < nElements; i+=nSteps)
      f(i); 
#endif
}
   // old impl (to be removed)  
#if 0
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

   A.GetThreadExecutor().Map(f, ROOT::TSeqI(A.GetNoElements()));
}
#endif

   

} // namespace DNN
} // namespace TMVA
