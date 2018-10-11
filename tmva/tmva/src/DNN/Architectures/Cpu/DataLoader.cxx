// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Implementation for the DataLoader for the the multi-threaded //
// CPU implementation of DNNs.                                  //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu/DataLoader.h"
#include "TMVA/Event.h"
#include <iostream>
#include <random>

namespace TMVA
{
namespace DNN
{

// TCpuBatchIterator
//______________________________________________________________________________
template<typename Data_t, typename Real_t>
TCpuBatchIterator<Data_t, Real_t>::TCpuBatchIterator(
    TCpuDataLoader<Data_t, Real_t> & dataLoader,
    size_t batchIndex)
    : fDataLoader(dataLoader), fBatchIndex(batchIndex)
{
    // Nothing to do here.
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
TCpuBatch<Real_t> TCpuBatchIterator<Data_t, Real_t>::operator*()
{
   return fDataLoader.GetBatch(fBatchIndex);
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
TCpuBatchIterator<Data_t, Real_t> & TCpuBatchIterator<Data_t, Real_t>::operator++()
{
    fBatchIndex++;
    return *this;
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
bool TCpuBatchIterator<Data_t, Real_t>::operator!=(const TCpuBatchIterator & other)
{
    return fBatchIndex != other.GetBatchIndex();
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
bool TCpuBatchIterator<Data_t, Real_t>::operator==(const TCpuBatchIterator & other)
{
    return fBatchIndex == other.GetBatchIndex();
}

// TCpuDataLoader
//______________________________________________________________________________
template<typename Data_t, typename Real_t>
TCpuDataLoader<Data_t, Real_t>::TCpuDataLoader(const Data_t &input,
                                               size_t nsamples,
                                               size_t batchSize,
                                               size_t ninputFeatures,
                                               size_t noutputFeatures,
                                               size_t bufferSize)
    : fInput(input), fNSamples(nsamples), fBatchSize(batchSize),
      fBufferSize(bufferSize), fNInputFeatures(ninputFeatures),
      fNOutputFeatures(noutputFeatures), fNBatches(nsamples / batchSize),
      fInputMatrices(), fOutputMatrices(), fSampleIndices()
{
   fInputMatrices.reserve(fBufferSize);
   fOutputMatrices.reserve(fBufferSize);
   for (size_t i = 0; i < fBufferSize; i++) {
      fInputMatrices.emplace_back(fBatchSize, fNInputFeatures);
      fOutputMatrices.emplace_back(fBatchSize, fNOutputFeatures);
   }

   fSampleIndices.reserve(fNBatches);
   for (size_t i = 0; i < fNSamples; i++) {
      fSampleIndices.emplace_back(i);
   }
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
inline void TCpuDataLoader<Data_t, Real_t>::CopyData(size_t batchIndex)
{
   auto copy = [this](UInt_t workerID)
   {
      CopyBatch(this->fInputMatrices[workerID % this->fBufferSize],
                this->fOutputMatrices[workerID % this->fBufferSize],
                this->fInput,
                this->fSampleIndices.begin() + sampleIndex,
                this->fSampleIndices.begin() + sampleIndex + this->fBatchSize);
      sampleIndex += this->fBatchSize;
      return 0;
   };

   size_t end   = std::min(batchIndex + fBufferSize, fNBatches);
   size_t start = batchIndex;
   ROOT::TThreadExecutor pool{};
   pool.Map(copy, ROOT::TSeqI(start, end));
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
TCpuBatch<Real_t> TCpuDataLoader<Data_t, Real_t>::GetBatch(size_t batchIndex)
{
   size_t fBufferIndex = batchIndex % fBufferSize;
   if (fBufferIndex == 0) {
      CopyData(batchIndex);
   }
   return TCpuBatch<Real_t>(fInputMatrices[fBufferIndex],
                            fOutputMatrices[fBufferIndex]);
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
auto TCpuDataLoader<Data_t, Real_t>::begin()
    -> BatchIterator_t
{
   std::shuffle(fSampleIndices.begin(), fSampleIndices.end(), std::default_random_engine{});
   return BatchIterator_t(*this, 0);
}

//______________________________________________________________________________
template<typename Data_t, typename Real_t>
auto TCpuDataLoader<Data_t, Real_t>::end()
    -> BatchIterator_t
{
   return BatchIterator_t(*this, fNBatches);
}

//______________________________________________________________________________
template <>
void TCpuDataLoader<MatrixInput_t, Double_t>::CopyBatch(
    Matrix_t &inputMatrix,
    Matrix_t &outputMatrix,
    const MatrixInput_t &input,
    IndexIterator_t indexBegin,
    IndexIterator_t indexEnd)
{
   auto &in  = std::get<0>(input);
   auto &out = std::get<1>(input);

   size_t batchIndex = 0;
   for (IndexIterator_t i = indexBegin; i != indexEnd; i++) {
      size_t index = *i;
      for (size_t j = 0; j < (size_t) in.GetNcols(); j++) {
         inputMatrix(batchIndex, j) = in(index, j);
      }
      for (size_t j = 0; j < (size_t) out.GetNcols(); j++) {
         outputMatrix(batchIndex, j) = out(index, j);
      }
      batchIndex++;
   }
}

//______________________________________________________________________________
template <>
void TCpuDataLoader<TMVAInput_t, Double_t>::CopyBatch(
    Matrix_t &inputMatrix,
    Matrix_t &outputMatrix,
    const TMVAInput_t &input,
    IndexIterator_t indexBegin,
    IndexIterator_t indexEnd)
{
   size_t batchIndex = 0;
   for (IndexIterator_t i = indexBegin; i != indexEnd; i++) {
      size_t index = *i;
      Event *event = input.at(index);
      for (size_t j = 0; j < event->GetNVariables(); j++) {
         inputMatrix(batchIndex, j) = event->GetValue(j);
      }
      if (event->GetNTargets() > 0) {
         for (size_t j = 0; j < event->GetNTargets(); j++) {
            outputMatrix(batchIndex, j) = event->GetTarget(j);
         }
      } else {
         outputMatrix(batchIndex, 0) = (event->GetClass() == 0) ? 1.0 : 0.0;
         batchIndex++;
      }
   }
}

// Explicit instantiation.
//______________________________________________________________________________
template class TCpuDataLoader<MatrixInput_t, Double_t>;
template class TCpuDataLoader<TMVAInput_t, Double_t>;
template class TCpuBatchIterator<MatrixInput_t, Double_t>;
template class TCpuBatchIterator<TMVAInput_t, Double_t>;
template class TCpuBatch<Double_t>;

} // namespace DNN
} // namespace TMVA
