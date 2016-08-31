// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////
// Implementation for the DataLoader for the reference //
// implementation.                                     //
/////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference/DataLoader.h"
#include "TMVA/Event.h"

namespace TMVA
{
namespace DNN
{

template<typename Data_t, typename Real_t>
TReferenceDataLoader<Data_t, Real_t>::TReferenceDataLoader(const Data_t &input,
                                                         size_t nsamples,
                                                         size_t batchSize,
                                                         size_t ninputFeatures,
                                                         size_t noutputFeatures)
    : fInput(input), fNSamples(nsamples), fBatchSize(batchSize),
      fNInputFeatures(ninputFeatures), fNOutputFeatures(noutputFeatures),
      fNBatches(nsamples / batchSize), fInputMatrices(), fOutputMatrices(),
      fBatches(), fSampleIndices()
{
   fInputMatrices.reserve(fNBatches);
   fOutputMatrices.reserve(fNBatches);
   for (size_t i = 0; i < fNBatches; i++) {
      fInputMatrices.emplace_back(fBatchSize, fNInputFeatures);
      fOutputMatrices.emplace_back(fBatchSize, fNOutputFeatures);
   }

   fBatches.reserve(fNBatches);
   fSampleIndices.reserve(fNBatches);

   for (size_t i = 0; i < fNSamples; i++) {
      fSampleIndices.emplace_back(i);
   }
}

template<typename Data_t, typename Real_t>
auto TReferenceDataLoader<Data_t, Real_t>::begin()
    -> BatchIterator_t
{
   random_shuffle(fSampleIndices.begin(), fSampleIndices.end());
   fBatches.clear();
   fBatches.reserve(fNSamples);

   size_t sampleIndex = 0;
   for (size_t batchIndex = 0; batchIndex < fNBatches; batchIndex++) {

      CopyBatch(fInputMatrices[batchIndex],
                fOutputMatrices[batchIndex],
                fInput,
                fSampleIndices.begin() + sampleIndex,
                fSampleIndices.begin() + sampleIndex + fBatchSize);
      fBatches.emplace_back(fInputMatrices[batchIndex],
                            fOutputMatrices[batchIndex]);
      sampleIndex += fBatchSize;
   }
   return fBatches.begin();
}

template<typename Data_t, typename Real_t>
auto TReferenceDataLoader<Data_t, Real_t>::end()
    -> BatchIterator_t
{
   return fBatches.end();
}

using MatrixInput_t = std::pair<const TMatrixT<Double_t>&,
                                const TMatrixT<Double_t> &>;

template <>
void TReferenceDataLoader<MatrixInput_t, Double_t>::CopyBatch(
    Matrix_t &inputMatrix,
    Matrix_t &outputMatrix,
    const MatrixInput_t &input,
    IndexIterator_t indexBegin,
    IndexIterator_t indexEnd)
{
   const Matrix_t &in  = std::get<0>(input);
   const Matrix_t &out = std::get<1>(input);

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

using TMVAInput_t   = std::vector<TMVA::Event*>;

template <>
void TReferenceDataLoader<TMVAInput_t, Double_t>::CopyBatch(
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

template class TReferenceDataLoader<MatrixInput_t, Double_t>;
template class TReferenceDataLoader<TMVAInput_t, Double_t>;

} // namespace DNN
} // namespace TMVA
