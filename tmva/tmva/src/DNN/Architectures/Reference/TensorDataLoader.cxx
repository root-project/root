// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TTensorDataLoader                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Specialization of the Tensor Data Loader Class                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

///////////////////////////////////////////////////////////////////
// Specializations of Copy functions for the TensorDataLoader    //
// specialized for the reference architecture.                   //
///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA {
namespace DNN {

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TReference<Real_t>>::CopyTensorInput(std::vector<TMatrixT<Real_t>> &tensor,
                                                                         IndexIterator_t sampleIterator)
{
   const std::vector<TMatrixT<Double_t>> &inputTensor = std::get<0>(fData);

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < fBatchHeight; j++) {
         for (size_t k = 0; k < fBatchWidth; k++) {
            tensor[i](j, k) = static_cast<Real_t>(inputTensor[sampleIndex](j, k));
         }
      }

      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TReference<Real_t>>::CopyTensorOutput(TMatrixT<Real_t> &matrix,
                                                                          IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         matrix(i, j) = static_cast<Real_t>(outputMatrix(sampleIndex, j));
      }

      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TReference<Real_t>>::CopyTensorWeights(TMatrixT<Real_t> &matrix,
                                                                           IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &weightMatrix = std::get<2>(fData);

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      matrix(i, 0) = static_cast<Real_t>(weightMatrix(sampleIndex, 0));
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TReference<Double_t>>::CopyTensorInput(std::vector<TMatrixT<Double_t>> &tensor,
                                                                           IndexIterator_t sampleIterator)
{
   const std::vector<TMatrixT<Double_t>> &inputTensor = std::get<0>(fData);

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < fBatchHeight; j++) {
         for (size_t k = 0; k < fBatchWidth; k++) {
            tensor[i](j, k) = inputTensor[sampleIndex](j, k);
         }
      }

      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TReference<Double_t>>::CopyTensorOutput(TMatrixT<Double_t> &matrix,
                                                                            IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &outputMatrix = std::get<1>(fData);
   size_t n = outputMatrix.GetNcols();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < n; j++) {
         matrix(i, j) = outputMatrix(sampleIndex, j);
      }

      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TensorInput, TReference<Double_t>>::CopyTensorWeights(TMatrixT<Double_t> &matrix,
                                                                             IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &weightMatrix = std::get<2>(fData);

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      matrix(i, 0) = weightMatrix(sampleIndex, 0);
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TReference<Real_t>>::CopyTensorInput(std::vector<TMatrixT<Real_t>> &tensor,
                                                                         IndexIterator_t sampleIterator)
{
   // one event, one  example in the batch
   Event *event = fData.front();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < fBatchHeight; j++) {
         for (size_t k = 0; k < fBatchWidth; k++) {
            event = fData[sampleIndex];
            tensor[i](j, k) = static_cast<Real_t>(event->GetValue(j * fBatchHeight + k));
         }
      }

      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TReference<Real_t>>::CopyTensorOutput(TMatrixT<Real_t> &matrix,
                                                                          IndexIterator_t sampleIterator)
{
   Event *event = fData.front();
   Int_t n = matrix.GetNcols();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      event = fData[sampleIndex];

      for (size_t j = 0; j < n; j++) {
         if (event->GetNTargets() == 0) {
            if (n == 1) {
               matrix(i, j) = (event->GetClass() == 0) ? 1.0 : 0.0;
            } else {
               matrix(i, j) = 0.0;
               if (j == event->GetClass()) {
                  matrix(i, j) = 1.0;
               }
            }
         } else {
            matrix(i, j) = static_cast<Real_t>(event->GetTarget(j));
         }
      }
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TReference<Real_t>>::CopyTensorWeights(TMatrixT<Real_t> &matrix,
                                                                           IndexIterator_t sampleIterator)
{
   Event *event = fData.front();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      event = fData[sampleIndex];
      matrix(i, 0) = static_cast<Real_t>(event->GetWeight());
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TReference<Double_t>>::CopyTensorInput(std::vector<TMatrixT<Double_t>> &tensor,
                                                                           IndexIterator_t sampleIterator)
{
   // one event, one  example in the batch
   Event *event = fData.front();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator;
      for (size_t j = 0; j < fBatchHeight; j++) {
         for (size_t k = 0; k < fBatchWidth; k++) {
            event = fData[sampleIndex];
            tensor[i](j, k) = event->GetValue(j * fBatchHeight + k);
         }
      }

      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TReference<Double_t>>::CopyTensorOutput(TMatrixT<Double_t> &matrix,
                                                                            IndexIterator_t sampleIterator)
{
   Event *event = fData.front();
   Int_t n = matrix.GetNcols();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      event = fData[sampleIndex];

      for (size_t j = 0; j < n; j++) {
         if (event->GetNTargets() == 0) {
            if (n == 1) {
               matrix(i, j) = (event->GetClass() == 0) ? 1.0 : 0.0;
            } else {
               matrix(i, j) = 0.0;
               if (j == event->GetClass()) {
                  matrix(i, j) = 1.0;
               }
            }
         } else {
            matrix(i, j) = event->GetTarget(j);
         }
      }
   }
}

//______________________________________________________________________________
template <>
void TTensorDataLoader<TMVAInput_t, TReference<Double_t>>::CopyTensorWeights(TMatrixT<Double_t> &matrix,
                                                                             IndexIterator_t sampleIterator)
{
   Event *event = fData.front();

   for (size_t i = 0; i < fBatchSize; i++) {
      size_t sampleIndex = *sampleIterator++;
      event = fData[sampleIndex];
      matrix(i, 0) = event->GetWeight();
   }
}

} // namespace DNN
} // namespace TMVA