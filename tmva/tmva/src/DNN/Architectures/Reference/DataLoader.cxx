// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 06/06/17

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////
// Specializations of Copy functions for the DataLoader    //
// specialized for the reference architecture.             //
/////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DataSetInfo.h"

namespace TMVA {
namespace DNN {

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TReference<Real_t>>::CopyInput(TMatrixT<Real_t> &matrix, IndexIterator_t sampleIterator)
{
   const TMatrixT<Real_t> &input = std::get<0>(fData);
   Int_t m = matrix.GetNrows();
   Int_t n = input.GetNcols();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator;
      for (Int_t j = 0; j < n; j++) {
         matrix(i, j) = static_cast<Real_t>(input(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TReference<Real_t>>::CopyOutput(TMatrixT<Real_t> &matrix,
                                                                IndexIterator_t sampleIterator)
{
   const TMatrixT<Real_t> &output = std::get<1>(fData);
   Int_t m = matrix.GetNrows();
   Int_t n = output.GetNcols();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator;
      for (Int_t j = 0; j < n; j++) {
         matrix(i, j) = static_cast<Real_t>(output(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TReference<Real_t>>::CopyWeights(TMatrixT<Real_t> &matrix,
                                                                 IndexIterator_t sampleIterator)
{
   const TMatrixT<Real_t> &weights = std::get<2>(fData);
   Int_t m = matrix.GetNrows();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator;
      matrix(i, 0) = static_cast<Real_t>(weights(sampleIndex, 0));
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TReference<Double_t>>::CopyInput(TMatrixT<Double_t> &matrix,
                                                                 IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &input = std::get<0>(fData);
   Int_t m = matrix.GetNrows();
   Int_t n = input.GetNcols();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator;
      for (Int_t j = 0; j < n; j++) {
         matrix(i, j) = static_cast<Double_t>(input(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TReference<Double_t>>::CopyOutput(TMatrixT<Double_t> &matrix,
                                                                  IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &output = std::get<1>(fData);
   Int_t m = matrix.GetNrows();
   Int_t n = output.GetNcols();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator;
      for (Int_t j = 0; j < n; j++) {
         matrix(i, j) = static_cast<Double_t>(output(sampleIndex, j));
      }
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<MatrixInput_t, TReference<Double_t>>::CopyWeights(TMatrixT<Double_t> &matrix,
                                                                   IndexIterator_t sampleIterator)
{
   const TMatrixT<Double_t> &output = std::get<2>(fData);
   Int_t m = matrix.GetNrows();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator;
      matrix(i, 0) = static_cast<Double_t>(output(sampleIndex, 0));
      sampleIterator++;
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TReference<Real_t>>::CopyInput(TMatrixT<Real_t> &matrix, IndexIterator_t sampleIterator)
{
   // short-circuit on empty
   if (std::get<0>(fData).empty())
      return;
   Event *event = nullptr;

   Int_t m = matrix.GetNrows();
   Int_t n = matrix.GetNcols();

   // Copy input variables.

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      for (Int_t j = 0; j < n; j++) {
         matrix(i, j) = event->GetValue(j);
      }
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TReference<Real_t>>::CopyOutput(TMatrixT<Real_t> &matrix, IndexIterator_t sampleIterator)
{
   // short-circuit on empty
   if (std::get<0>(fData).empty())
      return;
   Event *event = nullptr;
   const DataSetInfo &info = std::get<1>(fData);
   Int_t m = matrix.GetNrows();
   Int_t n = matrix.GetNcols();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      for (Int_t j = 0; j < n; j++) {
         // Classification
         if (event->GetNTargets() == 0) {
            if (n == 1) {
               // Binary.
               matrix(i, j) = (info.IsSignal(event)) ? 1.0 : 0.0;
            } else {
               // Multiclass.
               matrix(i, j) = 0.0;
               if (j == static_cast<Int_t>(event->GetClass())) {
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
void TDataLoader<TMVAInput_t, TReference<Real_t>>::CopyWeights(TMatrixT<Real_t> &matrix, IndexIterator_t sampleIterator)
{
   // short-circuit on empty
   if (std::get<0>(fData).empty())
      return;
   Event *event = nullptr;
   for (Int_t i = 0; i < matrix.GetNrows(); i++) {
      Int_t sampleIndex = *sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      matrix(i, 0) = event->GetWeight();
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TReference<Double_t>>::CopyInput(TMatrixT<Double_t> &matrix,
                                                               IndexIterator_t sampleIterator)
{
   // short-circuit on empty
   if (std::get<0>(fData).empty())
      return;
   Event *event = nullptr;
   Int_t m = matrix.GetNrows();

   // Copy input variables.

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      for (Int_t j = 0; j < static_cast<Int_t>(event ? event->GetNVariables() : 0); j++) {
         matrix(i, j) = event->GetValue(j);
      }
   }
}

//______________________________________________________________________________
template <>
void TDataLoader<TMVAInput_t, TReference<Double_t>>::CopyOutput(TMatrixT<Double_t> &matrix,
                                                                IndexIterator_t sampleIterator)
{
   Event *event = nullptr;
   const DataSetInfo &info = std::get<1>(fData);
   Int_t m = matrix.GetNrows();
   Int_t n = matrix.GetNcols();

   for (Int_t i = 0; i < m; i++) {
      Int_t sampleIndex = *sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      for (Int_t j = 0; j < n; j++) {
         // Classification
         if (event->GetNTargets() == 0) {
            if (n == 1) {
               // Binary.
               matrix(i, j) = (info.IsSignal(event)) ? 1.0 : 0.0;
            } else {
               // Multiclass.
               matrix(i, j) = 0.0;
               if (j == static_cast<Int_t>(event->GetClass())) {
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
void TDataLoader<TMVAInput_t, TReference<Double_t>>::CopyWeights(TMatrixT<Double_t> &matrix,
                                                                 IndexIterator_t sampleIterator)
{
   Event *event = nullptr;

   for (Int_t i = 0; i < matrix.GetNrows(); i++) {
      Int_t sampleIndex = *sampleIterator++;
      event = std::get<0>(fData)[sampleIndex];
      matrix(i, 0) = event->GetWeight();
   }
}

// Explicit instantiations.
template class TDataLoader<MatrixInput_t, TReference<Real_t>>;
template class TDataLoader<TMVAInput_t, TReference<Real_t>>;
template class TDataLoader<MatrixInput_t, TReference<Double_t>>;
template class TDataLoader<TMVAInput_t, TReference<Double_t>>;

} // namespace DNN
} // namespace TMVA
