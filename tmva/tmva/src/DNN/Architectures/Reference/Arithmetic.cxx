// @(#)root/tmva/tmva/dnn:$Id$
// Author: Ravi Kiran S

/*************************************************************************
 * Copyright (C) 2018, Ravi Kiran S                                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Implementation of the Helper arithmetic functions for the    //
// reference implementation.                                    //
//////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"
#include <math.h>

namespace TMVA {
namespace DNN {

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::SumColumns(TMatrixT<AReal> &B, const TMatrixT<AReal> &A)
{
   B = 0.0;
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         B(0, j) += A(i, j);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::Hadamard(TMatrixT<AReal> &A, const TMatrixT<AReal> &B)
{
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         A(i, j) *= B(i, j);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::ConstAdd(TMatrixT<AReal> &A, AReal beta)
{
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         A(i, j) += beta;
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::ConstMult(TMatrixT<AReal> &A, AReal beta)
{
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         A(i, j) *= beta;
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::ReciprocalElementWise(TMatrixT<AReal> &A)
{
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         A(i, j) = 1.0 / A(i, j);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::SquareElementWise(TMatrixT<AReal> &A)
{
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         A(i, j) *= A(i, j);
      }
   }
}

//______________________________________________________________________________
template <typename AReal>
void TReference<AReal>::SqrtElementWise(TMatrixT<AReal> &A)
{
   for (Int_t i = 0; i < A.GetNrows(); i++) {
      for (Int_t j = 0; j < A.GetNcols(); j++) {
         A(i, j) = sqrt(A(i, j));
      }
   }
}
/// Adam updates 
//____________________________________________________________________________
template<typename AReal>
void TReference<AReal>::AdamUpdate(TMatrixT<AReal> &A, const TMatrixT<AReal> & M, const TMatrixT<AReal> & V, AReal alpha, AReal eps)
{
   // ADAM update the weights.
   // Weight = Weight - alpha * M / (sqrt(V) + epsilon)
   AReal * a = A.GetMatrixArray();
   const AReal * m = M.GetMatrixArray();
   const AReal * v = V.GetMatrixArray();
   for (int index = 0; index < A.GetNoElements() ; ++index) {
      a[index] = a[index] - alpha * m[index]/( sqrt(v[index]) + eps);
   }
}

//____________________________________________________________________________
template<typename AReal>
void TReference<AReal>::AdamUpdateFirstMom(TMatrixT<AReal> &A, const TMatrixT<AReal> & B, AReal beta)
{
   // First momentum weight gradient update for ADAM 
   // Mt = beta1 * Mt-1 + (1-beta1) * WeightGradients
   AReal * a = A.GetMatrixArray();
   const AReal * b = B.GetMatrixArray();
   for (int index = 0; index < A.GetNoElements() ; ++index) {
      a[index] = beta * a[index] + (1.-beta) * b[index];
   }
}   
//____________________________________________________________________________
template<typename AReal>
void TReference<AReal>::AdamUpdateSecondMom(TMatrixT<AReal> &A, const TMatrixT<AReal> & B, AReal beta)
{
   // Second  momentum weight gradient update for ADAM 
   // Vt = beta2 * Vt-1 + (1-beta2) * WeightGradients^2
   AReal * a = A.GetMatrixArray();
   const AReal * b = B.GetMatrixArray();
   for (int index = 0; index < A.GetNoElements() ; ++index) {
      a[index] = beta * a[index] + (1.-beta) * b[index] * b[index];
   }
}
   
} // namespace DNN
} // namespace TMVA
