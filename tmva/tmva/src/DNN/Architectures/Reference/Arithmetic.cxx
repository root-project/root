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

} // namespace DNN
} // namespace TMVA
