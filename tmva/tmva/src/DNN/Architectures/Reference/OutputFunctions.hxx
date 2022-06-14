// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 11/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////
// Explicit instantiation of the TReference architecture class //
// template for Double_t scalar types.                        //
////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA {
namespace DNN  {

template<typename AReal>
void TReference<AReal>::Sigmoid(TMatrixT<AReal> & B,
                                const TMatrixT<AReal> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         AReal sig = 1.0 / (1.0 + std::exp(-A(i,j)));
         B(i,j) = sig;
      }
   }
}

template<typename AReal>
void TReference<AReal>::Softmax(TMatrixT<AReal> & B,
                                const TMatrixT<AReal> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      AReal sum = 0.0;
      for (size_t j = 0; j < n; j++) {
         sum += exp(A(i,j));
      }
      for (size_t j = 0; j < n; j++) {
         B(i,j) = exp(A(i,j)) / sum;
      }
   }
}

} // namespace TMVA
} // namespace DNN
