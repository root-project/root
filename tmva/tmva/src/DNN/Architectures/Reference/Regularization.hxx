// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////////
 // Implementation of the regularization functions for the reference //
 // implementation.                                                  //
 //////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t>
Real_t TReference<Real_t>::L1Regularization(const TMatrixT<Real_t> & W)
{
   size_t m,n;
   m = W.GetNrows();
   n = W.GetNcols();

   Real_t result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         result += std::abs(W(i,j));
      }
   }
   return result;
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::AddL1RegularizationGradients(TMatrixT<Real_t> & A,
                                                     const TMatrixT<Real_t> & W,
                                                     Real_t weightDecay)
{
   size_t m,n;
   m = W.GetNrows();
   n = W.GetNcols();

   Real_t sign = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         sign = (W(i,j) > 0.0) ? 1.0 : -1.0;
         A(i,j) += sign * weightDecay;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
Real_t TReference<Real_t>::L2Regularization(const TMatrixT<Real_t> & W)
{
   size_t m,n;
   m = W.GetNrows();
   n = W.GetNcols();

   Real_t result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         result += W(i,j) * W(i,j);
      }
   }
   return result;
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::AddL2RegularizationGradients(TMatrixT<Real_t> & A,
                                                     const TMatrixT<Real_t> & W,
                                                     Real_t weightDecay)
{
   size_t m,n;
   m = W.GetNrows();
   n = W.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) += weightDecay * 2.0 * W(i,j);
      }
   }
}

} // namespace DNN
} // namespace TMVA
