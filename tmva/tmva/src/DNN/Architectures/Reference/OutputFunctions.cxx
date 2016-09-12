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

namespace TMVA {
namespace DNN  {

template<typename Real_t>
void TReference<Real_t>::Sigmoid(TMatrixT<Real_t> & B,
                                const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t sig = 1.0 / (1.0 + std::exp(-A(i,j)));
         B(i,j) = sig;
      }
   }
}

} // namespace TMVA
} // namespace DNN
