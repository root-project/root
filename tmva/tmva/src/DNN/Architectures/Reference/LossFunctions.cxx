// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 ////////////////////////////////////////////////////////////
 // Implementation of the loss functions for the reference //
 // implementation.                                        //
 ////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{
//______________________________________________________________________________
template<typename Real_t>
Real_t TReference<Real_t>::MeanSquaredError(const TMatrixT<Real_t> &Y,
                                           const TMatrixT<Real_t> &output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   Real_t result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t dY = (Y(i,j) - output(i,j));
         result += dY * dY;
      }
   }
   result /= (Real_t) (m * n);
   return result;
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::MeanSquaredErrorGradients(TMatrixT<Real_t> & dY,
                                                  const TMatrixT<Real_t> & Y,
                                                  const TMatrixT<Real_t> & output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();

   dY.Minus(Y, output);
   dY *= - 2.0 / ((Real_t) (m*n));
}

//______________________________________________________________________________
template<typename Real_t>
Real_t TReference<Real_t>::CrossEntropy(const TMatrixT<Real_t> &Y,
                                       const TMatrixT<Real_t> &output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   Real_t result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t sig = 1.0 / (1.0 + std::exp(-output(i,j)));
         result      += Y(i,j) * std::log(sig)
         + (1.0 - Y(i,j)) * std::log(1.0 - sig);
      }
   }
   result /= - (Real_t) (m * n);
   return result;
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::CrossEntropyGradients(TMatrixT<Real_t> & dY,
                                              const TMatrixT<Real_t> & Y,
                                              const TMatrixT<Real_t> & output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();

   Real_t norm = 1.0 / ((Real_t) (m * n));
   for (size_t i = 0; i < m; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         Real_t y   = Y(i,j);
         Real_t sig = 1.0 / (1.0 + std::exp(-output(i,j)));
         dY(i,j) = norm * (sig - y);
      }
   }
}

} // namespace DNN
} // namespace TMVA
