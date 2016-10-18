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
template<typename AReal>
AReal TReference<AReal>::MeanSquaredError(const TMatrixT<AReal> &Y,
                                           const TMatrixT<AReal> &output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         AReal dY = (Y(i,j) - output(i,j));
         result += dY * dY;
      }
   }
   result /= (AReal) (m * n);
   return result;
}

//______________________________________________________________________________
template<typename AReal>
void TReference<AReal>::MeanSquaredErrorGradients(TMatrixT<AReal> & dY,
                                                  const TMatrixT<AReal> & Y,
                                                  const TMatrixT<AReal> & output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();

   dY.Minus(Y, output);
   dY *= - 2.0 / ((AReal) (m*n));
}

//______________________________________________________________________________
template<typename AReal>
AReal TReference<AReal>::CrossEntropy(const TMatrixT<AReal> &Y,
                                       const TMatrixT<AReal> &output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal result = 0.0;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         AReal sig = 1.0 / (1.0 + std::exp(-output(i,j)));
         result      += Y(i,j) * std::log(sig)
         + (1.0 - Y(i,j)) * std::log(1.0 - sig);
      }
   }
   result /= - (AReal) (m * n);
   return result;
}

//______________________________________________________________________________
template<typename AReal>
void TReference<AReal>::CrossEntropyGradients(TMatrixT<AReal> & dY,
                                              const TMatrixT<AReal> & Y,
                                              const TMatrixT<AReal> & output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();

   AReal norm = 1.0 / ((AReal) (m * n));
   for (size_t i = 0; i < m; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         AReal y   = Y(i,j);
         AReal sig = 1.0 / (1.0 + std::exp(-output(i,j)));
         dY(i,j) = norm * (sig - y);
      }
   }
}

//______________________________________________________________________________
template<typename AReal>
AReal TReference<AReal>::SoftmaxCrossEntropy(const TMatrixT<AReal> &Y,
                                               const TMatrixT<AReal> &output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal result = 0.0;

   for (size_t i = 0; i < m; i++) {
      AReal sum = 0.0;
      for (size_t j = 0; j < n; j++) {
         sum += exp(output(i,j));
      }
      for (size_t j = 0; j < n; j++) {
         result += Y(i,j) * log(exp(output(i,j)) / sum);
      }
   }
   result /= - m;
   return result;
}

//______________________________________________________________________________
template<typename AReal>
void TReference<AReal>::SoftmaxCrossEntropyGradients(TMatrixT<AReal> & dY,
                                                      const TMatrixT<AReal> & Y,
                                                      const TMatrixT<AReal> & output)
{
   size_t m,n;
   m = Y.GetNrows();
   n = Y.GetNcols();
   AReal norm = 1.0 / m ;

   for (size_t i = 0; i < m; i++)
   {
      AReal sum  = 0.0;
      AReal sumY = 0.0;
      for (size_t j = 0; j < n; j++) {
         sum  += exp(output(i,j));
         sumY += Y(i,j);
      }
      for (size_t j = 0; j < n; j++) {
         dY(i,j) = norm * (exp(output(i,j)) / sum * sumY - Y(i,j));
      }
   }
}

} // namespace DNN
} // namespace TMVA
