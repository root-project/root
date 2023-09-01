// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the activation functions for the reference //
 // implementation.                                              //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"
#include <math.h>

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::IdentityDerivative(TMatrixT<Real_t> & B,
                                            const TMatrixT<Real_t> &/*A*/)
{
   size_t m,n;
   m = B.GetNrows();
   n = B.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         B(i,j) = 1.0;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Relu(TMatrixT<Real_t> &A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = std::max((Real_t) 0.0, A(i,j));
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::ReluDerivative(TMatrixT<Real_t> & B,
                                              const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         B(i,j) = (A(i,j) < 0) ? 0.0 : 1.0;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::Sigmoid(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t sig = 1.0 / (1.0 + std::exp(-A(i,j)));
         A(i,j) = sig;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::SigmoidDerivative(TMatrixT<Real_t> & B,
                                                 const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t sig = 1.0 / (1.0 + std::exp(-A(i,j)));
         B(i,j) = sig * (1.0 - sig);
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::Tanh(TMatrixT<Real_t> & B)
{
   size_t m,n;
   m = B.GetNrows();
   n = B.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t t = tanh(B(i,j));
         B(i,j) = t;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::TanhDerivative(TMatrixT<Real_t> & B,
                                              const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t t = tanh(A(i,j));
         B(i,j) = 1 - t * t;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::SymmetricRelu(TMatrixT<Real_t> & B)
{
   size_t m,n;
   m = B.GetNrows();
   n = B.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         B(i,j) = fabs(B(i,j));
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::SymmetricReluDerivative(TMatrixT<Real_t> & B,
                                                       const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         B(i,j) = (A(i,j) < 0.0) ? -1.0 : 1.0;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::SoftSign(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t x = A(i,j);
         A(i,j)   = x / (1 + fabs(x));
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::SoftSignDerivative(TMatrixT<Real_t> & B,
                                                  const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t x = 1.0 + fabs(A(i,j));
         B(i,j)   = 1.0 / (x * x);
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::Gauss(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t x = A(i,j);
         A(i,j)   = exp(- x * x);
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
inline void TReference<Real_t>::GaussDerivative(TMatrixT<Real_t> & B,
                                               const TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t x = A(i,j);
         B(i,j)   = - 2.0 * x * exp(- x * x);
      }
   }
}
} // namespace DNN
} // namespace TMVA
