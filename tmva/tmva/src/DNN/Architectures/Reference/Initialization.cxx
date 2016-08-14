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
 // Implementation of the initialization functions for the reference //
 // implementation.                                                  //
 //////////////////////////////////////////////////////////////////////

#include "TRandom.h"
#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::InitializeGauss(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom rand(time(nullptr));

   Real_t sigma = sqrt(2.0 / ((Real_t) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = rand.Gaus(0.0, sigma);
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::InitializeUniform(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom rand(time(nullptr));

   Real_t range = sqrt(2.0 / ((Real_t) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = rand.Uniform(-range, range);
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::InitializeIdentity(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = 0.0;
      }

      if (i < n) {
         A(i,i) = 1.0;
      }
   }
}

template<typename Real_t>
void TReference<Real_t>::InitializeZero(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m ; i++) {
      for (size_t j = 0; j < n ; j++) {
         A(i,j) = 0.0;
      }
   }
}

} // namespace DNN
} // namespace TMVA
