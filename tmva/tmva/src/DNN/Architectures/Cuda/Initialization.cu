// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 14/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 /////////////////////////////////////////////////////////////
 // Implementation of the initialization functions for CUDA //
 // Architectures                                           //
 /////////////////////////////////////////////////////////////

#include "TRandom.h"
#include "TMVA/DNN/Architectures/Cuda.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
void TCuda::InitializeGauss(TCudaMatrix & A)
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
void TCuda::InitializeUniform(TCudaMatrix & A)
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
void TCuda::InitializeIdentity(TCudaMatrix & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n ; j++) {
         A(i,j) = 0.0;
      }

      if (i < n) {
         A(i,i) = 1.0;
      }
   }
}

//______________________________________________________________________________
void TCuda::InitializeZero(TCudaMatrix & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m * n; i++) {
      for (size_t j = 0; j < n ; j++) {
         A(i,j) = 0.0;
      }
   }
}

} // namespace DNN
} // namespace TMVA
