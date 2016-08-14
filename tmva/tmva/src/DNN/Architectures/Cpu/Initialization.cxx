// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////
 // Implementation of the DNN initialization methods for the //
 // multi-threaded CPU backend.                              //
 //////////////////////////////////////////////////////////////

#include "TRandom.h"
#include "TMVA/DNN/Architectures/Cpu.h"

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::InitializeGauss(TCpuMatrix<Real_t> & A)
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
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::InitializeUniform(TCpuMatrix<Real_t> & A)
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
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::InitializeIdentity(TCpuMatrix<Real_t> & A)
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
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::InitializeZero(TCpuMatrix<Real_t> & A)
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
