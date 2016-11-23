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
template<typename AFloat>
void TCpu<AFloat>::InitializeGauss(TCpuMatrix<AFloat> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom rand(time(nullptr));

   AFloat sigma = sqrt(2.0 / ((AFloat) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = rand.Gaus(0.0, sigma);
      }
   }
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::InitializeUniform(TCpuMatrix<AFloat> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom rand(time(nullptr));

   AFloat range = sqrt(2.0 / ((AFloat) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = rand.Uniform(-range, range);
      }
   }
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::InitializeIdentity(TCpuMatrix<AFloat> & A)
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
template<typename AFloat>
void TCpu<AFloat>::InitializeZero(TCpuMatrix<AFloat> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n ; j++) {
         A(i,j) = 0.0;
      }
   }
}

} // namespace DNN
} // namespace TMVA
