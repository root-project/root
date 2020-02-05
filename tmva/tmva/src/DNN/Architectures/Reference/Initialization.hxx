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

#include "TRandom3.h"
#include "TMVA/DNN/Architectures/Reference.h"

namespace TMVA
{
namespace DNN
{

template <typename Real_t>
TRandom * TReference<Real_t>::fgRandomGen = nullptr;
//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::SetRandomSeed(size_t seed)
{
   if (!fgRandomGen) fgRandomGen = new TRandom3();
   fgRandomGen->SetSeed(seed); 
}
template<typename Real_t>
TRandom & TReference<Real_t>::GetRandomGenerator()
{
   if (!fgRandomGen) fgRandomGen = new TRandom3(0);
   return *fgRandomGen; 
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::InitializeGauss(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom &  rand = GetRandomGenerator();
 
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

   TRandom &  rand = GetRandomGenerator();

   Real_t range = sqrt(2.0 / ((Real_t) n));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         A(i,j) = rand.Uniform(-range, range);
      }
   }
}

 //______________________________________________________________________________
///  Truncated normal initialization (Glorot, called also Xavier normal)
///  The values are sample with a normal distribution with stddev = sqrt(2/N_input + N_output) and
///   values larger than 2 * stddev are discarded 
///  See Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
template<typename Real_t>
void TReference<Real_t>::InitializeGlorotNormal(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom &  rand = GetRandomGenerator();

   Real_t sigma = sqrt(2.0 /( ((Real_t) n) + ((Real_t) m)) );

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t value = rand.Gaus(0.0, sigma);
         if ( std::abs(value) > 2*sigma) continue; 
         A(i,j) = rand.Gaus(0.0, sigma);
      }
   }
}

//______________________________________________________________________________
/// Sample from a uniform distribution in range [ -lim,+lim] where
///  lim = sqrt(6/N_in+N_out).
/// This initialization is also called Xavier uniform
/// see Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
template<typename Real_t>
void TReference<Real_t>::InitializeGlorotUniform(TMatrixT<Real_t> & A)
{
   size_t m,n;
   m = A.GetNrows();
   n = A.GetNcols();

   TRandom &  rand = GetRandomGenerator();

   Real_t range = sqrt(6.0 /( ((Real_t) n) + ((Real_t) m)) );

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
      for (size_t j = 0; j < n ; j++) {
         A(i,j) = 0.0;
      }

      if (i < n) {
         A(i,i) = 1.0;
      }
   }
}

//______________________________________________________________________________
template<typename Real_t>
void TReference<Real_t>::InitializeZero(TMatrixT<Real_t> & A)
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
