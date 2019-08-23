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

#include "TRandom3.h"

#include "TMVA/DNN/Architectures/TCudnn.h"
#include "TMVA/DNN/Architectures/Cuda/CudaTensor.h"



namespace TMVA
{
namespace DNN
{

template <typename AFloat>
TRandom * TCudnn<AFloat>::fgRandomGen = nullptr;
//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::SetRandomSeed(size_t seed)
{
   if (!fgRandomGen) fgRandomGen = new TRandom3();
   fgRandomGen->SetSeed(seed); 
}
template<typename AFloat>
TRandom & TCudnn<AFloat>::GetRandomGenerator()
{
   if (!fgRandomGen) fgRandomGen = new TRandom3(0);
   return *fgRandomGen; 
}
//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::InitializeGauss(TCudaTensor<AFloat> & A)
{
   // n is the size of the feature map
   size_t n = A.GetFirstStride();
   
   TRandom &  rand = GetRandomGenerator();

   Double_t sigma = sqrt(2.0 / ((Double_t) n));

   size_t nelements = A.GetSize(); 
   TCudaHostBuffer<AFloat> xhost(nelements);
   for (size_t i = 0; i < nelements; i++) {
      xhost[i] = rand.Gaus(0,sigma); 
   }
   A.GetDeviceBuffer().CopyFrom(xhost);
   PrintTensor(A,"A after init Gaus");
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::InitializeUniform(TCudaTensor<AFloat> & A)
{
   // n is the size of the feature map
   size_t n = A.GetFirstStride();
   
   TRandom &  rand = GetRandomGenerator();

   Double_t range = sqrt(2.0 / ((Double_t) n));

   size_t nelements = A.GetSize(); 
   TCudaHostBuffer<AFloat> xhost(nelements);
   for (size_t i = 0; i < nelements; i++) {
      xhost[i] = rand.Uniform(-range, range);
   }
   A.GetDeviceBuffer().CopyFrom(xhost);
  
}

//______________________________________________________________________________
///  Truncated normal initialization (Glorot, called also Xavier normal)
///  The values are sample with a normal distribution with stddev = sqrt(2/N_input + N_output) and
///   values larger than 2 * stddev are discarded 
///  See Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
template<typename AFloat>
void TCudnn<AFloat>::InitializeGlorotNormal(TCudaTensor<AFloat> & A)
{
   // n is the size of the tensor
   size_t n = A.GetSize();

   TRandom &  rand = GetRandomGenerator();
   Double_t sigma = sqrt(2.0 / ((Double_t) n));  

   TCudaHostBuffer<AFloat> xhost(n);
   for (size_t i = 0; i < n; i++) {
      AFloat value =  rand.Gaus(0.0, sigma); 
      if ( std::abs(value) > 2*sigma) continue; 
      xhost[i] = rand.Gaus(0, sigma);
   }
   A.GetDeviceBuffer().CopyFrom(xhost); 
}

//______________________________________________________________________________
/// Sample from a uniform distribution in range [ -lim,+lim] where
///  lim = sqrt(6/N_in+N_out).
/// This initialization is also called Xavier uniform
/// see Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
template<typename AFloat>
void TCudnn<AFloat>::InitializeGlorotUniform(TCudaTensor<AFloat> & A)
{\
   // n is the size of the tensor
   size_t n = A.GetSize();

   TRandom &  rand = GetRandomGenerator();
   Double_t range = sqrt(6.0 /( (Double_t) n) );

   TCudaHostBuffer<AFloat> xhost(n);
   for (size_t i = 0; i < n; i++) {
      xhost[i] = rand.Uniform(-range, range);
   }
   A.GetDeviceBuffer().CopyFrom(xhost); 
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::InitializeIdentity(TCudaTensor<AFloat> & A)
{
   size_t m,n;
   m = A.GetFirstSize();
   n = A.GetFirstStride();
   // assume weight trnsor is like a matrix M x N 
   TMatrixT<AFloat> B(m, n);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n ; j++) {
         B(i,j) = 0.0;
      }
      if (i < n) {
         B(i,i) = 1.0;
      }
   }
   TCudaMatrix<AFloat> mB = B; 
   A.GetDeviceBuffer() = mB.GetDeviceBuffer(); 
   PrintTensor(A,"A after init Identity");
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::InitializeZero(TCudaTensor<AFloat> & A)
{
   // use fast zero initialization on the device
   A.Zero();
}

} // namespace DNN
} // namespace TMVA
