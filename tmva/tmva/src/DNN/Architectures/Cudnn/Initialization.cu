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
   //PrintTensor(A,"A after init Gaus");
}

//______________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::InitializeUniform(TCudaTensor<AFloat> & A)
{
   // n is the size of the feature map
   size_t n = A.GetFirstStride();
   
   TRandom &  rand = GetRandomGenerator();

   Double_t range = sqrt(2.0 / ((Double_t) n));
   
   // range = 1; 
   // rand.SetSeed(111);

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
   // n,m are the output/input units of the tensor
   // default is caseof tensor of 2D (dense layer)
   size_t n = A.GetShape()[0];  // output size
   size_t m = A.GetShape()[1];  // input size
   // for convolutions
   if (A.GetShape().size() > 2) { 
      // n is number of inputs
      for (size_t j = 2; j < A.GetShape().size(); ++j) {
         m *= A.GetShape()[j];
         n *= A.GetShape()[j];
      }
   }

   TRandom &  rand = GetRandomGenerator();
   Double_t sigma = sqrt(2.0 /((Double_t) n + (Double_t) m) );

   //std::cout << "Initialize Glorot normal for tensor " << n <<","<<m << " ";
   //A.PrintShape();

   size_t nsize = A.GetSize(); 
   TCudaHostBuffer<AFloat> xhost(nsize);
   for (size_t i = 0; i < nsize; i++) {
      AFloat value = 0; 
         do { 
            value = rand.Gaus(0.0, sigma);
         } while ( std::abs(value) > 2*sigma);
         R__ASSERT( std::abs(value) < 2*sigma); 
         xhost[i] = value;
   }
   A.GetDeviceBuffer().CopyFrom(xhost);
 
   // if (A.GetNDim() == 2) assert( xhost[0] == A(0,0));
   // if (A.GetNDim() == 4) assert( xhost[0] == A(0,0,0,0));
}

//______________________________________________________________________________
/// Sample from a uniform distribution in range [ -lim,+lim] where
///  lim = sqrt(6/N_in+N_out).
/// This initialization is also called Xavier uniform
/// see Glorot & Bengio, AISTATS 2010 - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
template<typename AFloat>
void TCudnn<AFloat>::InitializeGlorotUniform(TCudaTensor<AFloat> & A)
{
   // n,m  are the input/output  units of the tensor
   // size_t n = 0;
   // size_t m = 0; 
   // if (A.GetNDim() > 2) { 
   //    n = A.GetFirstSize();
   //    m = A.GetCSize();
   // }
   // else {
   //    n = A.GetNrows();
   //    m = A.GetNcols(); 
   // }
   size_t n = A.GetShape()[0]; // output size
   size_t m = A.GetShape()[1]; // input size
   // for convolutions
   if (A.GetShape().size() > 2) {
      // n is number of inputs
      for (size_t j = 2; j < A.GetShape().size(); ++j) {
         m *= A.GetShape()[j];
         n *= A.GetShape()[j];
      }
   }

   TRandom &  rand = GetRandomGenerator();
   Double_t range = sqrt(6.0 /( (Double_t) n +  (Double_t) m) );

   //std::cout << "Initialize Glorot uniform for tensor " << n << "," << m << " ";
   //A.PrintShape();

   size_t nsize = A.GetSize(); 
   TCudaHostBuffer<AFloat> xhost(nsize);
   for (size_t i = 0; i < nsize; i++) {
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
