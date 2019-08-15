// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Generic tests of the layer activation functions                  //
//                                                                  //
// Contains tests for each of the layer activation functions that   //
// test the evaluation of the function using the evaluate(...)      //
// method and the computation of the derivatives using              //
// evaluate_derivative(...) on a randomly generated matrix. Each    //
// function returns the maximum relative error between the expected //
// result and the result obtained for the given arcthitecture.      //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_ACTIVATION_FUNCTIONS
#define TMVA_TEST_DNN_TEST_ACTIVATION_FUNCTIONS

#include "TMatrixT.h"
//#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Net.h"
#include "Utility.h"

using namespace TMVA::DNN;

//______________________________________________________________________________
//
//  Identity Activation Function
//______________________________________________________________________________

/*! Test application of identity function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testIdentity(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Tensor_t = typename Architecture::Tensor_t;

   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);
      Tensor_t tAArch(AArch);

      evaluate<Architecture>(tAArch, EActivationFunction::kIdentity);

      TMatrixT<Double_t> A = tAArch.GetMatrix();
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

#if 0  // fix to not use reference architecture
/*! Test computation of the first derivative of the identity function. */
//______________________________________________________________________________
template <typename Architecture>
auto testIdentityDerivative(size_t ntests)
    -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Tensor_t = typename Architecture::Tensor_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kIdentity, AArch);
      
      evaluateDerivative<TReference<Double_t>>(BRef, EActivationFunction::kIdentity,
                                                ARef);

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, BRef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}
#endif
//______________________________________________________________________________
//
//  ReLU Activation Function
//______________________________________________________________________________

/*! Test application of ReLU function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testRelu(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, EActivationFunction::kRelu);
      applyMatrix(ARef, [](double x){return x < 0.0 ? 0.0 : x;});

      TMatrixT<Double_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/*! Test computation of the first derivative of the ReLU function. */
//______________________________________________________________________________
template <typename Architecture>
auto testReluDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kRelu, AArch);
      applyMatrix(ARef, [](double x){return x > 0.0 ? 1.0 : 0.0;});

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

//______________________________________________________________________________
//
//  Sigmoid Activation Function
//______________________________________________________________________________

/*! Test application of Sigmoid function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testSigmoid(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, EActivationFunction::kSigmoid);
      applyMatrix(ARef, [](double x){return 1.0 / (1.0 + std::exp(-x));});

      TMatrixT<Double_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/*! Test computation of the first derivative of the ReLU function. */
//______________________________________________________________________________
template <typename Architecture>
auto testSigmoidDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kSigmoid, AArch);
      applyMatrix(ARef, [](Double_t x){
             Double_t sig = 1.0 / (1.0 + std::exp(-x));
             return sig * (1.0 - sig);
          });

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

//______________________________________________________________________________
//
//  Tanh Activation Function
//______________________________________________________________________________

/*! Test application of tanh function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testTanh(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, EActivationFunction::kTanh);
      applyMatrix(ARef, [](double x){return tanh(x);});

      TMatrixT<Double_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/*! Test computation of the first derivative of the tanh function. */
//______________________________________________________________________________
template <typename Architecture>
auto testTanhDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kTanh, AArch);
      applyMatrix(ARef, [](Double_t x){
             Double_t t = tanh(x);
             return 1 - t * t;
          });

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

//______________________________________________________________________________
//
//  Symmetric ReLU Activation Function
//______________________________________________________________________________

/*! Test application of symmetric ReLU function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testSymmetricRelu(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, EActivationFunction::kSymmRelu);
      applyMatrix(ARef, [](double x){return fabs(x);});

      TMatrixT<Double_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/*! Test computation of the first derivative of the symmetric ReLU function. */
//______________________________________________________________________________
template <typename Architecture>
auto testSymmetricReluDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kSymmRelu, AArch);
      applyMatrix(ARef, [](Double_t x){
             return (x < 0) ? -1.0 : 1.0;
          });

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

//______________________________________________________________________________
//
//  Soft Sign Activation Function
//______________________________________________________________________________

/*! Test application of symmetric soft sign function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testSoftSign(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, EActivationFunction::kSoftSign);
      applyMatrix(ARef, [](double x){return x / (1 + fabs(x));});

      TMatrixT<Double_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/*! Test computation of the first derivative of the soft sign function. */
//______________________________________________________________________________
template <typename Architecture>
auto testSoftSignDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kSoftSign, AArch);
      applyMatrix(ARef, [](Double_t x){
             Double_t y = 1 + fabs(x);
             return 1.0 / (y * y);
          });

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

//______________________________________________________________________________
//
//  Gauss Activation Functions
//______________________________________________________________________________

/*! Test application of Gauss activation function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testGauss(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, EActivationFunction::kGauss);
      applyMatrix(ARef, [](double x){return exp(- x * x);});

      TMatrixT<Double_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/*! Test computation of the first derivative of the Gauss activation function. */
//______________________________________________________________________________
template <typename Architecture>
auto testGaussDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = rand() % 100 + 1;
      size_t n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, EActivationFunction::kGauss, AArch);
      applyMatrix(ARef, [](Double_t x){return -2.0 * x * exp(- x * x);});

      TMatrixT<Double_t> B = BArch;
      Double_t error = maximumRelativeError(B, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}
#endif
