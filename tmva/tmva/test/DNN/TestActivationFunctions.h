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

// wrapper functions for calling on Matrix instead of tensors
/*  impl using Matrix */
template<typename Architecture_t>
inline void evaluate(typename Architecture_t::Matrix_t & A,
                                EActivationFunction f)
{
    typename Architecture_t::Tensor_t tA(A);
    evaluate<Architecture_t>(tA, f);
}
template<typename Architecture_t>
inline void evaluateDerivative(typename Architecture_t::Matrix_t & B,
                                EActivationFunction f,
                                const typename Architecture_t::Matrix_t & A)
{
    typename Architecture_t::Tensor_t tA(A);
    typename Architecture_t::Tensor_t tB(B);
    evaluateDerivative<Architecture_t>(tB, f, tA);
}
//________________________________________________________________________________________________
// Activation function evaluation
template <typename Architecture, typename Func>
auto testActivationFunctionEvaluation(size_t ntests, EActivationFunction afType, Func func) ->
   typename Architecture::Scalar_t
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Tensor_t = typename Architecture::Tensor_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = Architecture::GetRandomGenerator().Uniform(100) + 1;
      size_t n = Architecture::GetRandomGenerator().Uniform(100) + 1;

      TMatrixT<Scalar_t> ARef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef);

      evaluate<Architecture>(AArch, afType);
      applyMatrix(ARef, func);

      TMatrixT<Scalar_t> A = AArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }

   // test the tensor API
   Tensor_t t1(10, 100, 100);
   Tensor_t t2(10, 100, 100);
   for (size_t i = 0; i < ntests; i++) {
      size_t b = Architecture::GetRandomGenerator().Uniform(10) + 1;
      size_t m = Architecture::GetRandomGenerator().Uniform(100) + 1;
      size_t n = Architecture::GetRandomGenerator().Uniform(100) + 1;
      // m=1;
      Tensor_t ARef(t1.GetDeviceBuffer(), {b, m, n});

      randomBatch(ARef);

      Tensor_t AArch(t2.GetDeviceBuffer(), {b, m, n});
      Architecture::Copy(AArch, ARef);

      evaluate<Architecture>(AArch, afType);
      TMatrixT<Scalar_t> mRef = ARef;
      applyMatrix(mRef, func);

      TMatrixT<Scalar_t> mRes = AArch; 
      Double_t error = maximumRelativeError(mRes, mRef);
      maximumError = std::max(error, maximumError);
   }

   return maximumError;
}
//___________________________________________________________________________________
// test derivative of activation
template <typename Architecture, typename Func>
auto testActivationFunctionDerivative(size_t ntests, EActivationFunction afType, Func func) ->
   typename Architecture::Scalar_t
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Tensor_t = typename Architecture::Tensor_t;
   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = Architecture::GetRandomGenerator().Uniform(100) + 1;
      size_t n = Architecture::GetRandomGenerator().Uniform(100) + 1;

      TMatrixT<Scalar_t> ARef(m, n), BRef(m, n);
      randomMatrix(ARef);
      Matrix_t AArch(ARef), BArch(BRef);

      evaluateDerivative<Architecture>(BArch, afType, AArch);
      applyMatrix(ARef, func);

      TMatrixT<Scalar_t> A = BArch;
      Double_t error = maximumRelativeError(A, ARef);
      maximumError = std::max(error, maximumError);
   }

   // test the tensor API
   Tensor_t t1(10, 100, 100);
   Tensor_t t2(10, 100, 100);
   for (size_t i = 0; i < ntests; i++) {
      size_t b = Architecture::GetRandomGenerator().Uniform(10) + 1;
      size_t m = Architecture::GetRandomGenerator().Uniform(100) + 1;
      size_t n = Architecture::GetRandomGenerator().Uniform(100) + 1;
      // m=1;
      Tensor_t ARef(t1.GetDeviceBuffer(), {b, m, n});

      randomBatch(ARef);

      Tensor_t AArch(t2.GetDeviceBuffer(), {b, m, n});
      Architecture::Copy(AArch, ARef);
      Tensor_t BArch(AArch);  // same buffer can be used in eval Derivative

      evaluateDerivative<Architecture>(BArch, afType, AArch);
      applyTensor(ARef, func);
     
      // applyTensor(ARef, [](double x) { return x < 0.0 ? 0.0 : x; });

      TMatrixT<Scalar_t> mRes = BArch;
      TMatrixT<Scalar_t> mRef = ARef;
      Double_t error = maximumRelativeError(mRes, mRef);
      maximumError = std::max(error, maximumError);
   }

   return maximumError;
}

   //______________________________________________________________________________
   //
   //  Identity Activation Function
   //______________________________________________________________________________

   /*! Test application of identity function to matrix. */
   //______________________________________________________________________________
   template <typename Architecture>
   auto testIdentity(size_t ntests) -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Tensor_t = typename Architecture::Tensor_t;

   Double_t maximumError = 0.0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m = Architecture::GetRandomGenerator().Uniform(100) + 1;
      size_t n = Architecture::GetRandomGenerator().Uniform(100) + 1;

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
      size_t m = Architecture::GetRandomGenerator().Uniform(100) + 1;
      size_t n = Architecture::GetRandomGenerator().Uniform(100) + 1;

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
   auto func = [](double x) { return x < 0.0 ? 0.0 : x; };
   return testActivationFunctionEvaluation<Architecture>(ntests, EActivationFunction::kRelu, func);
}

/*! Test computation of the first derivative of the ReLU function. */
//______________________________________________________________________________
template <typename Architecture>
auto testReluDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   auto df = [](double x) { return x > 0.0 ? 1.0 : 0.0; };
   return testActivationFunctionDerivative<Architecture>(ntests, EActivationFunction::kRelu, df);
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
   auto func = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
   return testActivationFunctionEvaluation<Architecture>(ntests, EActivationFunction::kSigmoid, func);
}

/*! Test computation of the first derivative of the Sigmoid function. */
//______________________________________________________________________________
template <typename Architecture>
auto testSigmoidDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   auto df = [](Double_t x) {
      Double_t sig = 1.0 / (1.0 + std::exp(-x));
      return sig * (1.0 - sig);
   };
   return testActivationFunctionDerivative<Architecture>(ntests, EActivationFunction::kSigmoid, df);
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
   auto func = [](double x) { return tanh(x); };
   return testActivationFunctionEvaluation<Architecture>(ntests, EActivationFunction::kTanh, func);
}
/*! Test computation of the first derivative of the tanh function. */
//______________________________________________________________________________
template <typename Architecture>
auto testTanhDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   auto df = [](Double_t x) {
      Double_t t = tanh(x);
      return 1 - t * t;
   };
   return testActivationFunctionDerivative<Architecture>(ntests, EActivationFunction::kTanh, df);
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
   auto func =  [](double x){return fabs(x);};
   return testActivationFunctionEvaluation<Architecture>(ntests, EActivationFunction::kSymmRelu, func);
}

/*! Test computation of the first derivative of the symmetric ReLU function. */
//______________________________________________________________________________
template <typename Architecture>
auto testSymmetricReluDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   auto df =  [](Double_t x){ return (x < 0) ? -1.0 : 1.0; };
   return testActivationFunctionDerivative<Architecture>(ntests, EActivationFunction::kSymmRelu, df);
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
   auto func = [](double x){return x / (1 + fabs(x));};
   return testActivationFunctionEvaluation<Architecture>(ntests, EActivationFunction::kSoftSign, func);
}

/*! Test computation of the first derivative of the soft sign function. */
//______________________________________________________________________________
template <typename Architecture>
auto testSoftSignDerivative(size_t ntests)
-> typename Architecture::Scalar_t
{
   auto df = [](Double_t x) {
      Double_t y = 1 + fabs(x);
      return 1.0 / (y * y);
   };
   return testActivationFunctionDerivative<Architecture>(ntests, EActivationFunction::kSoftSign, df);
}

//______________________________________________________________________________
//
//  Gauss Activation Functions
//______________________________________________________________________________

/*! Test application of Gauss activation function to matrix. */
//______________________________________________________________________________
template <typename Architecture>
auto testGauss(size_t ntests) -> typename Architecture::Scalar_t
{
   auto func = [](double x) { return exp(-x * x); };
   return testActivationFunctionEvaluation<Architecture>(ntests, EActivationFunction::kGauss, func);
}

/*! Test computation of the first derivative of the Gauss activation function. */
//______________________________________________________________________________
template <typename Architecture>
auto testGaussDerivative(size_t ntests) -> typename Architecture::Scalar_t
{
   auto df = [](Double_t x) { return -2.0 * x * exp(-x * x); };
   return testActivationFunctionDerivative<Architecture>(ntests, EActivationFunction::kGauss, df);
}
#endif
