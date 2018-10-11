// @(#)root/tmva/tmva/dnn:$Id$ // Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Test arithmetic functions defined on matrices and compare the //
// results to the reference implementation.                      //
///////////////////////////////////////////////////////////////////

#include "TMatrix.h"
#include "Utility.h"
#include "math.h"
#include "TMVA/DNN/Architectures/Reference.h"

/** Test multiplication (standard, transposed, hadamard) operation on
 *  architecture specific matrix types and compare with results
 *  obtained with TMatrixT.
 */
//______________________________________________________________________________
template<typename Architecture_t>
auto testMultiplication(size_t ntests)
    -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;
   Scalar_t mean = 5.0, sigma = 2.0;

   for (size_t t = 0; t < ntests; t++) {
      size_t m, n, k;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;
      k = rand() % 100 + 1;

      TMatrixT<Scalar_t> ARef(m,k), A2Ref(m,k), ATRef(k,m) , BRef(k,n), BTRef(n,k), CRef(m,n);
      TMVA::DNN::randomMatrix(ARef, mean, sigma);
      TMVA::DNN::randomMatrix(A2Ref, mean, sigma);
      TMVA::DNN::randomMatrix(ATRef, mean, sigma);
      TMVA::DNN::randomMatrix(BRef, mean, sigma);
      TMVA::DNN::randomMatrix(BTRef, mean, sigma);
      Matrix_t A(ARef), A2(A2Ref), AT(ATRef), B(BRef), BT(BTRef),  C(CRef);

      // A * B
      CRef.Mult(ARef,BRef);
      Architecture_t::Multiply(C, A, B);
      Scalar_t error = TMVA::DNN::maximumRelativeError(C, CRef);
      maximumError   = std::max(error, maximumError);

      // A^T * B
      CRef.TMult(ATRef,BRef);
      Architecture_t::TransposeMultiply(C, AT, B);
      error = TMVA::DNN::maximumRelativeError(C, CRef);
      maximumError   = std::max(error, maximumError);

      // A * B^T
      CRef.MultT(ARef,BTRef);
      Architecture_t::MultiplyTranspose(C, A, BT);
      error = TMVA::DNN::maximumRelativeError(C, CRef);
      maximumError   = std::max(error, maximumError);

      // A .* B
      for (size_t i = 0; i < (size_t) ARef.GetNrows(); i++) {
         for (size_t j = 0; j < (size_t) ARef.GetNcols(); j++) {
            ARef(i,j) *= A2Ref(i,j);
         }
      }
      Architecture_t::Hadamard(A, A2);
      error = TMVA::DNN::maximumRelativeError(A, ARef);
      maximumError   = std::max(error, maximumError);
   }

   return maximumError;
}

/** Test the summing over columns by summing by the sums obtained
 *  from a matrix filled with column indices as elements.
 */
//______________________________________________________________________________
template<typename Architecture_t>
auto testSumColumns(size_t ntests)
    -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;
   for (size_t t = 0; t < ntests; t++) {

      Scalar_t error;

      size_t m, n;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m,n), BRef(n,1);

      for (size_t i = 0; i < (size_t) ARef.GetNrows(); i++) {
         for (size_t j = 0; j < (size_t) ARef.GetNcols(); j++) {
            ARef(i,j) = j;
            if (i == 0) BRef(j, 0) = m * j;
         }
      }

      Matrix_t A(ARef), B(n, 1);
      Architecture_t::SumColumns(B, A);

      error = TMVA::DNN::maximumRelativeError((TMatrixT<Scalar_t>) B ,BRef);
      maximumError   = std::max(error, maximumError);
   }
   return maximumError;
}

/** Test the addition of a constant to every element of
 *  the given matrix.
 */
//______________________________________________________________________________
template <typename Architecture_t>
auto testConstAdd(size_t ntests) -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;
   Scalar_t beta;

   for (size_t t = 0; t < ntests; t++) {

      Scalar_t error;

      size_t m, n;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;

      beta = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      TMVA::DNN::randomMatrix(ARef);

      Matrix_t A(ARef);
      Architecture_t::ConstAdd(A, beta);

      for (size_t i = 0; i < (size_t)ARef.GetNrows(); i++) {
         for (size_t j = 0; j < (size_t)ARef.GetNcols(); j++) {
            ARef(i, j) += (Double_t)beta;
         }
      }

      error = TMVA::DNN::maximumRelativeError((TMatrixT<Scalar_t>)A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/** Test the multiplication of a constant to every element of
 *  the given matrix.
 */
//______________________________________________________________________________
template <typename Architecture_t>
auto testConstMult(size_t ntests) -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;
   Scalar_t beta;

   for (size_t t = 0; t < ntests; t++) {

      Scalar_t error;

      size_t m, n;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;

      beta = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      TMVA::DNN::randomMatrix(ARef);

      Matrix_t A(ARef);
      Architecture_t::ConstMult(A, beta);

      for (size_t i = 0; i < (size_t)ARef.GetNrows(); i++) {
         for (size_t j = 0; j < (size_t)ARef.GetNcols(); j++) {
            ARef(i, j) *= (Double_t)beta;
         }
      }

      error = TMVA::DNN::maximumRelativeError((TMatrixT<Scalar_t>)A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/** Test the application of reciprocating every element of
 *  the given matrix.
 */
//______________________________________________________________________________
template <typename Architecture_t>
auto testReciprocalElementWise(size_t ntests) -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;

   for (size_t t = 0; t < ntests; t++) {

      Scalar_t error;

      size_t m, n;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      TMVA::DNN::randomMatrix(ARef);

      Matrix_t A(ARef);
      Architecture_t::ReciprocalElementWise(A);

      TMVA::DNN::applyMatrix(ARef, [](double x) { return 1.0 / x; });

      error = TMVA::DNN::maximumRelativeError((TMatrixT<Scalar_t>)A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/** Test the application of squaring every element of
 *  the given matrix.
 */
//______________________________________________________________________________
template <typename Architecture_t>
auto testSquareElementWise(size_t ntests) -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;

   for (size_t t = 0; t < ntests; t++) {

      Scalar_t error;

      size_t m, n;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      TMVA::DNN::randomMatrix(ARef);

      Matrix_t A(ARef);
      Architecture_t::SquareElementWise(A);

      TMVA::DNN::applyMatrix(ARef, [](double x) { return x * x; });

      error = TMVA::DNN::maximumRelativeError((TMatrixT<Scalar_t>)A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

/** Test the application of taking square root of every element of
 *  the given matrix.
 */
//______________________________________________________________________________
template <typename Architecture_t>
auto testSqrtElementWise(size_t ntests) -> typename Architecture_t::Scalar_t
{

   using Scalar_t = typename Architecture_t::Scalar_t;
   using Matrix_t = typename Architecture_t::Matrix_t;

   Scalar_t maximumError = 0.0;
   Scalar_t mean = 5.0, sigma = 2.0;

   for (size_t t = 0; t < ntests; t++) {

      Scalar_t error;

      size_t m, n;
      m = rand() % 100 + 1;
      n = rand() % 100 + 1;

      TMatrixT<Double_t> ARef(m, n);
      TMVA::DNN::randomMatrix(ARef, mean, sigma);

      Matrix_t A(ARef);
      Architecture_t::SqrtElementWise(A);

      TMVA::DNN::applyMatrix(ARef, [](double x) { return sqrt(x); });

      error = TMVA::DNN::maximumRelativeError((TMatrixT<Scalar_t>)A, ARef);
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}
