// Authors: Monica Dessole   Jan 2024

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TMatrixD.h>
#include <TMatrixDSym.h>
#include <TMatrixF.h>
#include <TMatrixDEigen.h>

#include <ROOT/TestSupport.hxx>
#include <gtest/gtest.h>

#include <iostream>

double tol = std::numeric_limits<double>::epsilon() * 100;
float tol_f = std::numeric_limits<float>::epsilon() * 100;

// Helper functions for element-wise comparison of TMatrix.
#define CHECK_TMATRIX_FLOAT(a, b, m, n)                                                     \
   {                                                                                        \
      for (Int_t i = 0; i < m; i++) {                                                       \
         for (Int_t j = 0; j < n; j++) {                                                    \
            EXPECT_NEAR(a(i, j), b(i, j), tol_f) << "  at entry (" << i << "," << j << ")"; \
         }                                                                                  \
      }                                                                                     \
   }

#define CHECK_TMATRIX_DOUBLE(a, b, m, n)                                                  \
   {                                                                                      \
      for (Int_t i = 0; i < m; i++) {                                                     \
         for (Int_t j = 0; j < n; j++) {                                                  \
            EXPECT_NEAR(a(i, j), b(i, j), tol) << "  at entry (" << i << "," << j << ")"; \
         }                                                                                \
      }                                                                                   \
   }

void CompareTMatrix(TMatrixD result, TMatrixD expected)
{
   CHECK_TMATRIX_DOUBLE(result, expected, result.GetNrows(), result.GetNcols())
}

void CompareTMatrix(TMatrixF result,
                    TMatrixF expected){CHECK_TMATRIX_FLOAT(result, expected, result.GetNrows(), result.GetNcols())}

Int_t n = 5;
double values[5] = {-0.26984126984127, -0.1375661375661377, -0.0052910052910052, 0.1269841269841271,
                    0.2592592592592592};

template <typename MTX>
class testMatrix : public testing::Test {
protected:
   void SetUp() override
   {
      m1.ResizeTo(n, n);
      m2.ResizeTo(n, n * 2);
      eye.ResizeTo(n, n);
      eye.UnitMatrix();

      for (int i = 0; i < n; i++)
         for (int j = 0; j < n; j++)
            m1(i, j) = n * i + j;

      for (int i = 0; i < n; i++)
         for (int j = 0; j < 2 * n; j++)
            m2(i, j) = 1;
   }

   MTX m1;
   MTX m2;
   MTX eye;
};

using MyTypes = ::testing::Types<TMatrixF, TMatrixD>;
TYPED_TEST_SUITE(testMatrix, MyTypes);

TYPED_TEST(testMatrix, kPlus)
{
   TypeParam b(n, n);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j + 1 * (i == j);

   TypeParam c(TestFixture::m1, TypeParam::kPlus, TestFixture::eye);

   EXPECT_EQ(c, b);
}

TYPED_TEST(testMatrix, kMinus)
{
   TypeParam b(n, n);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j - 1 * (i == j);

   TypeParam c(TestFixture::m1, TypeParam::kMinus, TestFixture::eye);

   EXPECT_EQ(c, b);
}

TYPED_TEST(testMatrix, kMult)
{

   TypeParam b(n, 2 * n);
   double sum;

   for (int i = 0; i < n; i++) {
      sum = 0.0;
      for (int j = 0; j < n; j++)
         sum += n * i + j;
      for (int j = 0; j < 2 * n; j++)
         b(i, j) = sum;
   }

   TypeParam c(TestFixture::m1, TypeParam::kMult, TestFixture::m2);

   EXPECT_EQ(c, b);
}

TYPED_TEST(testMatrix, kInvMult)
{

   TypeParam b(n, 2 * n);
   double sum;

   for (int i = 0; i < n; i++) {
      sum = values[i];
      for (int j = 0; j < 2 * n; j++)
         b(i, j) = sum;
   }

   TestFixture::m1 += TestFixture::eye;

   TypeParam c(TestFixture::m1, TypeParam::kInvMult, TestFixture::m2);

   CompareTMatrix(c, b);
}

TYPED_TEST(testMatrix, Invert)
{
   TypeParam b(n, n);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j + 1 * (i == j);

   b.Invert();

   TestFixture::m1 += TestFixture::eye;

   TypeParam c(TestFixture::m1, TypeParam::kMult, b);

   CompareTMatrix(c, TestFixture::eye);
}

TYPED_TEST(testMatrix, SetElement)
{
   TypeParam b(n, n);
   TypeParam c(n, n);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j + 1 * (i == j);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         c.SetElement(i, j, n * i + j + 1 * (i == j));

   CompareTMatrix(b, c);
}

class testMatrixD : public testing::Test {
protected:
   void SetUp() override
   {
      A.ResizeTo(4, 4);

      A(0, 0) = 0.9901;
      A(0, 1) = -0.04032;
      A(0, 2) = -0.0669;
      A(0, 3) = -0.1163;
      A(1, 0) = 0.02073;
      A(1, 1) = 0.9891;
      A(1, 2) = -0.0546;
      A(1, 3) = -0.1351;
      A(2, 0) = 0.05771;
      A(2, 1) = 0.04076;
      A(2, 2) = 0.9931;
      A(2, 3) = -0.09409;
      A(3, 0) = 0.1259;
      A(3, 1) = 0.1355;
      A(3, 2) = 0.07992;
      A(3, 3) = 0.9795;
   }

   TMatrixD A;
};

TEST_F(testMatrixD, Eigen)
{
   TMatrixDEigen DE(A);
   TMatrixD B = DE.GetEigenValues();
   TMatrixD C(4, 4);
   C(0, 0) = 1.0000211760782869;
   C(1, 1) = 0.9759052775448657;
   C(2, 2) = 0.9759052775448657;
   C(3, 3) = 0.9999682688319808;
   C(1, 2) = 0.2182389465192604;
   C(2, 1) = -0.2182389465192604;

   CompareTMatrix(B, C);
}

TEST(TMatrixTSym, IsSymmetric)
{
   using ROOT::TestSupport::CheckDiagsRAII;
   Double_t edata[] = {1, 2, 3, 4};
   CheckDiagsRAII diags;
   diags.requiredDiag(kError, "SetMatrixArray", "Matrix is not symmetric after Set");
   TMatrixDSym m(2, edata);
}
