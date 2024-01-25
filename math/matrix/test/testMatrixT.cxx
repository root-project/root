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
#include <TMatrixFSym.h>
//#include <Math.h>

#include <gtest/gtest.h>

#include <iostream>

typedef TMatrixF MTX;
typedef float Scalar;
double tol = std::numeric_limits<double>::epsilon() * 100;
float tol_f = std::numeric_limits<float>::epsilon() * 100;

// Helper functions for element-wise comparison of TMatrix.

#define CHECK_TMATRIX_FLOAT(a, b, m, n)                                                     \
   {                                                                                        \
      for (size_t i = 0; i < m; i++) {                                                      \
         for (size_t j = 0; j < n; j++) {                                                   \
            EXPECT_NEAR(a(i, j), b(i, j), tol_f) << "  at entry (" << i << "," << j << ")"; \
         }                                                                                  \
      }                                                                                     \
   }

#define CHECK_TMATRIX_DOUBLE(a, b, m, n)                                                  \
   {                                                                                      \
      for (size_t i = 0; i < m; i++) {                                                    \
         for (size_t j = 0; j < n; j++) {                                                 \
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
Scalar values[5] = {-0.26984126984127, -0.1375661375661377, -0.0052910052910052, 0.1269841269841271,
                    0.2592592592592592};

// template <typename MTX>
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

TEST_F(testMatrix, kPlus)
{
   MTX b(n, n);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j + 1 * (i == j);

   MTX c(m1, MTX::kPlus, eye);

   EXPECT_EQ(c, b);
}

TEST_F(testMatrix, kMinus)
{
   MTX b(n, n);

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j - 1 * (i == j);

   MTX c(m1, MTX::kMinus, eye);

   EXPECT_EQ(c, b);
}

TEST_F(testMatrix, kMult)
{

   MTX b(n, 2 * n);
   // b = m1;
   Scalar sum;

   for (int i = 0; i < n; i++) {
      sum = 0.0;
      for (int j = 0; j < n; j++)
         sum += n * i + j;
      for (int j = 0; j < 2 * n; j++)
         b(i, j) = sum;
   }

   MTX c(m1, MTX::kMult, m2);

   EXPECT_EQ(c, b);
}

TEST_F(testMatrix, kInvMult)
{

   MTX b(n, 2 * n);
   MTX tol(n, 2 * n);
   // b = m1;
   Scalar sum;
   Scalar eps = std::numeric_limits<Scalar>::min();

   for (int i = 0; i < n; i++) {
      sum = values[i];
      for (int j = 0; j < 2 * n; j++) {
         b(i, j) = sum;
         tol(i, j) = eps;
         if (i == j)
            m1(i, j) += 1;
      }
   }

   MTX c(m1, MTX::kInvMult, m2);

   CompareTMatrix(c, b);
}

TEST_F(testMatrix, Invert)
{
   MTX b(n, n);
   // b = m1;

   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
         b(i, j) = n * i + j + 1 * (i == j);

   b.Invert();

   m1 += eye;

   MTX c(m1, MTX::kMult, b);

   CompareTMatrix(c, eye);
}
