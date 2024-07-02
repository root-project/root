// Authors: Eddy Offermann   Oct 2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMatrixD.h"
#include "TMatrixDSparse.h"
#include "TMath.h"

#include "gtest/gtest.h"

#include <array>

// https://github.com/root-project/root/issues/13848
TEST(testSparse, LwbInit)
{
  constexpr int msize = 5;
  TMatrixDSparse m1(1, 4, 0, msize - 1);
  {
    constexpr int nr = 4 * msize;
    std::array<int, nr> irow;
    std::array<int, nr> icol;
    std::array<double, nr> val;

    Int_t n = 0;
    for (int i = m1.GetRowLwb(); i <= m1.GetRowUpb(); i++) {
      for (int j = m1.GetColLwb(); j <= m1.GetColUpb(); j++) {
        irow[n] = i;
        icol[n] = j;
        val[n] = TMath::Pi() * i + TMath::E() * j;
        n++;
      }
    }
    m1.SetMatrixArray(nr, irow.data(), icol.data(), val.data());
  }

  TMatrixD m2(1, 4, 0, msize - 1);
  for (int i = m2.GetRowLwb(); i <= m2.GetRowUpb(); i++)
    for (int j = m2.GetColLwb(); j <= m2.GetColUpb(); j++)
      m2(i,j) = TMath::Pi() * i + TMath::E() * j;

  EXPECT_EQ(m1, m2);
}

TEST(testSparse, Transpose)
{
   Int_t nr = 4;
   Int_t row[] = {0, 0, 1, 2};
   Int_t col[] = {1, 0, 1, 0};
   Double_t data[] = {1, 2, 3, 4};

   TMatrixDSparse m1(0, 2, 0, 1, nr, row, col, data);
   TMatrixDSparse m2(0, 1, 0, 2, nr, col, row, data);

   m1.T();

   EXPECT_EQ(m1, m2);
}

TEST(testSparse, AMultB)
{
   int n = 3;
   int m = 4;
   int nr = 4;
   int mr = 4;
   int nnz = 5;

   Int_t lhsrows[] = {0, 1, 2, 2};
   Int_t lhscols[] = {3, 2, 0, 3};
   Double_t lhsdata[] = {1., -1., 3., 4.};
   Int_t rhsrows[] = {1, 2, 3, 3};
   Int_t rhscols[] = {1, 2, 0, 2};
   Double_t rhsdata[] = {-2., 9., -2., 1.};
   Int_t rows[] = {0, 0, 1, 2, 2};
   Int_t cols[] = {0, 2, 2, 0, 2};
   Double_t data[] = {-2., 1., -9., -8., 4.};

   TMatrixDSparse lhs(0, n - 1, 0, m - 1, nr, lhsrows, lhscols, lhsdata);
   TMatrixDSparse rhs(0, m - 1, 0, n - 1, mr, rhsrows, rhscols, rhsdata);
   TMatrixDSparse m1(0, n - 1, 0, n - 1, nnz, rows, cols, data);
   TMatrixDSparse m2(lhs, TMatrixDSparse::kMult, rhs);

   EXPECT_EQ(m1, m2);
}

TEST(testSparse, AMultBt)
{
   int n = 3;
   int m = 4;
   int nr = 4;
   int mr = 4;
   int nnz = 5;

   Int_t lhsrows[] = {0, 1, 2, 2};
   Int_t lhscols[] = {3, 2, 0, 3};
   Double_t lhsdata[] = {1., -1., 3., 4.};
   Int_t rhsrows[] = {1, 2, 3, 3};
   Int_t rhscols[] = {1, 2, 0, 2};
   Double_t rhsdata[] = {-2., 9., -2., 1.};
   Int_t rows[] = {0, 0, 1, 2, 2};
   Int_t cols[] = {0, 2, 2, 0, 2};
   Double_t data[] = {-2., 1., -9., -8., 4.};

   TMatrixDSparse lhs(0, n - 1, 0, m - 1, nr, lhsrows, lhscols, lhsdata);
   TMatrixDSparse rhs(0, n - 1, 0, m - 1, mr, rhscols, rhsrows, rhsdata);
   TMatrixDSparse m1(0, n - 1, 0, n - 1, nnz, rows, cols, data);
   TMatrixDSparse m2(lhs, TMatrixDSparse::kMultTranspose, rhs);

   EXPECT_EQ(m1, m2);
}

TEST(testSparse, kAtA)
{
   int n = 3;
   int m = 4;
   int nr = 4;
   int nnz = 5;

   Int_t Arows[] = {0, 1, 2, 2};
   Int_t Acols[] = {3, 2, 0, 3};
   Double_t Adata[] = {1., -1., 3., 4.};
   Int_t rows[] = {0, 0, 2, 3, 3};
   Int_t cols[] = {0, 3, 2, 0, 3};
   Double_t data[] = {9., 12, 1., 12., 17.};

   TMatrixDSparse A(0, n - 1, 0, m - 1, nr, Arows, Acols, Adata);
   TMatrixDSparse m1(0, m - 1, 0, m - 1, nnz, rows, cols, data);
   TMatrixDSparse m2(TMatrixDSparse::kAtA, A);

   EXPECT_EQ(m1, m2);
}