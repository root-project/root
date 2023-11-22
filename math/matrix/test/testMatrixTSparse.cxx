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
