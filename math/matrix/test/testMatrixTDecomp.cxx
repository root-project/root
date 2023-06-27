// Authors: Nicolas Morange   Dec 2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TDecompBase.h>

#include <gtest/gtest.h>

#include <iostream>

// This is just so the can use the protected DiagProd funciton in the test.
class TDecompDummy : public TDecompBase {
public:
   static void DiagProd(const TVectorD &diag, Double_t tol, Double_t &d1, Double_t &d2)
   {
      return TDecompBase::DiagProd(diag, tol, d1, d2);
   }
};

// https://github.com/root-project/root/issues/13110
TEST(testDecomp, DiagProd)
{
   TVectorD v(1);
   v[0] = 1024;
   double d1;
   double d2;
   TDecompDummy::DiagProd(v, 0.1, d1, d2);

   // DiagProd returns the product of matrix diagonal elements in d1 and d2. d1
   // is a mantissa and d2 an exponent for powers of 2. This is why we are
   // using this specific formula to validate the method.
   EXPECT_EQ(d1 * std::pow(2, d2), v[0]);
}
