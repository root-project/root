/// \file TFormulaHessianTests.cxx
///
/// \brief The file contain unit tests which test the clad-based hessian
///        computations.
///
/// \author Baidyanath Kundu <kundubaidya99@gmail.com>
///
/// \date Aug, 2021
///
/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOTUnitTestSupport.h"

#include <Math/MinimizerOptions.h>
#include <TFormula.h>

TEST(TFormulaHessianPar, Sanity)
{
   TFormula f("f", "x*std::sin([0]) - y*std::cos([1])");
   double p[] = {30, 60};
   f.SetParameters(p);
   double x[] = {1, 2};
   TFormula::CladStorage result(4);
   f.HessianPar(x, result);

   ASSERT_FLOAT_EQ(-1 * x[0] * std::sin(p[0]), result[0]);
   ASSERT_FLOAT_EQ(0, result[1]);
   ASSERT_FLOAT_EQ(0, result[2]);
   ASSERT_FLOAT_EQ(x[1] * std::cos(p[1]), result[3]);
}

TEST(TFormulaHessianPar, ResultUpsize)
{
   TFormula f("f", "std::sin([1]) - std::cos([0])");
   double p[] = {60, 30};
   f.SetParameters(p);
   TFormula::CladStorage result;
   double x[] = {2, 1};

   ASSERT_TRUE(0 == result.size());
   ROOT_EXPECT_WARNING(f.HessianPar(x, result),
   "TFormula::HessianPar",
   "The size of hessian result is 0 but 4 is required. Resizing."
   );

   ASSERT_FLOAT_EQ(std::cos(p[0]), result[0]);
   ASSERT_FLOAT_EQ(0, result[1]);
   ASSERT_FLOAT_EQ(0, result[2]);
   ASSERT_FLOAT_EQ(- std::sin(p[1]), result[3]);
   ASSERT_TRUE(4 == result.size());
}

TEST(TFormulaHessianPar, ResultDownsize)
{
   TFormula f("f", "std::sin([0])");
   double p[] = {60};
   f.SetParameters(p);
   TFormula::CladStorage result(2);
   double x[] = {1};

   ASSERT_TRUE(2 == result.size());

   ROOT_EXPECT_NODIAG(f.HessianPar(x, result));

   ASSERT_FLOAT_EQ(- std::sin(p[0]), result[0]);
   ASSERT_TRUE(2 == result.size());
}

TEST(TFormulaHessianPar, GetHessFormula)
{
   TFormula f("f", "gaus");
   double p[] = {3, 1, 2};
   f.SetParameters(p);
   ASSERT_TRUE(f.GenerateHessianPar());
   std::string s = f.GetHessianFormula().Data();
   // Windows does not support posix regex which are necessary here.
   #ifndef R__WIN32
   ASSERT_THAT(s, testing::ContainsRegex("void TFormula____id[0-9]*_hessian_1"));
   #endif // R__WIN32
}