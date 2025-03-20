// Tests for the RooFormula
// Authors: Stephan Hageboeck, CERN  2020
//          Jonas Rembser, CERN 2023

#include "../src/RooFormula.h"
#include <RooFormulaVar.h>
#include <RooRealVar.h>

#include <ROOT/TestSupport.hxx>

#include <gtest/gtest.h>

/// Since TFormula does very surprising things,
/// RooFit needs to do safety checks.
/// ```
/// TFormula form("form", "x+y");
/// form.Eval(3.);
/// ```
/// is, for example, legal, and silently uses an undefined
/// value for y. RooFit needs to detect this.
TEST(RooFormula, TestInvalidFormulae)
{
   ROOT::TestSupport::CheckDiagsRAII checkDiag;
   checkDiag.requiredDiag(kError, "prepareMethod", "Can't compile function TFormula", false);
   checkDiag.requiredDiag(kError, "TFormula::InputFormulaIntoCling", "Error compiling formula expression in Cling",
                          true);
   checkDiag.requiredDiag(kError, "TFormula::ProcessFormula", " is invalid", false);
   checkDiag.requiredDiag(kError, "TFormula::ProcessFormula", "has not been matched in the formula expression", false);
   checkDiag.requiredDiag(kError, "cling", "undeclared identifier", false);

   RooRealVar x("x", "x", 1.337);
   RooRealVar y("y", "y", -1.);
   RooFormula form("form", "x+10", x);
   EXPECT_FLOAT_EQ(form.eval(nullptr), 11.337);

   ASSERT_ANY_THROW(RooFormula("form", "x+y", x)) << "Formulae with y,z,t and no RooFit variable cannot work.";
   ASSERT_ANY_THROW(RooFormula("form", "x+z", x)) << "Formulae with y,z,t and no RooFit variable cannot work.";
   ASSERT_ANY_THROW(RooFormula("form", "x+t", x)) << "Formulae with y,z,t and no RooFit variable cannot work.";
   ASSERT_ANY_THROW(RooFormula("form", "x+a", x)) << "Formulae with unknown variable cannot work.";

   std::unique_ptr<RooFormula> form6;
   ASSERT_NO_THROW(form6 = std::make_unique<RooFormula>("form", "x+y", RooArgList{x, y}))
      << "Formula with x,y must work.";
   ASSERT_NE(form6, nullptr);
   EXPECT_FLOAT_EQ(form6->eval(nullptr), 1.337 - 1.);
}

// In case of named arguments, the RooFormula will replace the argument names
// with x[0] to x[n]. There are two things that can go wrong if RooFormula is
// not implemented right. First, if there is a variable named "x" it should
// only be substituted if the matching substring is not followed by "[", to not
// replace existing x[i]. Second, variables with integer names like "0" should
// only be substituted if the match is not followed by a "]", again to avoid
// replacing x[i]. This test checks that these cases are handled correctly.
TEST(RooFormula, TestDangerousVariableNames)
{
   RooRealVar dt("dt", "dt", -10, 10);
   RooRealVar x("x", "x", 1.547);
   RooRealVar zero("0", "0", 0);

   // Create the formula, triggers an error if the formula doesn't compile
   // correctly because the dangerous variable names haven't been treated right.
   RooFormula formula("formula", "exp(-abs(@0)/@1)*cos(@0*@2)", {dt, x, zero});
}

/// Check that the RooFormulaVar has the right number of servers when some
/// variables are unused.
TEST(RooFormula, UnusedVariables)
{
   RooRealVar x{"x", "x", 1};
   RooRealVar y{"y", "y", 2};
   RooRealVar z{"z", "z", 3};

   RooFormulaVar func{"func", "x * y", {x, y, z}};

   // There are expected to be two servers only because "z" is not used in the
   // formula.
   EXPECT_EQ(func.servers().size(), 2);
}

TEST(RooFormula, UndefinedVariables)
{
   RooRealVar B("B", "", 0.516952);
   RooRealVar r("r", "", 0.214107);
   RooRealVar x("x", "", 0.2, 1);
   RooRealVar y("y", "", 0.2, 1);

   ASSERT_ANY_THROW(RooFormulaVar f1("f1", "r + B + x", {r, B}))  << "Formulae with missing x in arg list cannot work.";
   ASSERT_ANY_THROW(RooFormulaVar f2("f2", "r + B + y", {r, B}))  << "Formulae with missing (x,)y in arg list cannot work.";
   ASSERT_NO_THROW(RooFormulaVar f2("f2", "r + B + y", {r, B, y})) << "Formula with specified y must work.";
}
