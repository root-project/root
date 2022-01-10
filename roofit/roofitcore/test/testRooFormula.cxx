// Tests for the RooFormula
// Author: Stephan Hageboeck, CERN  2020

#include "RooFormula.h"
#include "RooRealVar.h"

#include "ROOT/TestSupport.hxx"

#include "gtest/gtest.h"

/// Since TFormula does very surprising things,
/// RooFit needs to do safety checks.
/// ```
/// TFormula form("form", "x+y");
/// form.Eval(3.);
/// ```
/// is, for example, legal, and silently uses an undefined
/// value for y. RooFit needs to detect this.
TEST(RooFormula, TestInvalidFormulae) {
  ROOT::TestSupport::CheckDiagsRAII checkDiag;
  checkDiag.requiredDiag(kError, "prepareMethod", "Can't compile function TFormula", false);
  checkDiag.requiredDiag(kError, "TFormula::InputFormulaIntoCling", "Error compiling formula expression in Cling", true);
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

  RooFormula* form6 = nullptr;
  ASSERT_NO_THROW( form6 = new RooFormula("form", "x+y", RooArgSet(x,y))) << "Formula with x,y must work.";
  ASSERT_NE(form6, nullptr);
  EXPECT_FLOAT_EQ(form6->eval(nullptr), 1.337 - 1.);
  delete form6;
}
