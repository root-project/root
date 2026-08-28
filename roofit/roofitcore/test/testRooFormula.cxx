// Tests for the RooFormulaUtils functions and RooFormulaVar
// Authors: Stephan Hageboeck, CERN  2020
//          Jonas Rembser, CERN 2023
//          Andrea Germinario, CERN 2025
#include <TFile.h>

#include "../src/RooFormulaUtils.h"
#include <RooFit/Evaluator.h>
#include <RooFormulaVar.h>
#include <RooRealVar.h>
#include <RooConstVar.h>
#include <RooWorkspace.h>

#include <ROOT/TestSupport.hxx>

#include <gtest/gtest.h>

#include <span>
#include <vector>

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
   auto form = RooFormulaUtils::makeFormulaEvaluator("form", "x+10", x);
   EXPECT_FLOAT_EQ(RooFormulaUtils::evalFormula(*form, RooArgList{x}), 11.337);

   using RooFormulaUtils::makeFormulaEvaluator;
   ASSERT_ANY_THROW(makeFormulaEvaluator("form", "x+y", x))
      << "Formulae with y,z,t and no RooFit variable cannot work.";
   ASSERT_ANY_THROW(makeFormulaEvaluator("form", "x+z", x))
      << "Formulae with y,z,t and no RooFit variable cannot work.";
   ASSERT_ANY_THROW(makeFormulaEvaluator("form", "x+t", x))
      << "Formulae with y,z,t and no RooFit variable cannot work.";
   ASSERT_ANY_THROW(makeFormulaEvaluator("form", "x+a", x)) << "Formulae with unknown variable cannot work.";

   std::unique_ptr<RooFormulaEvaluator> form6;
   ASSERT_NO_THROW(form6 = makeFormulaEvaluator("form", "x+y", RooArgList{x, y})) << "Formula with x,y must work.";
   ASSERT_NE(form6, nullptr);
   EXPECT_FLOAT_EQ(RooFormulaUtils::evalFormula(*form6, RooArgList{x, y}), 1.337 - 1.);
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
   RooConstVar zero("0", "0", 0);

   // Create the formula, triggers an error if the formula doesn't compile
   // correctly because the dangerous variable names haven't been treated right.
   RooFormulaUtils::makeFormulaEvaluator("formula", "exp(-abs(@0)/@1)*cos(@0*@2)", {dt, x, zero});
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

// Regression test for https://github.com/root-project/root/issues/21371:
// an unused parameter (b) is pruned, so the persisted @N indices must be
// remapped or the formula silently mismaps after a write/read cycle.
TEST(RooFormula, SerializationWithUnusedParam)
{
   RooWorkspace w("w");
   w.factory("a[2,-10,10]");
   w.factory("b[99,-10,10]");
   w.factory("c[3,-10,10]");
   w.factory("d[4,-10,10]");
   w.factory("expr::f('@0*@2+d', a, b, c, d)");

   TString fn = "RooFormulaSerialization.root";
   w.writeToFile(fn);
   TFile fin(fn);
   RooWorkspace *w2 = nullptr;
   fin.GetObject("w", w2);
   ASSERT_NE(w2, nullptr);
   auto *f = static_cast<RooAbsReal *>(w2->function("f"));

   // If @2 still maps to c, changing c updates f = a*c + d = 2*5 + 4 = 14.
   static_cast<RooRealVar *>(w2->var("c"))->setVal(5.0);
   EXPECT_DOUBLE_EQ(f->getVal(), 2.0 * 5.0 + 4.0);
}

TEST(RooFormula, RooConstVarSafeSubstitution)
{
   // Check RooConst are substituted only by index
   ASSERT_NO_THROW(RooFormulaVar f("f", "2.7*@0", RooFit::RooConst(2.)))
      << "Formulae with RooConstVar argument should be substituted only by index.";

   // Check that constant values to be used in RooFormulaVar have to be RooConstVar
   RooRealVar x("x", "x", 1.547);
   RooRealVar zero("0", "0", 0); // Constant values should be RooConstVar
   ASSERT_ANY_THROW(RooFormulaVar f1("f1", "x + 0", {x, zero}))
      << "Const arguments in a RooFormula should be of type RooConstVar";

   // Check that RooConstVar having a value as name has value==(double)name
   RooConstVar troubleConst("3.4", "troubleConst", 2.1);
   ASSERT_ANY_THROW(RooFormulaVar f1("f1", "x + 0", {x, zero}))
      << "RooConst variables, if having numeric name, should have name value equal to actual value.";
}

// Regression test for a crash in batch evaluation: RooFormula::doEval
// dereferenced the empty input span of a dependent that is not used by the
// formula, segfaulting the batch evaluation backend for e.g.
// RooFormulaVar("f", "x*p", {x, p, q}).
TEST(RooFormula, UnusedDependentBatchEval)
{
   RooRealVar x("x", "x", 5.0, 0.0, 10.0);
   RooRealVar p("p", "p", 2.0, 0.1, 3.0);
   RooRealVar q("q", "q", 1.5, 0.1, 3.0); // not used by the formula
   RooFormulaVar f("f", "x*p", {x, p, q});

   const std::vector<double> xData{1.0, 2.0, 3.0, 4.0, 5.0};
   RooFit::Evaluator ev(f);
   ev.setInput("x", {xData.data(), xData.size()}, false);
   std::span<const double> out = ev.run();
   ASSERT_EQ(out.size(), xData.size());
   for (std::size_t i = 0; i < xData.size(); ++i) {
      EXPECT_DOUBLE_EQ(out[i], xData[i] * 2.0) << "event " << i;
   }
}
