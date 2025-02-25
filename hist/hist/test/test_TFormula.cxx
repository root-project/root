#include "gtest/gtest.h"
#include "TF1.h"
#include "ROOT/TestSupport.hxx"
#include <cmath>

#include "TFormula.h"

using namespace ROOT;

// Test that autoloading works (ROOT-9840)
TEST(TFormula, Interp)
{
  TFormula f("func", "TGeoBBox::DeclFileLine()");
}

// Test for TFormula Extended syntax support
TEST(TFormulaPolTest, BasicPolynomialConstruction)
{

   TFormula f1("f1", "pol1");
   EXPECT_EQ(f1.GetExpFormula(), TString("([p0]+[p1]*x)"));

   TFormula f2("f2", "pol2");
   EXPECT_EQ(f2.GetExpFormula(), TString("([p0]+[p1]*x+[p2]*TMath::Sq(x))"));
}

TEST(TFormulaPolTest, VariablePolynomials)
{

   TFormula f1("f1", "pol1(y,0)");
   EXPECT_EQ(f1.GetExpFormula(), TString("([p0]+[p1]*y)"));

   TFormula f2("f2", "pol2(z,0)");
   EXPECT_EQ(f2.GetExpFormula(), TString("([p0]+[p1]*z+[p2]*TMath::Sq(z))"));
}

TEST(TFormulaPolTest, ParameterPlaceholders)
{

   TFormula f1("f1", "pol1(x,[A], [B])");
   EXPECT_EQ(f1.GetExpFormula(), TString("([A]+[B]*x)"));
   f1.SetParameter("A", -1.234);
   EXPECT_EQ(f1.GetParameter("A"), -1.234);
   EXPECT_EQ(f1.GetParameter(0), -1.234);
   f1.SetParameter(0, -1.2345);
   EXPECT_EQ(f1.GetParameter("A"), -1.2345);
   EXPECT_EQ(f1.GetParameter(0), -1.2345);
}

TEST(TFormulaPolTest, NumericEvaluation)
{

   TF1 f1("f1", "pol1(x,0)");
   f1.SetParameters(1.0, 2.0); // f(x) = 1 + 2x

   EXPECT_NEAR(f1.Eval(0.0), 1.0, 1e-10);
   EXPECT_NEAR(f1.Eval(1.0), 3.0, 1e-10);
}

TEST(TFormulaPolTest, Parameters)
{

   TFormula f2("f2", "pol2(x, [A], [B], [C])");
   EXPECT_EQ(f2.GetExpFormula(), TString("([A]+[B]*x+[C]*TMath::Sq(x))"));
   f2.SetParameter("A", -1.234);
   EXPECT_EQ(f2.GetParameter("A"), -1.234);
   EXPECT_EQ(f2.GetParameter(0), -1.234);
   f2.SetParameter(0, -1.234);
   EXPECT_EQ(f2.GetParameter("A"), -1.234);
   EXPECT_EQ(f2.GetParameter(0), -1.234);
   f2.SetParameter("B", -1.235);
   EXPECT_EQ(f2.GetParameter("B"), -1.235);
   EXPECT_EQ(f2.GetParameter(1), -1.235);
   f2.SetParameter(1, -1.235);
   EXPECT_EQ(f2.GetParameter("B"), -1.235);
   EXPECT_EQ(f2.GetParameter(1), -1.235);
   f2.SetParameter("C", -1.236);
   EXPECT_EQ(f2.GetParameter("C"), -1.236);
   EXPECT_EQ(f2.GetParameter(2), -1.236);
   f2.SetParameter(2, -1.236);
   EXPECT_EQ(f2.GetParameter("C"), -1.236);
   EXPECT_EQ(f2.GetParameter(2), -1.236);
}

TEST(TFormulaPolTest, CompoundExpressions)
{

   TFormula f1("f1", "pol1(x,0) + pol1(y,2)");
   EXPECT_EQ(f1.GetExpFormula(), TString("([p0]+[p1]*x)+([p2]+[p3]*y)"));
}
