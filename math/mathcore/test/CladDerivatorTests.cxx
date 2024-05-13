/// \file CladDerivatorTests.cxx
///
/// \brief The file contain unit tests which test the CladDerivator facility.
///
/// \author Vassil Vassilev <vvasilev@cern.ch>
///
/// \date July, 2018
///
/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TInterpreter.h>
#include <TInterpreterValue.h>

#include "gtest/gtest.h"

TEST(CladDerivator, Sanity)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           static double pow2(double x) { return x * x; })cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(pow2, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_FLOAT_EQ(2, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, power)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double cube(double x){ return TMath::Power(x,3);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(cube, 0).execute(4)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(48, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, Power2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double cube2(double x){ return TMath::Power(x,3);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(cube2, 0).execute(4)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(24, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, SinCos1)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double sincos1(double x){ return TMath::Sin(x)+ TMath::Cos(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(sincos1, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, SinCos2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double sincos2(double x){ return TMath::Sin(x)+ TMath::Cos(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(sincos2, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-1, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, SinHCosH)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double sinhcosh(double x){ return TMath::SinH(x)+ TMath::CosH(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(sinhcosh, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, absolute)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double absolute(double x){ return TMath::Abs(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(absolute, 0).execute(4)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, exp)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double ex(double x){ return TMath::Exp(2*x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(ex, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(2, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, exp2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double ex2(double x){ return TMath::Exp(2*x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(ex2, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(4, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, logx)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double l(double x){ return TMath::Log(2*x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(l, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, logx2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double l2(double x){ return TMath::Log(2*x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(l2, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-1, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, logx3)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double l3(double x){ return TMath::Log2(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(l3, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1.4426950408889634, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, logx4)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double l4(double x){ return TMath::Log2(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(l4, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-1.4426950408889634, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, logx5)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double l5(double x){ return TMath::Log10(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(l5, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(0.43429448190325176, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, logx6)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double l6(double x){ return TMath::Log10(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(l6, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-0.43429448190325176, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, minimum)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double minimum(double x, double y){ return TMath::Min(x,y);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(minimum, 0).execute(5,7)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1, value->GetAsDouble());
   delete value;
}

TEST(CladDerivator, erf1)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double e1(double x){ return TMath::Erf(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(e1, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(0.41510749742059477, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, erf2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double e2(double x){ return TMath::Erf(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(e2, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-0.83021499484118955, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, erfc1)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double ec1(double x){ return TMath::Erfc(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(ec1, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-0.41510749742059477, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, erfc2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double ec2(double x){ return TMath::Erfc(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(ec2, 0).execute(1)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(0.83021499484118955, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, power3)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double powneg(double x,double y){ return TMath::Power(x,y);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(powneg, 0).execute(2, -2)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(-0.25, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, power4)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double powneg2(double x,double y){ return TMath::Power(x,y);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(powneg2, 0).execute(2, -2)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(0.375, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, maxtest)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double max1(double x,double y){ return TMath::Max(x,y);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(max1, 0).execute(5, 2)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(1, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, maxtest2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double max2(double x,double y){ return TMath::Max(x,y);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(max1, 0).execute(5, 2)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(0, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, tantest)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double tan1(double x){ return TMath::Tan(x)+ TMath::TanH(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate(tan1, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(2, value->GetAsDouble());
   delete value;
}
TEST(CladDerivator, tantest2)
{
   gInterpreter->Declare(R"cpp(
                           #include <Math/CladDerivator.h>
                           #include "TMath.h"
                           double tan2(double x){ return TMath::Tan(x)+ TMath::TanH(x);})cpp");
   auto value = gInterpreter->CreateTemporary();
   std::string code = "clad::differentiate<2>(tan2, 0).execute(0)";
   ASSERT_TRUE(gInterpreter->Evaluate(code.c_str(), *value));
   ASSERT_DOUBLE_EQ(0, value->GetAsDouble());
   delete value;
}