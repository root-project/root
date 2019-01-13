// @(#)root/tmva $Id$
// Author: Omar Zapata  http://oproject.org/pages/Ipopt.html

/*************************************************************************
 * Copyright (C) 2019, Omar.Zapata@cern.ch                               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <gtest/gtest.h>

#include "Math/IpoptMinimizer.h"
#include "Math/Functor.h"
#include "Math/Factory.h"
#include "Math/MultiNumGradFunction.h"
#include "Math/FitMethodFunction.h"

// Class with gradient

class RosenBrockGradientFunction : public ROOT::Math::IGradientFunctionMultiDim {
public:
   double DoEval(const double *xx) const
   {
      const Double_t x = xx[0];
      const Double_t y = xx[1];
      const Double_t tmp1 = y - x * x;
      const Double_t tmp2 = 1 - x;
      return 100 * tmp1 * tmp1 + tmp2 * tmp2;
   }
   unsigned int NDim() const { return 2; }
   ROOT::Math::IGradientFunctionMultiDim *Clone() const { return new RosenBrockGradientFunction(); }
   double DoDerivative(const double *x, unsigned int ipar) const
   {
      if (ipar == 0)
         return -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
      else
         return 200 * (x[1] - x[0] * x[0]);
   }
};

// for tests with Functor intead class
double RosenBrock(const double *xx)
{
   const Double_t x = xx[0];
   const Double_t y = xx[1];
   const Double_t tmp1 = y - x * x;
   const Double_t tmp2 = 1 - x;
   return 100 * tmp1 * tmp1 + tmp2 * tmp2;
}

// Testing fit using class with gradient
class IpoptFitClass : public ::testing::Test {
public:
   ROOT::Math::Experimental::IpoptMinimizer *minimizer1;
   ROOT::Math::Experimental::IpoptMinimizer *minimizer2;
   ROOT::Math::Experimental::IpoptMinimizer *minimizer3;

   ROOT::Math::Minimizer *minimizer4; // using plugin manager with Factory

   RosenBrockGradientFunction rgf;
   ROOT::Math::Functor f;
   ROOT::Math::GradFunctor gf;
   ROOT::Math::Functor f2;

   IpoptFitClass() : f(rgf, 2), gf(rgf, 2), f2(&RosenBrock, 2)
   {
      minimizer1 = new ROOT::Math::Experimental::IpoptMinimizer("mumps");
      minimizer2 = new ROOT::Math::Experimental::IpoptMinimizer();
      minimizer3 = new ROOT::Math::Experimental::IpoptMinimizer();

      minimizer4 = ROOT::Math::Factory::CreateMinimizer("Ipopt", "mumps");
   }
   ~IpoptFitClass()
   {
      delete minimizer1;
      delete minimizer2;
      delete minimizer3;
      delete minimizer4;
   }
   void Fit1() // instance with param mumps
   {
      minimizer1->SetMaxFunctionCalls(1000000);
      minimizer1->SetMaxIterations(100000);
      minimizer1->SetTolerance(0.001);

      double step[2] = {0.01, 0.01};
      double variable[2] = {0.1, 1.2};

      minimizer1->SetFunction(this->f);
      minimizer1->SetFunction(this->gf);

      // Set the free variables to be minimized!
      minimizer1->SetVariable(0, "x", variable[0], step[0]);
      minimizer1->SetVariable(1, "y", variable[1], step[1]);

      minimizer1->Minimize();
   }

   void Fit2() // instance without params
   {
      minimizer2->SetMaxFunctionCalls(1000000);
      minimizer2->SetMaxIterations(100000);
      minimizer2->SetTolerance(0.001);

      double step[2] = {0.01, 0.01};
      double variable[2] = {0.1, 1.2};

      minimizer2->SetFunction(this->f);
      minimizer2->SetFunction(this->gf);

      // Set the free variables to be minimized!
      minimizer2->SetVariable(0, "x", variable[0], step[0]);
      minimizer2->SetVariable(1, "y", variable[1], step[1]);

      minimizer2->Minimize();
   }

   void Fit3() // calling Functor using RosenBrock function instead class
   {
      minimizer3->SetMaxFunctionCalls(1000000);
      minimizer3->SetMaxIterations(100000);
      minimizer3->SetTolerance(0.001);

      double step[2] = {0.01, 0.01};
      double variable[2] = {0.1, 1.2};

      minimizer3->SetFunction(this->f2);

      // Set the free variables to be minimized!
      minimizer3->SetVariable(0, "x", variable[0], step[0]);
      minimizer3->SetVariable(1, "y", variable[1], step[1]);

      minimizer3->Minimize();
   }

   void Fit4() // calling Functor using RosenBrock function instead class
   {
      minimizer4->SetMaxFunctionCalls(1000000);
      minimizer4->SetMaxIterations(100000);
      minimizer4->SetTolerance(0.001);

      double step[2] = {0.01, 0.01};
      double variable[2] = {0.1, 1.2};

      minimizer4->SetFunction(this->f2);

      // Set the free variables to be minimized!
      minimizer4->SetVariable(0, "x", variable[0], step[0]);
      minimizer4->SetVariable(1, "y", variable[1], step[1]);

      minimizer4->Minimize();
   }
};

// Fit options and param settings tests for Class with param "mumps"
TEST_F(IpoptFitClass, Fit1)
{
   Fit1();
   EXPECT_EQ(1000000, minimizer1->MaxFunctionCalls());

   EXPECT_EQ(100000, minimizer1->MaxIterations());

   EXPECT_EQ(0.001, minimizer1->Tolerance());

   auto step = minimizer1->StepSizes();
   EXPECT_EQ(0.01, step[0]);
   EXPECT_EQ(0.01, step[1]);

   auto v1name = minimizer1->VariableName(0);
   auto v2name = minimizer1->VariableName(1);
   EXPECT_EQ("x", v1name);
   EXPECT_EQ("y", v2name);

   EXPECT_EQ(0, minimizer1->VariableIndex("x"));
   EXPECT_EQ(1, minimizer1->VariableIndex("y"));

   auto opts = minimizer1->Options();

   EXPECT_EQ("Ipopt", opts.MinimizerType());
   EXPECT_EQ("mumps", opts.MinimizerAlgorithm());

   auto x = minimizer1->X();
   ASSERT_NEAR(1, x[0], 0.5);
   ASSERT_NEAR(1, x[1], 0.5);
   ASSERT_NEAR(0, RosenBrock(x), 0.5);
}

TEST_F(IpoptFitClass, Fit2)
{
   Fit2();
   auto x = minimizer2->X();
   ASSERT_NEAR(1, x[0], 0.5);
   ASSERT_NEAR(1, x[1], 0.5);
   ASSERT_NEAR(0, RosenBrock(x), 0.5);
}

TEST_F(IpoptFitClass, Fit3)
{
   Fit3();
   auto x = minimizer3->X();
   ASSERT_NEAR(1, x[0], 0.5);
   ASSERT_NEAR(1, x[1], 0.5);
   ASSERT_NEAR(0, RosenBrock(x), 0.5);
}

TEST_F(IpoptFitClass, Fit4)
{
   Fit4();
   auto x = minimizer4->X();
   ASSERT_NEAR(1, x[0], 0.5);
   ASSERT_NEAR(1, x[1], 0.5);
   ASSERT_NEAR(0, RosenBrock(x), 0.5);
}
