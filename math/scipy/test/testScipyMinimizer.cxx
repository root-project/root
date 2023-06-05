// @(#)root/math/scipy $Id$
// Author: Omar Zapata  2023

#include <gtest/gtest.h>

#include "Math/ScipyMinimizer.h"
#include "Math/Functor.h"
#include "Math/Factory.h"
#include "Math/MultiNumGradFunction.h"
#include "Math/FitMethodFunction.h"

// target function
double RosenBrock(const double *xx)
{
   const Double_t x = xx[0];
   const Double_t y = xx[1];
   const Double_t tmp1 = y - x * x;
   const Double_t tmp2 = 1 - x;
   return 100 * tmp1 * tmp1 + tmp2 * tmp2;
}

// gradient function(jacobian)
double RosenBrockGrad(const double *x, unsigned int ipar)
{
   if (ipar == 0)
      return -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
   else
      return 200 * (x[1] - x[0] * x[0]);
}
// Hessian function
bool RosenBrockHessian(const std::vector<double> &xx, double *hess)
{
   const double x = xx[0];
   const double y = xx[1];

   hess[0] = 1200 * x * x - 400 * y + 2;
   hess[1] = -400 * x;
   hess[2] = -400 * x;
   hess[3] = 200;

   return true;
}

// https://de.mathworks.com/help/optim/ug/example-nonlinear-constrained-minimization.html
// constraint function
double ConstRosenBrock(const std::vector<double> &x)
{
   return x[0] * x[0] - x[1] * x[1] + 1;
}
// NOTE: "dogleg" is problematic, requires better tuning of the parameters
std::string methods[] = {"Nelder-Mead", "L-BFGS-B",  "Powell",      "CG",           "BFGS",
                         "TNC",         "COBYLA",    "SLSQP",       "trust-constr", "Newton-CG",
                         "trust-ncg", "trust-exact", "trust-krylov"};

// Testing fit using class with gradient
class ScipyFitClass : public ::testing::Test {
public:
   ROOT::Math::Experimental::ScipyMinimizer *minimizer = nullptr;

   ScipyFitClass() = default;
   void Fit(const char *method, bool useConstraint = false)
   {
      ROOT::Math::GradFunctor f(&RosenBrock, &RosenBrockGrad, 2);

      minimizer = new ROOT::Math::Experimental::ScipyMinimizer(method);
      minimizer->SetMaxFunctionCalls(1000000);
      minimizer->SetMaxIterations(100000);
      minimizer->SetTolerance(0.001);

      double step[2] = {0.01, 0.01};
      double variable[2] = {0.1, 1.2};

      minimizer->SetFunction(f);
      minimizer->SetHessianFunction(RosenBrockHessian);

      // Set the free variables to be minimized!
      minimizer->SetVariable(0, "x", variable[0], step[0]);
      minimizer->SetVariable(1, "y", variable[1], step[1]);
      if (useConstraint)
         minimizer->AddConstraintFunction(ConstRosenBrock, "ineq");
      
      minimizer->Minimize();
   }
};

TEST_F(ScipyFitClass, Fit) 
{
   for (const std::string &text : methods) {
      Fit(text.c_str(),false);
      EXPECT_EQ(1000000, minimizer->MaxFunctionCalls());

      EXPECT_EQ(100000, minimizer->MaxIterations());

      EXPECT_EQ(0.001, minimizer->Tolerance());

      auto step = minimizer->StepSizes();
      EXPECT_EQ(0.01, step[0]);
      EXPECT_EQ(0.01, step[1]);

      auto v1name = minimizer->VariableName(0);
      auto v2name = minimizer->VariableName(1);
      EXPECT_EQ("x", v1name);
      EXPECT_EQ("y", v2name);

      EXPECT_EQ(0, minimizer->VariableIndex("x"));
      EXPECT_EQ(1, minimizer->VariableIndex("y"));

      auto opts = minimizer->Options();

      EXPECT_EQ("Scipy", opts.MinimizerType());
      EXPECT_EQ(text, opts.MinimizerAlgorithm());

      auto x = minimizer->X();
      ASSERT_NEAR(1, x[0], 0.5);
      ASSERT_NEAR(1, x[1], 0.5);
      ASSERT_NEAR(0, RosenBrock(x), 0.5);
   }
}


TEST_F(ScipyFitClass, FitContraint) // using constraint function
{
   for (const std::string &text : methods) {
      Fit(text.c_str(), true);
      auto x = minimizer->X();
      ASSERT_NEAR(1, x[0], 0.5);
      ASSERT_NEAR(1, x[1], 0.5);
      ASSERT_NEAR(0, RosenBrock(x), 0.5);
   }
}