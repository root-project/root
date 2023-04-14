#include "Math/ScipyMinimizer.h"
#include "Math/MultiNumGradFunction.h"
#include <Fit/ParameterSettings.h>
#include "Math/Functor.h"
#include <string>
#include "Math/MinimizerOptions.h"
#include "TStopwatch.h"


double RosenBrock(const double *xx )
{
  const Double_t x = xx[0];
  const Double_t y = xx[1];
  const Double_t tmp1 = y-x*x;
  const Double_t tmp2 = 1-x;
  return 100*tmp1*tmp1+tmp2*tmp2;
}

double RosenBrockGrad(const double *x, unsigned int ipar)
{
   if (ipar == 0)
      return -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
   else
      return 200 * (x[1] - x[0] * x[0]);
}

bool RosenBrockHessian(const std::vector<double> &xx, double *hess)
{
   const double x = xx[0];
   const double y = xx[1];
   
   hess[0] = 1200*x*x - 400*y + 2;
   hess[1] = -400*x;
   hess[2] = -400*x;
   hess[3] = 200;
   
   return true;
}

// methods that requires hessian to work "dogleg", "trust-ncg","trust-exact","trust-krylov"
using namespace std;
int scipy()
{ 
   
   std::string methods[]={"Nelder-Mead","L-BFGS-B","Powell","CG","BFGS","TNC","COBYLA","SLSQP","trust-constr","Newton-CG", "dogleg", "trust-ncg","trust-exact","trust-krylov"};
   TStopwatch t;
   for(const std::string &text : methods)
   {
      ROOT::Math::Experimental::ScipyMinimizer minimizer(text.c_str());
      minimizer.SetMaxFunctionCalls(1000000);
      minimizer.SetMaxIterations(100000);
      minimizer.SetTolerance(1e-3);
      minimizer.SetExtraOption("gtol",1e-3);
      ROOT::Math::GradFunctor f(&RosenBrock,&RosenBrockGrad,2); 
      double step[2] = {0.01,0.01};
      double variable[2] = { -1.2,1.0};
   
      minimizer.SetFunction(f);
      minimizer.SetHessianFunction(RosenBrockHessian);
   
      // variables to be minimized!
      minimizer.SetVariable(0,"x",variable[0], step[0]);
      minimizer.SetVariable(1,"y",variable[1], step[1]);
      minimizer.SetVariableLimits(0, -2.0, 2.0);
      minimizer.SetVariableLimits(1, -2.0, 2.0);

      t.Reset();
      t.Start();
      minimizer.Minimize(); 
      t.Stop();
      const double *xs = minimizer.X();
      cout << "Minimum: f(" << xs[0] << "," << xs[1] << "): " 
         << RosenBrock(xs) << endl;
      cout << "Cpu Time (sec) = " << t.CpuTime() <<endl<< "Real Time (sec) = " << t.RealTime() << endl;
      cout << endl << "===============" << endl;
   }
   return 0;
}

int main()
{
  return scipy();
}