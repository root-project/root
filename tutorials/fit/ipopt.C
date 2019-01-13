/// \file
/// \ingroup tutorial_fit
/// \notebook -nodraw
/// Ipopt Multi-Dimensional Parametrisation and Fitting
///
/// \macro_output
/// \macro_code
///
/// \authors Omar Zapata

#include "Math/IpoptMinimizer.h"
#include "Math/Functor.h"
#include "Math/MultiNumGradFunction.h"
#include "Math/FitMethodFunction.h"


class RosenBrockGradientFunction: public ROOT::Math::IGradientFunctionMultiDim{
public:
   double DoEval(const double* xx) const{
    const Double_t x = xx[0];
    const Double_t y = xx[1];
    const Double_t tmp1 = y-x*x;
    const Double_t tmp2 = 1-x;
    return 100*tmp1*tmp1+tmp2*tmp2;
   }
   unsigned int NDim() const{
      return 2;
   }
   ROOT::Math::IGradientFunctionMultiDim* Clone() const{
      return new RosenBrockGradientFunction();
   }
   double DoDerivative(const double* x, unsigned int ipar) const{
      if ( ipar == 0 )
         return -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
      else
         return 200 * (x[1] - x[0] * x[0]);
   }
};
 
int ipopt()
{
   // Choose method upon creation between:
   ROOT::Math::Experimental::IpoptMinimizer minimizer("mumps");
   minimizer.SetMaxFunctionCalls(1000000);
   minimizer.SetMaxIterations(100000);
   minimizer.SetTolerance(0.001);

   RosenBrockGradientFunction rgf;

   ROOT::Math::Functor f(rgf,2);
   ROOT::Math::GradFunctor gf(rgf,2);
   
 
   double step[2] = {0.01,0.01};
   double variable[2] = { 0.1,1.2};
 
  
   minimizer.SetFunction(f);
   minimizer.SetFunction(gf);
 
   // Set the free variables to be minimized!
   minimizer.SetVariable(0,"x",variable[0], step[0]);
   minimizer.SetVariable(1,"y",variable[1], step[1]);
 
   minimizer.Options().Print();
   minimizer.Minimize(); 
 
   const double *xs = minimizer.X();
   std::cout << "Minimum: f(" << xs[0] << "," << xs[1] << "): " 
        << rgf(xs) << std::endl;
 
   return 0;
}
