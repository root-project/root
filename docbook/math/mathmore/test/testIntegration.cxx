#include "Math/Polynomial.h"
#include "Math/Functor.h"
#include <iostream>


#ifdef HAVE_ROOTLIBS
#include "TStopwatch.h"
#include "TF1.h"
#include "TError.h"
#endif


#include "Math/GSLIntegrator.h"
// temp before having new Integrator class 
namespace ROOT { 
   namespace Math { 
      typedef GSLIntegrator Integrator; 
   }
}

const double ERRORLIMIT = 1E-8;

double exactIntegral ( const std::vector<double> & par, double a, double b) { 

  ROOT::Math::Polynomial *func = new ROOT::Math::Polynomial( par.size() +1);

  std::vector<double> p = par;
  p.push_back(0);
  p[0] = 0; 
  for (unsigned int i = 1; i < p.size() ; ++i) { 
    p[i] = par[i-1]/double(i); 
  }
  func->SetParameters(&p.front());

  return (*func)(b)-(*func)(a); 
}

double singularFunction(double x) { 
   return 1./x;
   if (x >= 0) 
      return 1./sqrt(x);
   else 
      return 1./sqrt(-x);
}


int testIntegration() {

  int status = 0;


#ifdef HAVE_ROOTLIBS
  gErrorIgnoreLevel = 5000;
#endif

  ROOT::Math::Polynomial * f = new ROOT::Math::Polynomial(2);

  std::vector<double> p(3);
  p[0] = 4;
  p[1] = 2;
  p[2] = 6;
  f->SetParameters(&p[0]);
  ROOT::Math::IGenFunction &func = *f; 


  double exactresult = exactIntegral(p, 0,3);
  std::cout << "Exact value " << exactresult << std::endl << std::endl; 


  //ROOT::Math::Integrator ig(func, 0.001, 0.01, 100 );
  ROOT::Math::GSLIntegrator ig(0.001, 0.01, 100 );
  ig.SetFunction(func);


  double value = ig.Integral( 0, 3); 
  // or ig.Integral(*f, 0, 10); if new function 

  std::cout.precision(20);

  std::cout << "Adaptive singular integration:" << std::endl;
  std::cout << "Return code " << ig.Status() << std::endl; 
  std::cout << "Result      " << value << " +/- " << ig.Error() << std::endl << std::endl; 
  status += fabs(exactresult-value) > ERRORLIMIT;

  
  // integrate again ADAPTIve, with different rule 
  ROOT::Math::GSLIntegrator ig2(ROOT::Math::Integration::kADAPTIVE, ROOT::Math::Integration::kGAUSS61, 0.001, 0.01, 100 );
  ig2.SetFunction(func);
  value = ig2.Integral(0, 3); 
  // or ig2.Integral(*f, 0, 10); if different function

  std::cout << "Adaptive Gauss61 integration:" << std::endl;
  std::cout << "Return code " << ig2.Status() << std::endl; 
  std::cout << "Result      " << value << " +/- " << ig2.Error() << std::endl << std::endl; 
  status += fabs(exactresult-value) > ERRORLIMIT;

  
  std::cout << "Testing SetFunction member function" << std::endl;
  ROOT::Math::Integrator ig3;
  ROOT::Math::Polynomial *pol = new ROOT::Math::Polynomial(2);
  
  pol->SetParameters(&p.front());
  ROOT::Math::IGenFunction &func2 = *pol; 
  ig3.SetFunction(func2);
  std::cout << "Result      " << ig3.Integral( 0, 3) << " +/- " << ig3.Error() << std::endl; 
  status += fabs(exactresult-ig3.Integral( 0, 3)) > ERRORLIMIT;

  // test error 
  //typedef double ( * FreeFunc ) ( double);

  std::cout << "Testing a singular function: 1/sqrt(x)" << std::endl;
  //ROOT::Math::WrappedFunction<FreeFunc> wf(&singularFunction); 
  ROOT::Math::Functor1D wf(&singularFunction);
  
  ig.SetFunction(wf); 
  double r = ig.Integral(0,1); 
  if (ig.Status() != 0) 
     std::cout << "Error integrating a singular function " << std::endl; 
  else 
     std::cout << "Result:(0,1]      " << r << " +/- " << ig.Error() << std::endl; 
  
  double singularPts[3] = {-1,0,1};
  std::vector<double> sp(singularPts, singularPts+3);
  double r2 = ig.Integral(sp); 
  if (ig.Status() != 0) 
     std::cout << "Error integrating a singular function using vector of points" << std::endl; 
  else 
     std::cout << "Result:[-1,1]      " << r2 << " +/- " << ig.Error() << std::endl; 


  std::vector<double> sp2(2); 
  sp2[0] = -1.; sp2[1] = -0.5; 
  double r3 = ig.Integral(sp2); 
  std::cout << "Result on [-1,-0.5] = " << r3 << std::endl;

  return status;
}

void  testIntegPerf(){

#ifdef HAVE_ROOTLIBS

   std::cout << "\n\n***************************************************************\n";
   std::cout << "Test integration performances....\n\n";


  ROOT::Math::Polynomial f1(2); 
  double p[3] = {2,3,4};
  f1.SetParameters(p);
  
  TStopwatch timer; 
  int n = 100000; 
  double x1 = 0; double x2 = 10; 
  double dx = (x2-x1)/double(n); 
  double a = -1;

  timer.Start(); 
  ROOT::Math::Integrator ig; ig.SetFunction(f1);
  double s1 = 0; 
  for (int i = 0; i < n; ++i) { 
     double x = x1 + dx*i; 
     s1+= ig.Integral(a,x);
  }
  timer.Stop(); 
  std::cout << "Time using ROOT::Math::Integrator :\t" << timer.RealTime() << std::endl; 
  int pr = std::cout.precision(18);  std::cout << s1 << std::endl;  std::cout.precision(pr);

  timer.Start(); 
  s1 = 0; 
  for (int i = 0; i < n; ++i) { 
     ROOT::Math::Integrator ig2; ig2.SetFunction(f1);
     double x = x1 + dx*i; 
     s1+= ig2.Integral(a,x);
  }
  timer.Stop(); 
  std::cout << "Time using ROOT::Math::Integrator(2):\t" << timer.RealTime() << std::endl; 
  pr = std::cout.precision(18);  std::cout << s1 << std::endl;  std::cout.precision(pr);


  TF1 f2("pol","pol2",0,10);
  f2.SetParameters(p);
  
  timer.Start(); 
  double s2 = 0; 
  for (int i = 0; i < n; ++i) { 
     double x = x1 + dx*i; 
     s2+= f2.Integral(a,x);
  }
  timer.Stop(); 
  std::cout << "Time using TF1::Integral :\t\t" << timer.RealTime() << std::endl; 
  pr = std::cout.precision(18);  std::cout << s1 << std::endl;  std::cout.precision(pr);

#endif  

}



int main() {

  int status = 0;

  status += testIntegration();
  testIntegPerf();

  return status;

}
