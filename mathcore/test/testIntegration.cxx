#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/AllIntegrationTypes.h"
#include "Math/WrappedFunction.h"

double f(double x) { 
   return x; 
} 

double f2(const double * x) { 
   return x[0] + x[1]; 
} 


void testIntegration1D() { 

   ROOT::Math::WrappedFunction<> wf(f);
   ROOT::Math::Integrator ig(ROOT::Math::IntegrationOneDim::ADAPTIVESINGULAR); 
   ig.SetFunction(wf);
   double val = ig.Integral(0,1);
   std::cout << "integral result is " << val << std::endl;

   ROOT::Math::Integrator ig2(ROOT::Math::IntegrationOneDim::NONADAPTIVE); 
   ig2.SetFunction(wf);
   val = ig2.Integral(0,1);
   std::cout << "integral result is " << val << std::endl;

   ROOT::Math::Integrator ig3(wf, ROOT::Math::IntegrationOneDim::ADAPTIVE); 
   val = ig3.Integral(0,1);
   std::cout << "integral result is " << val << std::endl;



}

void testIntegrationMultiDim() { 

   ROOT::Math::WrappedMultiFunction<> wf(f2,2);
   double a[2] = {0,0};
   double b[2] = {1,1};

   ROOT::Math::IntegratorMultiDim ig(ROOT::Math::IntegrationMultiDim::ADAPTIVE); 
   ig.SetFunction(wf);
   double val = ig.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;

   ROOT::Math::IntegratorMultiDim ig2(ROOT::Math::IntegrationMultiDim::VEGAS); 
   ig2.SetFunction(wf);
   val = ig2.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;

   ROOT::Math::IntegratorMultiDim ig3(wf,ROOT::Math::IntegrationMultiDim::PLAIN); 
   val = ig3.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;

   ROOT::Math::IntegratorMultiDim ig4(wf,ROOT::Math::IntegrationMultiDim::MISER); 
   val = ig4.Integral(a,b);
   std::cout << "integral result is " << val << std::endl;



}

int  main() { 
   testIntegration1D();
   testIntegrationMultiDim();
}

