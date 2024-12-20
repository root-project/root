/// \file
/// \ingroup tutorial_r
/// \notebook -nodraw
/// Numerical integration using R passing the function from ROOT
///
/// \macro_code
///
/// \author

#include<TMath.h>
#include<TRInterface.h>
#include<Math/Integrator.h>
#include<TF1.h>

//To integrate using R the function must be vectorized
//The idea is just to receive a vector like an argument,to evaluate
//every element saving the result in another vector
//and return the resultant vector.
std::vector<Double_t>  BreitWignerVectorized(std::vector<Double_t> xx)
{
   std::vector<Double_t> result(xx.size());
   for(Int_t i=0;i<xx.size();i++)
   {
      result[i]=TMath::BreitWigner(xx[i]);
   }
   return result;
}

double BreitWignerWrap( double x){
   return TMath::BreitWigner(x);
}


void Integration()
{
   ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();

   r["BreitWigner"]=ROOT::R::TRFunctionExport(BreitWignerVectorized);

   Double_t value=r.Eval("integrate(BreitWigner, lower = -2, upper = 2)$value");

   std::cout.precision(18);
   std::cout<<"Integral of the BreitWigner Function in the interval [-2, 2] R        = "<<value<<std::endl;


   ROOT::Math::WrappedFunction<> wf(BreitWignerWrap);
   ROOT::Math::Integrator i(wf);
   value=i.Integral(-2,2);
   std::cout<<"Integral of the BreitWigner Function in the interval [-2, 2] MathMore = "<<value<<std::endl;


   TF1 f1("BreitWigner","BreitWignerWrap(x)");
   value=f1.Integral(-2,2);
   std::cout<<"Integral of the BreitWigner Function in the interval [-2, 2] TF1      = "<<value<<std::endl;

   // infinite limits
   value=r.Eval("integrate(BreitWigner, lower = -Inf, upper = Inf)$value");
   std::cout<<"Integral of BreitWigner Function in the interval [-Inf, Inf] R    = "<<value<<std::endl;
}
