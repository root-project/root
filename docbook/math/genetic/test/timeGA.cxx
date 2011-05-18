#include "TH1.h"
#include "TF1.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"

#include "Math/WrappedMultiTF1.h"
#include "Math/Minimizer.h"
#include "Math/GeneticMinimizer.h"
#include "Math/Factory.h"
#include "HFitInterface.h"

#include "TMath.h"

#include <TStopwatch.h>

//#define DEBUG

// Quadratic background function
Double_t background(Double_t *x, Double_t *par) {
   return par[0] + par[1]*x[0];
}

// Gaussian Peak function
Double_t gaussianPeak(Double_t *x, Double_t *par) {
  return par[0]*TMath::Gaus(x[0],par[1], par[2]);
}

// Sum of background and peak function
Double_t fitFunction(Double_t *x, Double_t *par) {
  return background(x,par) + gaussianPeak(x,&par[2]) + gaussianPeak(x,&par[5]);
}

// We'll look for the minimum at 2 and 7
double par0[8] = { 1, 0.05, 10 , 2, 0.5 , 10 , 7 , 1. }; 
const int ndata = 10000; 
const double gAbsTolerance = 0.1; 
const int gVerbose = 0; 

using std::cout;
using std::endl;

int GAMinimize(ROOT::Math::IMultiGenFunction& chi2Func, double& xm1, double& xm2)
{
   // minimize the function
   ROOT::Math::GeneticMinimizer* min = new ROOT::Math::GeneticMinimizer();
   if (min == 0) { 
      cout << "Error creating minimizer " << endl;
      return -1;
   }

   // Set the parameters for the minimization.
   min->SetMaxFunctionCalls(1000000);
   min->SetMaxIterations(100000);
   min->SetTolerance(gAbsTolerance);
   min->SetPrintLevel(gVerbose);
   min->SetFunction(chi2Func); 
   min->SetParameters(100, 300);


   // initial values of the function
   double x0[8]; 
   std::copy(par0, par0 + 8, x0); 
   x0[3] = xm1; 
   x0[6] = xm2;

   for (unsigned int i = 0; i < chi2Func.NDim(); ++i) { 
#ifdef DEBUG
      cout << "set variable " << i << " to value " << x0[i] << endl;
#endif
      if ( i == 3 || i == 6 )
          min->SetLimitedVariable(i,"x" + ROOT::Math::Util::ToString(i),x0[i], 0.1,2,8);
       else
         min->SetLimitedVariable(i,"x" + ROOT::Math::Util::ToString(i),x0[i], 0.1,x0[i]-2,x0[i]+2);
   }


   // minimize
   if ( !min->Minimize() ) return 1;
   min->MinValue(); 

   // show the results
   cout << "Min values by GeneticMinimizer: " << min->X()[3] << "  " << min->X()[6] << endl;
   xm1 = min->X()[3];
   xm2 = min->X()[6];

   return 0;
}


int GAMinTutorial()
{
   double x1=0, x2=0;

   // create a TF1 from fit function and generate histogram 
   TF1 * fitFunc = new TF1("fitFunc",fitFunction,0,10,8);
   fitFunc->SetParameters(par0); 

   // Create a histogram filled with random data from TF1
   TH1D * h1 = new TH1D("h1","h1",100, 0, 10); 
   for (int i = 0; i < ndata; ++i) { 
      h1->Fill(fitFunc->GetRandom() ); 
   } 

   // perform the fit 
   ROOT::Fit::BinData d; 
   ROOT::Fit::FillData(d,h1); 
   ROOT::Math::WrappedMultiTF1 f(*fitFunc);

   // Create the function for fitting.
   ROOT::Fit::Chi2FCN<ROOT::Math::IMultiGenFunction> chi2Func(d,f); 

   // Look for an approximation with a Genetic Algorithm
   TStopwatch t;
   t.Start();
   GAMinimize(chi2Func, x1,x2);
   t.Stop();
   cout << "Time :\t " << t.RealTime() << " " << t.CpuTime() << endl;

   return 0; 
}

int main(int argc, char **argv)
{
   int status = GAMinTutorial();

   return status;
}
