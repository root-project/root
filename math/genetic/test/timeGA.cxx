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
int gVerbose = 0;

using std::cout;
using std::endl;

int GAMinimize(ROOT::Math::IMultiGenFunction& chi2Func, double& xm1, double& xm2)
{
   // minimize the function
   ROOT::Math::GeneticMinimizer* min = new ROOT::Math::GeneticMinimizer();
   if (min == 0) {
      std::cout << "Error creating minimizer " << std::endl;
      return -1;
   }

   // Set the parameters for the minimization.
   min->SetMaxFunctionCalls(1000000);
   min->SetMaxIterations(100000);
   min->SetTolerance(gAbsTolerance);
   min->SetPrintLevel(gVerbose);
   min->SetFunction(chi2Func);
   ROOT::Math::GeneticMinimizerParameters params; // construct with default values
   params.fNsteps = 100;  // increset number of steps top 100 (default is 40)
   min->SetParameters(params);


   // initial values of the function
   double x0[8];
   std::copy(par0, par0 + 8, x0);
   x0[3] = xm1;
   x0[6] = xm2;

   for (unsigned int i = 0; i < chi2Func.NDim(); ++i) {
#ifdef DEBUG
      std::cout << "set variable " << i << " to value " << x0[i] << std::endl;
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
   std::cout << "Min values by GeneticMinimizer: " << min->X()[3] << "  " << min->X()[6] << std::endl;
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
   std::cout << "Time :\t " << t.RealTime() << " " << t.CpuTime() << std::endl;

   return 0;
}

int main(int argc, char **argv)
{
  // Parse command line arguments
   for (Int_t i=1 ;  i<argc ; i++) {
      std::string arg = argv[i] ;
      if (arg == "-v") {
         gVerbose = 1;
      }
      if (arg == "-vv") {
         gVerbose = 3;
      }
      if (arg == "-h") {
         std::cout << "Usage: " << argv[0] << " [-v] [-vv]\n";
         std::cout << "  where:\n";
         std::cout << "     -v  : verbose mode\n";
         std::cout << "     -vv : very verbose mode\n";
         std::cout << std::endl;
         return -1;
      }
   }

   int status = GAMinTutorial();

   return status;
}
