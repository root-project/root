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

#include "TApplication.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TStopwatch.h"

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
bool showGraphics = false;

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
   ROOT::Math::GeneticMinimizerParameters params; // construct with default values
   params.fNsteps = 100;  // increset number of steps top 100 (default is 40)
   params.fPopSize = 200;  // increset number of steps top 100 (default is 40)
   params.fSeed = 111;     // use a fixed seed

   min->SetParameters(params);



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
   double minval  = min->MinValue();

   // show the results
   cout << "--------------------------------------" << endl;
   cout << "GeneticMinimizer(" << xm1 << "," << xm2 << ") : ";
   cout << "chi2  min value  = " << minval;
   cout << " minimum values m1 = " << min->X()[3] << " m2 = " << min->X()[6] << endl;
   if (gVerbose) {
      std::cout << "All Fit parameters : ";
      for (unsigned int i = 0; i < chi2Func.NDim(); ++i)
         cout << "x(" << i << ") = " << min->X()[i] << " ";
      cout << endl;
   }
   xm1 = min->X()[3];
   xm2 = min->X()[6];

   return 0;
}


const double* Min2Minimize(ROOT::Math::IMultiGenFunction& chi2Func, double &xm1, double &xm2) {

   // minimize the function
   ROOT::Math::Minimizer * min = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
   if (min == 0) {
      min = ROOT::Math::Factory::CreateMinimizer("Minuit", "Migrad");
      if (min == 0) {
         cout << "Error creating minimizer " << endl;
         exit(-1);
      }
   }

   // Set the minimizer options
   min->SetMaxFunctionCalls(1000000);
   min->SetMaxIterations(100000);
   min->SetTolerance(gAbsTolerance);
   min->SetPrintLevel(gVerbose);
   min->SetFunction(chi2Func);

   // initial values of the function
   double x0[8];
   std::copy(par0, par0 + 8, x0);
   x0[3] = xm1;
   x0[6] = xm2;

   for (unsigned int i = 0; i < chi2Func.NDim(); ++i) {
#ifdef DEBUG
      cout << "set variable " << i << " to value " << x0[i] << endl;
#endif
      min->SetVariable(i,"x" + ROOT::Math::Util::ToString(i),x0[i], 0.1);
   }

   // minimize
   if ( !min->Minimize() ) exit(1);
   double minval = min->MinValue();

   // show the results
   cout << "--------------------------------------" << endl;
   cout << "Minuit2Minimizer(" << xm1 << "," << xm2 << ") : ";
   cout << "chi2  min value  = " << minval;
   cout << " minimum values - m1 = " << min->X()[3] << " m2 = " << min->X()[6] << endl;
   if (gVerbose) {
      std::cout << "All Fit parameters : ";
      for (unsigned int i = 0; i < chi2Func.NDim(); ++i)
         cout << "x(" << i << ") = " << min->X()[i] << " ";
      cout << endl;
   }
   xm1 = min->X()[3];
   xm2 = min->X()[6];

   return min->X();
}

int GAMinTutorial()
{
   double x1=0, x2=0;


   // create a TF1 from fit function and generate histogram
   TF1 * fitFunc = new TF1("fitFunc",fitFunction,0,10,8);
   fitFunc->SetParameters(par0);

   // Create a histogram filled with random data from TF1
   TH1D * h1 = new TH1D("h1","",100, 0, 10);
   for (int i = 0; i < ndata; ++i) {
      h1->Fill(fitFunc->GetRandom() );
   }

   // perform the fit
   ROOT::Fit::BinData d;
   ROOT::Fit::FillData(d,h1);
   ROOT::Math::WrappedMultiTF1 f(*fitFunc);

   // Create the function for fitting.
   ROOT::Fit::Chi2FCN<ROOT::Math::IMultiGenFunction> chi2Func(d,f);

   // Minimizer a first time with Minuit2
   TStopwatch t;
   t.Start();
   const double* xmin1 = Min2Minimize(chi2Func, x1, x2);
   t.Stop();
   cout << "Minuit2 minimization Time :\t " << t.RealTime() << endl;


   fitFunc->SetParameters(&xmin1[0]);
   fitFunc->SetLineColor(kBlue+3);
   TF1 * fitFunc0 = new TF1;
   fitFunc->Copy(*fitFunc0);

   if (showGraphics) {
      h1->Draw();
      gPad->SetFillColor(kYellow-10);
      h1->SetFillColor(kYellow-5);
      gStyle->SetOptStat(0);
      fitFunc0->Draw("same");
   }



   // Look for an approximation with a Genetic Algorithm
   t.Start();
   GAMinimize(chi2Func, x1,x2);
   t.Stop();
   cout << "Genetic minimization Time :\t " << t.RealTime() << endl;


   // Minimize a second time with Minuit2
   t.Start();
   const double* xmin2 = Min2Minimize(chi2Func, x1, x2);
   t.Stop();
   cout << "Second Minuit2 minimization Time :\t " << t.RealTime() << endl;

   fitFunc->SetParameters(&xmin2[0]);
   if (showGraphics) {
      fitFunc->SetLineColor(kRed+3);
      fitFunc->DrawCopy("same");

      TLegend* legend = new TLegend(0.61,0.72,0.86,0.86);
      legend->AddEntry(h1, "Histogram Data","F");
      legend->AddEntry(fitFunc0, "Minuit only minimization");
      legend->AddEntry(fitFunc, "Minuit+Genetic minimization");
      legend->Draw();

      gPad->Update();
   }

   return 0;
   // Min2Minimize will exit with a different value if there is any
   // error.
}

int main(int argc, char **argv)
{
  // Parse command line arguments
   for (Int_t i=1 ;  i<argc ; i++) {
      std::string arg = argv[i] ;
      if (arg == "-g") {
         showGraphics = true;
      }
      if (arg == "-v") {
         gVerbose = 1;
      }
      if (arg == "-vv") {
         gVerbose = 3;
      }
      if (arg == "-h") {
         std::cout << "Usage: " << argv[0] << " [-g] [-v]\n";
         std::cout << "  where:\n";
         std::cout << "     -g : graphics mode\n";
         std::cout << "     -v  : verbose mode\n";
         std::cout << "     -vv : very verbose mode\n";
         std::cout << std::endl;
         return -1;
      }
   }
   TApplication* theApp = 0;
   if (showGraphics)
      theApp = new TApplication("App",&argc,argv);

   int status = GAMinTutorial();

   if (showGraphics) {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return status;
}
