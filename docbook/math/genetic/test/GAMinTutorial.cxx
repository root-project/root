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


const double* Min2Minimize(ROOT::Math::IMultiGenFunction& chi2Func, double xm1, double xm2) { 

   // minimize the function
   ROOT::Math::Minimizer * min = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad");
   if (min == 0) { 
      cout << "Error creating minimizer " << endl;
      exit(-1);
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
   cout << "Minuit2Minimizer(" << xm1 << "," << xm2 << ")" << endl;
   cout << "chi2  min value " << minval << endl; 
   cout << " x minimum values " << min->X()[3] << "  " << min->X()[6] << endl;
   for (unsigned int i = 0; i < chi2Func.NDim(); ++i)
      cout << min->X()[i] << " ";
   cout << endl;

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
   
   h1->Draw(); 
   gPad->SetFillColor(kYellow-10);
   h1->SetFillColor(kYellow-5);
   gStyle->SetOptStat(0);

   // perform the fit 
   ROOT::Fit::BinData d; 
   ROOT::Fit::FillData(d,h1); 
   ROOT::Math::WrappedMultiTF1 f(*fitFunc);

   // Create the function for fitting.
   ROOT::Fit::Chi2FCN<ROOT::Math::IMultiGenFunction> chi2Func(d,f); 

   // Minimizer a first time with Minuit2
   const double* xmin1 = Min2Minimize(chi2Func, 0, 0);
   fitFunc->SetParameters(&xmin1[0]);
   fitFunc->SetLineColor(kBlue+3);
   TF1 * fitFunc0 = new TF1; 
   fitFunc->Copy(*fitFunc0);
   fitFunc0->Draw("same");

   // Look for an approximation with a Genetic Algorithm
   GAMinimize(chi2Func, x1,x2);

   // Minimize a second time with Minuit2 
   const double* xmin2 = Min2Minimize(chi2Func, x1, x2);
   fitFunc->SetParameters(&xmin2[0]);
   fitFunc->SetLineColor(kRed+3);
   fitFunc->DrawCopy("same");

   TLegend* legend = new TLegend(0.61,0.72,0.86,0.86);
   legend->AddEntry(h1, "Histogram Data","F");
   legend->AddEntry(fitFunc0, "Minuit only minimization");
   legend->AddEntry(fitFunc, "Minuit+Genetic minimization");
   legend->Draw();

   return 0; 
   // Min2Minimize will exit with a different value if there is any
   // error.
}

int main(int argc, char **argv)
{
   TApplication* theApp = new TApplication("App",&argc,argv);

   int status = GAMinTutorial();

   theApp->Run();

   delete theApp;
   theApp = 0;

   return status;
}
