// test of multidimentional Integration 
// Calculates an intergal of a function 
// in 2,3,..., 8 dimensions 
// by using adaptive Genz Malik cubature
// and MonteCarlo methods:
// --PLAIN
// --VEGAS
// --MISER
//
// from
// IntegratorMultiDim class
// and GSLMCIntegrator class
//
// Compares time performance 
// for different dimensions
// draws a graph
//
// Author: Magdalena Slawinska
//

#include "TMath.h"
#include "TStopwatch.h"

#include <cmath>
#include <iostream>

#include "Math/Integrator.h"
#include "Math/Functor.h"
#include "Math/IFunction.h"
#include "Math/WrappedParamFunction.h"
#include "Math/AdaptiveIntegratorMultiDim.h"
#include "Math/IFunctionfwd.h"
#include "Math/GSLMCIntegrator.h"


// for graphical comparison of performance
#include "TGraph.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveLabel.h"
#include "TLegend.h"
#include "TH1.h"
#include "TCanvas.h"
//#include "TLegend.h"

bool showGraphics = true;

const int n = 3; //default dimensionality

Double_t Sum( const double* x, const double *p)
{
  double sum = 0.;
  for(int i = 0; i < p[0]; i++)
    sum += x[i];
  return  sum;
  
}

//multidim function to integrate
Double_t SimpleFun( const double* x, const double *p)
{
  double prod = 1.;
  for(int i = 0; i < p[0]; i++)
    prod *= TMath::Power(TMath::E(), -x[i]*x[i]);
  return  prod*Sum(x, p)*TMath::Sin(x[0]);
  
}

//singularity at (0,0, ..., 0)
Double_t SingularFun( const double* x, const double *p)
{
  
  double prod = 1.;
  for(int i = 0; i < p[0]; i++)
    prod *=    TMath::Cos(x[i]);
  return 1./((1-prod)*8*TMath::Power(TMath::Pi(), 3));

}


  // ################################################################
  //
  //      testing IntegratorMultiDim class 
  //
  // ################################################################

double integral_num(unsigned int dim, double* a, double* b, double* p)
{
  std::cout << "" << std::endl;
  std::cout << "testing IntegratorMultiDim class.." << std::endl;
  //std::cout << p[0] << std::endl;dimensionality

  TStopwatch timer; 
  timer.Start();
  ROOT::Math::WrappedParamFunction<> funptr1(&SimpleFun, dim, p, p+1);
  unsigned int nmax = (unsigned int) 1.E8; // apply cut off to avoid routine to explode
  //unsigned int size = (unsigned int) 1.E8; // apply cut off to avoid routine to explode
  unsigned int size = 0; // use default  value defined by nmax
  ROOT::Math::AdaptiveIntegratorMultiDim ig1(funptr1, 1.E-5, 1.E-5, nmax,size);
  //  std::cout << "1. integral= " << std::endl; 
  ig1.SetFunction(funptr1); 
  ig1.Integral(a, b);
  timer.Stop();
  std::cout.precision(12);
  std::cout << "result:  \t";
  std::cout << ig1.Result() << "\t" << "error: \t" << ig1.Error() << std::endl;
  std::cout << "Number of function evaluations: " << ig1.NEval() << std::endl;
  std::cout << "Time using IntegratorMultiDim: \t" << timer.RealTime() << std::endl; 
  std::cout << "------------------------------------" << std::endl;
  return timer.RealTime();
}

  // ################################################################
  //
  //      testing MCIntegrator class 
  //
  // ################################################################
double integral_MC(unsigned int dim, double* a, double* b, double* p)
{

  double timeVegas;
  std::cout << "" << std::endl; 
  std::cout << "testing GSLMCIntegrator class.." << std::endl;
  std::cout << "\t VEGAS.. " << std::endl;
  std::cout << "" << std::endl;

  TStopwatch timer; 
  //timer.Start();
  ROOT::Math::WrappedParamFunction<> funptr(&SimpleFun, dim, p, p+1);

  ROOT::Math::GSLMCIntegrator ig1(ROOT::Math::MCIntegration::kVEGAS);
  ig1.SetFunction(funptr);
  //ig1.SetMode(ROOT::Math::MCIntegration::kIMPORTANCE_ONLY);

  /*
  VegasParameters param;
  param.iterations = 2; 
  ig1.SetParameters(param); 
  */

  ig1.Integral(a, b); 
  timer.Stop();
  std::cout << "result: \t";
  std::cout << ig1.Result() << "\t" << "error: \t" << ig1.Error() << std::endl;
  std::cout << "sigma: \t" << ig1.Sigma();
  std::cout << "\t" << "chi2: \t" << ig1.ChiSqr() << std::endl;
  std::cout << std::endl;
  std::cout << "Time using GSLMCIntegrator::VEGAS :\t" << timer.RealTime() << std::endl; 
  //std::cout << "Number of function evaluations: " << ig1.Eval() << std::endl;
   //ig2_param.iterations = 1000; 
  std::cout << "" <<std::endl;
   std::cout << "------------------------------------" << std::endl;
  std::cout << "\t MISER.. " << std::endl;
  std::cout << "" << std::endl;
  timeVegas = timer.RealTime();

  timer.Start();
  ROOT::Math::GSLMCIntegrator ig2(ROOT::Math::MCIntegration::kMISER);
 
  ig2.SetFunction(funptr);

  
  //par.min_calls = 4*dim;
  //par.min_calls_per_bisection = 8*par.min_calls;

  
  //MiserParameters par(dim);
  //ig2.SetParameters(par);
  ig2.Integral(a, b);  
  timer.Stop();
  std::cout << "result: \t";
  std::cout << ig2.Result() << "\t" << "error: \t" << ig2.Error() << std::endl;
   
  std::cout << "Time using GSLMCIntegrator::MISER :\t" << timer.RealTime() << std::endl; 
  std::cout << "" << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "\t PLAIN.. " << std::endl;
  timer.Start();
  std::cout << "" << std::endl;
  ROOT::Math::GSLMCIntegrator ig3(ROOT::Math::MCIntegration::kPLAIN);
  ig3.SetFunction(funptr);
  ig3.Integral(a, b);
  timer.Stop();
  std::cout << "result: \t";
  std::cout << ig3.Result() << "\t" << "error: \t" << ig3.Error() << std::endl;
  std::cout << "Time using GSLMCIntegrator::PLAIN :\t" << timer.RealTime() << std::endl; 
 std::cout << "" << std::endl;  
 
 return timeVegas;
}

void performance()
{
  //dimensionality
  unsigned int Nmax = 9;
  unsigned int size = Nmax-1; 
  TH1D *num_performance = new TH1D("cubature", "", size, 1.5, Nmax+.5);
  TH1D *Vegas_performance = new TH1D("montecarlo", "", size, 1.5, Nmax+.5);

   num_performance->SetBinContent(1, 0.0);
   Vegas_performance->SetBinContent(1,0.0);
   for(unsigned int N = 2; N <=Nmax; N++)//dim
  {
    std::cout<< "*********************************************" << std::endl;
    std::cout<< "Number of dimensions: "<< N << std::endl;
  //integration limits
    double * a = new double[N]; 
    double * b = new double[N];
    double p[1];  
    p[0] = N;
    for (unsigned int i=0; i < N; i++)
    { 
	a[i] = -1.;//-TMath::Pi();
	b[i] = 1;//TMath::Pi();
    }
    //x[N] = N;
    num_performance->SetBinContent(N-1, integral_num(N, a, b, p));
    Vegas_performance->SetBinContent(N-1,integral_MC(N, a, b, p));
   }

   if ( showGraphics )
   {

      TCanvas * c1 = new TCanvas(); 
      c1->SetFillColor(kYellow-10);

      num_performance->SetBarWidth(0.45);
      num_performance->SetBarOffset(0.05);
      num_performance->SetFillColor(49);
      num_performance->SetStats(0);
      //num_performance->GetXaxis()->SetLimits(1.5, Nmax+0.5);
      num_performance->GetXaxis()->SetTitle("number of dimensions");
      num_performance->GetYaxis()->SetTitle("time [s]");
      num_performance->SetTitle("comparison of performance");
      TH1 *h1 = num_performance->DrawCopy("bar2");
      Vegas_performance->SetBarWidth(0.40);
      Vegas_performance->SetBarOffset(0.5);
      Vegas_performance->SetFillColor(kRed);
      TH1 *h2 =  Vegas_performance->DrawCopy("bar2,same");
      
      TLegend *legend = new TLegend(0.25,0.65,0.55,0.82);
      legend->AddEntry(h1,"Cubature","f");
      legend->AddEntry(h2,"MC Vegas","f");
      legend->Draw();
   }

   for (unsigned int i=1; i<=size; i++)
     std::cout << i << " " << num_performance->GetBinContent(i) << "\t" << Vegas_performance->GetBinContent(i)<<std::endl;
}


int main(int argc, char **argv)
{
   int status = 0;

   using std::cerr;
   using std::cout;
   using std::endl;

   if ( argc > 1 && argc != 2 )
   {
      cerr << "Usage: " << argv[0] << " [-ng]\n";
      cerr << "  where:\n";
      cerr << "     -ng : no graphics mode";
      cerr << endl;
      exit(1);
   }

   if ( argc == 2 && strcmp( argv[1], "-ng") == 0 ) 
   {
      showGraphics = false;
   }

   TApplication* theApp = 0;

   if ( showGraphics )
      theApp = new TApplication("App",&argc,argv);

   performance();

   if ( showGraphics )
   {
      theApp->Run();
      delete theApp;
      theApp = 0;
   }

   return status;
}
