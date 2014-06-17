// Author: E. Benazera   6/2014

#include <functional>
#include <map>
#include <string>
#include <iostream>

#include "TH1.h"
#include "TF1.h"
#include "TStopwatch.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"
#include "TPaveLabel.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TFile.h"
#include "TMath.h"
#include "TROOT.h"

class expstats
{
public:
  expstats() {}
  ~expstats() {}

  void add_exp(const bool &succ,
	       const double &fmin,
	       const std::vector<double> &x,
	       const double &cputime,
	       const double &budget)
  {
    if (succ)
      ++_succs;
    else ++_fails;
    _fmin.push_back(fmin);
    _x.push_back(x);
    _cputime.push_back(cputime);
    _cpu_avg = std::accumulate(_cputime.begin(),_cputime.end(),0.0) / static_cast<double>(_cputime.size());
    _budget.push_back(budget);
    _budget_avg = std::accumulate(_budget.begin(),_budget.end(),0.0) / static_cast<double>(_budget.size());
  }
  
  expstats diff(const expstats &stats)
  {
    //TODO.
    return expstats();
  }

  std::ostream& print(std::ostream &out) const
  {
    out << "succs=" << _succs << " / fails=" << _fails << " / cpu_avg=" << _cpu_avg << " / budget_avg=" << _budget_avg << std::endl;
    return out;
  }
  
  int _fails = 0;
  int _succs = 0;
  std::vector<double> _fmin;
  std::vector<std::vector<double>> _x;
  std::vector<double> _cputime;
  double _cpu_avg = 0.0;
  std::vector<int> _budget;
  double _budget_avg = 0.0;

  // diff
  double _fdiff = 0;
  double _cputime_diff = 0;
  double _budget_diff = 0;
  int _isuccs = 0;
  int _ifails = 0;
};

std::ostream& operator<<(std::ostream &out, const expstats &stats)
{
  return stats.print(out);
}

typedef std::function<expstats (const std::string)> ExpFunc;

ExpFunc experiment_gauss_fit = [](const std::string &fitter)
{
  int n = 1000;
  gRandom = new TRandom3();
  TVirtualFitter::SetDefaultFitter(fitter.c_str() );
  std::string name = "h1_" + fitter; 
  TH1D * h1 = new TH1D(name.c_str(),"Chi2 Fit",100, -5, 5. );
  name = "h1bis_" + fitter; 
  TH1D * h1bis = new TH1D(name.c_str(),"Likelihood Fit",100, -5, 5. );
  for (int i = 0; i < n; ++i) { 
     double x = gRandom->Gaus(0,1); 
     h1->Fill( x );
     h1bis->Fill( x );
  }
  TStopwatch timer;
  timer.Start();
  TFitResultPtr r1 = h1->Fit("gaus","QS0");
  timer.Stop();
  Double_t cputime1 = timer.CpuTime();
  TStopwatch timer2;
  timer2.Start();
  TFitResultPtr r2 = h1bis->Fit("gaus","QLES0");
  timer2.Stop();
  Double_t cputime2 = timer2.CpuTime();
  
  expstats stats;
  stats.add_exp(r1->Status()==0,r1->MinFcnValue(),r1->Parameters(),cputime1,r1->NCalls());
  stats.add_exp(r2->Status()==0,r2->MinFcnValue(),r2->Parameters(),cputime2,r2->NCalls());
  std::cout << "gaus_fit stats: " << stats << std::endl;

  return stats;
};

// Quadratic background function
Double_t background(Double_t *x, Double_t *par) {
  return par[0] + par[1]*x[0] + par[2]*x[0]*x[0];
}

// Lorenzian Peak function
Double_t lorentzianPeak(Double_t *x, Double_t *par) {
  return (0.5*par[0]*par[1]/TMath::Pi()) / 
    TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}

// Sum of background and peak function
Double_t fitFunction(Double_t *x, Double_t *par) {
  return background(x,par) + lorentzianPeak(x,&par[3]);
}

ExpFunc experiment_lorentz_fit = [](const std::string &fitter)
{
  expstats stats;
  int npass = 20;
  gRandom = new TRandom3();
  TVirtualFitter::SetDefaultFitter(fitter.c_str());
  //ROOT::Fit::FitConfig::SetDefaultMinimizer(fitter);
  ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default("cmaes");
  opts.SetIntValue("lambda",100);
  
  TF1 *fitFcn = new TF1("fitFcn",fitFunction,0,3,6);
  fitFcn->SetNpx(200);
  fitFcn->SetParameters(1,1,1,6,.03,1);
  fitFcn->Update();
  std::string title = fitter + " fit bench";
  TH1 *histo = new TH1D(fitter.c_str(),title.c_str(),200,0,3);
  for (Int_t pass=0;pass<npass;pass++) {
    TStopwatch timer;
    if (pass%100 == 0) printf("pass : %d\n",pass);
    fitFcn->SetParameters(1,1,1,6,.03,1);
    for (Int_t i=0;i<5000;i++) {
      histo->Fill(fitFcn->GetRandom());
    }
    //histo->Print("all");
    timer.Start();
    TFitResultPtr r = histo->Fit(fitFcn,"QS0");  // from TH1.cxx: Q: quiet, 0: do not plot 
    timer.Stop();
    Double_t cputime = timer.CpuTime();
    stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
  }
  TStopwatch timer;
  timer.Start();
  TFitResultPtr r = histo->Fit(fitFcn,"QS0"); // E: use Minos
  timer.Stop();
  double cputime = timer.CpuTime();
  stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
  printf("%s, npass=%d  : RT=%7.3f s, Cpu=%7.3f s\n",fitter.c_str(),npass,timer.RealTime(),cputime);
  delete fitFcn;
  delete histo;
  std::cout << "lorentz_fit stats: " << stats << std::endl;
  return stats;
};

// Quadratic background function
Double_t gaus2D(Double_t *x, Double_t *par) {
   double t1 =   x[0] - par[1];
   double t2 =   x[1] - par[2];
   return par[0]* exp( - 0.5 * (  t1*t1/( par[3]*par[3]) + t2*t2  /( par[4]*par[4] )  ) ) ;    
}

// Sum of background and peak function
Double_t fitFunction2(Double_t *x, Double_t *par) {
  return gaus2D(x,par);
}

void fillHisto(int n,
	       TH2D *histo)
{ 
  gRandom = new TRandom3();
  for (int i = 0; i < n; ++i) { 
    double x = gRandom->Gaus(2,3);
    double y = gRandom->Gaus(-1,4);
    histo->Fill(x,y,1.);
  }
}

ExpFunc experiment_gauss2D_fit = [](const std::string &fitter)
{
  int npass = 0;
  int n = 100000;
  TStopwatch timer;
  TVirtualFitter::SetDefaultFitter(fitter.c_str());
  TF2 *fitFcn = new TF2("fitFcn",fitFunction2,-10,10,-10,10,5);
  fitFcn->SetParameters(100,0,0,2,7);
  fitFcn->Update();
  TH2D *histo = new TH2D("h2","2D Gauss",100,-10,10,100,-10,10);
  fillHisto(n,histo);
  timer.Start();
  TFitResultPtr r = histo->Fit("fitFcn","S0");
  timer.Stop();
  Double_t cputime = timer.CpuTime();
  printf("%s, npass=%d  : RT=%7.3f s, Cpu=%7.3f s\n",fitter.c_str(),npass,timer.RealTime(),cputime);
  delete fitFcn;
  delete histo;
  expstats stats;
  stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
  std::cout << "gauss2D_fit stats: " << stats << std::endl;
  return stats;
};

void run_experiments()
{
  std::map<std::string,ExpFunc> mexperiments;
  mexperiments.insert(std::pair<std::string,ExpFunc>("gauss_fit",experiment_gauss_fit));
  mexperiments.insert(std::pair<std::string,ExpFunc>("lorentz_fit",experiment_lorentz_fit));
  mexperiments.insert(std::pair<std::string,ExpFunc>("gauss2D_fit",experiment_gauss2D_fit));
  std::map<std::string,ExpFunc>::iterator mit = mexperiments.begin();
  while(mit!=mexperiments.end())
    {
      (*mit).second("acmaes");
      ++mit;
    }
  
}
