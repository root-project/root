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
  expstats(const std::string &name):
    _name(name) {}
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
  
  void diff(const expstats &stats)
  {
    if (_fmin.size() != stats._fmin.size())
      {
	std::cout << "Error: diff requires same size sets\n";
	return;
      }
    for (size_t i=0;i<_fmin.size();i++)
      {
	double fdiff = _fmin.at(i)-stats._fmin.at(i);
	double ffdiff = fabs(fdiff);
	// should compare to tolerance but unable to access it.
	if (ffdiff < 1e-2 || fdiff < 0.0) // ftol appears to default to 0.01
	  ++_found;
	if (fdiff <= 0.0)
	  ++_isuccs;
	else ++_ifails;
	_fdiff.push_back(fdiff);
	_cputime_diff.push_back(_cputime.at(i)-stats._cputime.at(i));
	_cputime_ratio.push_back(_cputime.at(i)/stats._cputime.at(i));
	_budget_diff.push_back(_budget.at(i)-stats._budget.at(i));
	_budget_ratio.push_back(_budget.at(i)/stats._budget.at(i));
      }
    _cputime_diff_avg = std::accumulate(_cputime_diff.begin(),_cputime_diff.end(),0.0) / static_cast<double>(_cputime_diff.size());
    _cputime_ratio_avg = std::accumulate(_cputime_ratio.begin(),_cputime_ratio.end(),0.0) / static_cast<double>(_cputime_ratio.size());
    _budget_diff_avg = std::accumulate(_budget_diff.begin(),_budget_diff.end(),0.0) / static_cast<double>(_budget_diff.size());
        _budget_ratio_avg = std::accumulate(_budget_ratio.begin(),_budget_ratio.end(),0.0) / static_cast<double>(_budget_ratio.size());
  }

  std::ostream& print(std::ostream &out) const
  {
    out << _name << " / succs=" << _succs << " / fails=" << _fails << " / cpu_avg=" << _cpu_avg << " / budget_avg=" << _budget_avg << std::endl;
    return out;
  }

  std::ostream& print_diff(std::ostream &out) const
  {
    out << _name << " / found=" << _found << "/" << _fdiff.size() << " / isuccs=" << _isuccs << " / ifails=" << _ifails << " / cpu_diff_avg=" << _cputime_diff_avg << " / cpu_ratio_avg=" << _cputime_ratio_avg << " / budget_diff_avg=" << _budget_diff_avg << " / budget_ratio_avg=" << _budget_ratio_avg << std::endl;
    for (size_t i=0;i<_fdiff.size();i++)
      {
	out << "#" << i << " - " << _name << ": " << "fdiff=" << _fdiff.at(i) << " / cputime_diff=" << _cputime_diff.at(i) << " / cputime_ratio=" << _cputime_ratio.at(i) << " / budget_diff=" << _budget_diff.at(i) << " / budget_ratio=" << _budget_ratio.at(i) << std::endl;
      }
    return out;
  }

  std::string _name;
  int _fails = 0;
  int _succs = 0;
  std::vector<double> _fmin;
  std::vector<std::vector<double>> _x;
  std::vector<double> _cputime;
  double _cpu_avg = 0.0;
  std::vector<int> _budget;
  double _budget_avg = 0.0;

  // diff
  int _found = 0;
  std::vector<double> _fdiff;
  std::vector<double> _cputime_diff;
  std::vector<double> _cputime_ratio;
  std::vector<double> _budget_diff;
  std::vector<double> _budget_ratio;
  double _cputime_diff_avg = 0.0;
  double _cputime_ratio_avg = 1.0;
  double _budget_diff_avg = 0.0;
  double _budget_ratio_avg = 1.0;
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
  std::string ename = "gauss_fit";
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
  
  expstats stats(ename);
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
  std::string ename = "lorentz_fit";
  expstats stats(ename);
  int npass = 20;
  gRandom = new TRandom3();
  TVirtualFitter::SetDefaultFitter(fitter.c_str());
  //ROOT::Fit::FitConfig::SetDefaultMinimizer(fitter);
  ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default("cmaes");
  opts.SetIntValue("lambda",500);
  
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
  opts.SetIntValue("lambda",-1);
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
  std::string ename = "gauss2D_fit";
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
  expstats stats(ename);
  stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
  std::cout << "gauss2D_fit stats: " << stats << std::endl;
  return stats;
};

Double_t g2(Double_t *x, Double_t *par) {
   Double_t r1 = Double_t((x[0]-par[1])/par[2]);
   Double_t r2 = Double_t((x[1]-par[3])/par[4]);
   return par[0]*TMath::Exp(-0.5*(r1*r1+r2*r2));
}   
Double_t fun2(Double_t *x, Double_t *par) {
   Double_t *p1 = &par[0];
   Double_t *p2 = &par[5];
   Double_t *p3 = &par[10];
   Double_t result = g2(x,p1) + g2(x,p2) + g2(x,p3);
   return result;
}

ExpFunc experiment_fit2a = [](const std::string &fitter)
{
  TVirtualFitter::SetDefaultFitter(fitter.c_str());

  //ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default(fitter);
  //ROOT::Math::IOptions *opts = ROOT::Math::MinimizerOptions::FindDefault(fitter);
  //opts.SetIntValue("lambda",100);
  //opts.SetNamedValue("fplot","fit2a.dat");
  
  const Int_t npar = 15;
  Double_t f2params[npar] = {100,-3,3,-3,3,160,0,0.8,0,0.9,40,4,0.7,4,0.7};
  TF2 *f2 = new TF2("f2",fun2,-10,10,-10,10, npar);
  f2->SetParameters(f2params);
  
  //Create an histogram and fill it randomly with f2
  TH2F *h2 = new TH2F("h2","From f2",40,-10,10,40,-10,10);
  Int_t nentries = 100000;
  h2->FillRandom("f2",nentries);
  //Fit h2 with original function f2
  Float_t ratio = 4*nentries/100000;
  f2params[ 0] *= ratio;
  f2params[ 5] *= ratio;
  f2params[10] *= ratio;
  f2->SetParameters(f2params);
  TStopwatch timer;
  timer.Start();
  TFitResultPtr r = h2->Fit("f2","SN0");
  timer.Stop();
  Double_t cputime = timer.CpuTime();
  delete f2;
  delete h2;  
  expstats stats("fit2a");
  stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
  return stats;
};

double gauss2D(double *x, double *par) {
   double z1 = double((x[0]-par[1])/par[2]);
   double z2 = double((x[1]-par[3])/par[4]);
   return par[0]*exp(-0.5*(z1*z1+z2*z2));
}   
double my2Dfunc(double *x, double *par) {
   return gauss2D(x,&par[0]) + gauss2D(x,&par[5]);
}

void fit2D(Int_t & nPar, Double_t * grad , Double_t &fval, Double_t *p, Int_t iflag,
	   TH2D *h1, TH2D *h2)
{
  TAxis *xaxis1  = h1->GetXaxis();
  TAxis *yaxis1  = h1->GetYaxis();
  TAxis *xaxis2  = h2->GetXaxis();
  TAxis *yaxis2  = h2->GetYaxis();

  int nbinX1 = h1->GetNbinsX(); 
  int nbinY1 = h1->GetNbinsY(); 
  int nbinX2 = h2->GetNbinsX(); 
  int nbinY2 = h2->GetNbinsY(); 

  double chi2 = 0; 
  double x[2]; 
  double tmp;
  Int_t npfits = 0;
  for (int ix = 1; ix <= nbinX1; ++ix) { 
    x[0] = xaxis1->GetBinCenter(ix);
    for (int iy = 1; iy <= nbinY1; ++iy) { 
      if ( h1->GetBinError(ix,iy) > 0 ) { 
        x[1] = yaxis1->GetBinCenter(iy);
        tmp = (h1->GetBinContent(ix,iy) - my2Dfunc(x,p))/h1->GetBinError(ix,iy);
        chi2 += tmp*tmp; 
        npfits++;
      }
    }
  }
  for (int ix = 1; ix <= nbinX2; ++ix) { 
     x[0] = xaxis2->GetBinCenter(ix);
    for (int iy = 1; iy <= nbinY2; ++iy) { 
      if ( h2->GetBinError(ix,iy) > 0 ) { 
        x[1] = yaxis2->GetBinCenter(iy);
        tmp = (h2->GetBinContent(ix,iy) - my2Dfunc(x,p))/h2->GetBinError(ix,iy);
        chi2 += tmp*tmp; 
        npfits++;
      }
    }
  }
  fval = chi2; 
}

void FillHisto2(TH2D * h, int n, double * p)
{ 
  const double mx1 = p[1]; 
  const double my1 = p[3]; 
  const double sx1 = p[2]; 
  const double sy1 = p[4]; 
  const double mx2 = p[6]; 
  const double my2 = p[8]; 
  const double sx2 = p[7]; 
  const double sy2 = p[9]; 
  //const double w1 = p[0]*sx1*sy1/(p[5]*sx2*sy2); 
  const double w1 = 0.5; 
  TRandom3 rndm;
  
  double x, y; 
  for (int i = 0; i < n; ++i) {
    // generate randoms with larger gaussians
    rndm.Rannor(x,y);

    double r = rndm.Rndm(1);
    if (r < w1) { 
      x = x*sx1 + mx1; 
      y = y*sy1 + my1; 
    }
    else { 
      x = x*sx2 + mx2; 
      y = y*sy2 + my2; 
    }      
    h->Fill(x,y);     
  }
}

ExpFunc experiment_fit2dhist = [](const std::string &fitter)
{
  expstats stats("fit2dhist");
  int nbx1 = 50;
  int nby1 = 50;
  int nbx2 = 50;
  int nby2 = 50;
  double xlow1 = 0.; 
  double ylow1 = 0.; 
  double xup1 = 10.; 
  double yup1 = 10.; 
  double xlow2 = 5.; 
  double ylow2 = 5.; 
  double xup2 = 20.; 
  double yup2 = 20.; 

  TH2D *h1 = new TH2D("h1","core",nbx1,xlow1,xup1,nby1,ylow1,yup1);
  TH2D *h2 = new TH2D("h2","tails",nbx2,xlow2,xup2,nby2,ylow2,yup2);

  double iniParams[10] = { 100, 6., 2., 7., 3, 100, 12., 3., 11., 2. };
  // create fit function
  TF2 * func = new TF2("func",my2Dfunc,xlow2,xup2,ylow2,yup2, 10);
  func->SetParameters(iniParams);

  // fill Histos
  int n1 = 1000000;
  int n2 = 1000000; 
  FillHisto2(h1,n1,iniParams);
  FillHisto2(h2,n2,iniParams);

  // scale histograms to same heights (for fitting)
  double dx1 = (xup1-xlow1)/double(nbx1); 
  double dy1 = (yup1-ylow1)/double(nby1);
  double dx2 = (xup2-xlow2)/double(nbx2);
  double dy2 = (yup2-ylow2)/double(nby2);
  // scale histo 2 to scale of 1 
  h2->Sumw2();
  h2->Scale(  ( double(n1) * dx1 * dy1 )  / ( double(n2) * dx2 * dy2 ) );

  /*bool global = false;
  if (option > 10) global = true;
  if (global) { 
    // fill data structure for fit (coordinates + values + errors) 
    std::cout << "Do global fit" << std::endl;
    // fit now all the function together

    //The default minimizer 
    TVirtualFitter::SetDefaultFitter(fitter.c_str());
    TVirtualFitter *vfit = TVirtualFitter::Fitter(0,10);
    for (int i = 0; i < 10; ++i) {  
      vfit->SetParameter(i, func->GetParName(i), func->GetParameter(i), 0.01, 0,0);
    }
    vfit->SetFCN(fit2D);

    double arglist[100];
    arglist[0] = 0;
    // set print level
    vfit->ExecuteCommand("SET PRINT",arglist,2);

    // minimize
    arglist[0] = 5000; // number of function calls
    arglist[1] = 0.01; // tolerance
    vfit->ExecuteCommand("MIGRAD",arglist,2); //TODO.

    //get result
    double minParams[10];
    double parErrors[10];
    for (int i = 0; i < 10; ++i) {  
      minParams[i] = vfit->GetParameter(i);
      parErrors[i] = vfit->GetParError(i);
    }
    double chi2, edm, errdef; 
    int nvpar, nparx;
    vfit->GetStats(chi2,edm,errdef,nvpar,nparx);

    func->SetParameters(minParams);
    func->SetParErrors(parErrors);
    func->SetChisquare(chi2);
    int ndf = npfits-nvpar; // beware: npfits global.
    func->SetNDF(ndf);

    // add to list of functions
    h1->GetListOfFunctions()->Add(func);
    h2->GetListOfFunctions()->Add(func);
    }*/
  //else {     
    // fit independently
  TStopwatch timer;
  timer.Start();
  TVirtualFitter::SetDefaultFitter(fitter.c_str());
  TFitResultPtr r1 = h1->Fit(func,"S0");
  timer.Stop();
  Double_t cputime1 = timer.CpuTime();
  TStopwatch timer2;
  timer2.Start();
  TFitResultPtr r2 = h2->Fit(func,"S0");
  timer2.Stop();
  Double_t cputime2 = timer2.CpuTime();
  
  stats.add_exp(r1->Status()==0,r1->MinFcnValue(),r1->Parameters(),cputime1,r1->NCalls());
  stats.add_exp(r2->Status()==0,r2->MinFcnValue(),r2->Parameters(),cputime2,r2->NCalls());
  
  //}	     

  delete h1;
  delete h2;

  
  return stats; 
  };

void run_experiments()
{
  std::vector<expstats> acmaes_stats;
  std::vector<expstats> minuit2_stats;
  std::map<std::string,ExpFunc> mexperiments;
  mexperiments.insert(std::pair<std::string,ExpFunc>("gauss_fit",experiment_gauss_fit));
  mexperiments.insert(std::pair<std::string,ExpFunc>("lorentz_fit",experiment_lorentz_fit));
  mexperiments.insert(std::pair<std::string,ExpFunc>("gauss2D_fit",experiment_gauss2D_fit));
  mexperiments.insert(std::pair<std::string,ExpFunc>("fit2a",experiment_fit2a));
  mexperiments.insert(std::pair<std::string,ExpFunc>("fit2dhist",experiment_fit2dhist));
  std::map<std::string,ExpFunc>::iterator mit = mexperiments.begin();
  while(mit!=mexperiments.end())
    {
      acmaes_stats.push_back((*mit).second("acmaes"));
      minuit2_stats.push_back((*mit).second("Minuit2"));
      ++mit;
    }

  for (size_t i=0;i<acmaes_stats.size();i++)
    {
      acmaes_stats.at(i).diff(minuit2_stats.at(i));
      acmaes_stats.at(i).print_diff(std::cout);
    }
  
}
