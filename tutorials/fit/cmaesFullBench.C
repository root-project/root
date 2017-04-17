// Author: E. Benazera   6/2014

#include <functional>
#include <map>
#include <string>
#include <iostream>

#include "TH1.h"
#include "TF1.h"
#include "TH2D.h"
#include "TF2.h"
#include "TStopwatch.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"
#include "TPaveLabel.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TFile.h"
#include "TList.h"
#include "TMath.h"
#include "TROOT.h"

class expstats
{
public:
  expstats(const std::string &name,
	   const int &dim,
	   const int &lambda=-1):
    _name(name),_dim(dim),_lambda(lambda) {}
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
    _vsuccs.push_back(succ);
    _fmin.push_back(fmin);
    _x.push_back(x);
    _cputime.push_back(cputime);
    _cpu_avg = std::accumulate(_cputime.begin(),_cputime.end(),0.0) / static_cast<double>(_cputime.size());
    _cpu_std = stddev(_cputime,_cpu_avg);
    _budget.push_back(budget);
    _budget_avg = std::accumulate(_budget.begin(),_budget.end(),0.0) / static_cast<double>(_budget.size());
    _budget_std = stddev(_budget,_budget_avg);
  }

  void merge(const expstats &stats)
  {
    for (size_t i=0;i<stats._fmin.size();i++)
      {
	add_exp(stats._vsuccs.at(i),stats._fmin.at(i),stats._x.at(i),stats._cputime.at(i),stats._budget.at(i));
      }
  }
  
  void diff(const expstats &stats)
  {
    if (_fmin.size() != stats._fmin.size())
      {
	std::cout << "Error: diff requires same size sets: lhs=" << _fmin.size() << " / rhs=" << stats._fmin.size() << std::endl;
	return;
      }
    for (size_t i=0;i<_fmin.size();i++)
      {
	double fdiff = _fmin.at(i)-stats._fmin.at(i);
	double ffdiff = fabs(fdiff);
	// should compare to tolerance but unable to access it.
	if (ffdiff < 1e-1 || fdiff < -1e-1) // ftol appears to default to 0.01
	  ++_found;
	if (ffdiff > 1e-1 && fdiff < 0.0) // ftol as 0.01
	  ++_isuccs;
	else if (ffdiff < 1e-1)
	  ++_iequals;
	else ++_ifails;
	_fdiff.push_back(fdiff);
	_cputime_diff.push_back(_cputime.at(i)-stats._cputime.at(i));
	_cputime_ratio.push_back((_cputime.at(i)+1)/(stats._cputime.at(i)+1));
	_budget_diff.push_back(_budget.at(i)-stats._budget.at(i));
	_budget_ratio.push_back(_budget.at(i)/stats._budget.at(i));
      }
    _cputime_diff_avg = std::accumulate(_cputime_diff.begin(),_cputime_diff.end(),0.0) / static_cast<double>(_cputime_diff.size());
    _cputime_ratio_avg = std::accumulate(_cputime_ratio.begin(),_cputime_ratio.end(),0.0) / static_cast<double>(_cputime_ratio.size());
    _budget_diff_avg = std::accumulate(_budget_diff.begin(),_budget_diff.end(),0.0) / static_cast<double>(_budget_diff.size());
    _budget_ratio_avg = std::accumulate(_budget_ratio.begin(),_budget_ratio.end(),0.0) / static_cast<double>(_budget_ratio.size());
  }

  template <typename T> double stddev(const std::vector<T> &v, const double &avg)
  {
    double var = 0.0;
    for (size_t i=0;i<v.size();i++)
      var += (v.at(i)-avg)*(v.at(i)-avg);
    var /= static_cast<double>(v.size());
    return sqrt(var);
  }
  
  std::ostream& print(std::ostream &out) const
  {
    out << _name << " / dim=" << _dim << " / lambda=" << _lambda << " / succs=" << _succs << " / fails=" << _fails << " / cpu_avg=" << _cpu_avg << " / budget_avg=" << _budget_avg << std::endl;
    return out;
  }

  std::ostream& print_diff(std::ostream &out) const
  {
    out << _name << " / dim=" << _dim << " / lambda=" << _lambda << " / found=" << _found << "/" << _fdiff.size() << " / isuccs=" << _isuccs << " / ifails=" << _ifails << " / cpu_diff_avg=" << _cputime_diff_avg << " / cpu_ratio_avg=" << _cputime_ratio_avg << " / budget_diff_avg=" << _budget_diff_avg << " / budget_ratio_avg=" << _budget_ratio_avg << std::endl;
    for (size_t i=0;i<_fdiff.size();i++)
      {
	out << "#" << i << " - " << _name << ": " << "fdiff=" << _fdiff.at(i) << " / cputime_diff=" << _cputime_diff.at(i) << " / cputime_ratio=" << _cputime_ratio.at(i) << " / budget_diff=" << _budget_diff.at(i) << " / budget_ratio=" << _budget_ratio.at(i) << std::endl;
      }
    return out;
  }

  void print_avg_to_file() const
  {
    static std::string sep = "\t";
    static std::string ext = ".dat";
    std::ofstream fout(_name+ext,std::ofstream::out|std::ofstream::app);
    fout << "#dim\tlambda\tfound\tsuccs\tfails\tcpu_avg\tcpu_std\tbudget_avg\tbudget_std\tbest_fmin\tequal_fmin\tfail_fmin\n";
    fout << _dim << sep << _lambda << sep << _found << sep << _succs << sep << _fails << sep << _cpu_avg << sep << _cpu_std << sep << _budget_avg << sep << _budget_std << sep << _isuccs << sep << _iequals << sep << _ifails << std::endl;
    fout.close();
  }
  
  std::string _name;
  int _dim = 0;
  int _fails = 0;
  int _succs = 0;
  std::vector<bool> _vsuccs;
  std::vector<double> _fmin;
  std::vector<std::vector<double>> _x;
  std::vector<double> _cputime;
  double _cpu_avg = 0.0;
  double _cpu_std = 0.0;
  std::vector<int> _budget;
  double _budget_avg = 0.0;
  double _budget_std = 0.0;

  int _lambda = -1;
  
  // diff
  int _found = 0; /* number of times the correct minima was found. */
  std::vector<double> _fdiff;
  std::vector<double> _cputime_diff;
  std::vector<double> _cputime_ratio;
  std::vector<double> _budget_diff;
  std::vector<double> _budget_ratio;
  double _cputime_diff_avg = 0.0;
  double _cputime_ratio_avg = 1.0;
  double _budget_diff_avg = 0.0;
  double _budget_ratio_avg = 1.0;
  int _isuccs = 0; /* number of times the best f-min among the two algorithms, was found. */
  int _ifails = 0;
  int _iequals = 0; /* number of times the same fmin was found. */
};

std::ostream& operator<<(std::ostream &out, const expstats &stats)
{
  return stats.print(out);
}

typedef std::function<expstats (const std::string)> ExpFunc;

class experiment
{
public:
  experiment(const std::string &name)
    :_name(name)
  {
    ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default("cmaes");
    opts.SetRealValue("sigma",0.1);
  }

  void set_lambda(const int &lambda)
  {
    ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default("cmaes");
    opts.SetIntValue("lambda",lambda);
    _lambda = lambda;
  }

  void set_lscaling(const int &lscaling)
  {
    ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default("cmaes");
    opts.SetIntValue("lscaling",lscaling);
  }

  void set_restarts(const int &nrestarts)
  {
    ROOT::Math::IOptions &opts = ROOT::Math::MinimizerOptions::Default("cmaes");
    opts.SetIntValue("restarts",nrestarts);
  }
  
  virtual ~experiment() {}

  virtual void Setup() {}

  virtual void Cleanup() {}
  
  ExpFunc _ef;
  std::string _name;
  int _lambda = -1;
};

/*- gauss_fit -*/
class gauss_fit_e : public experiment
{
public:
  gauss_fit_e()
    :experiment("gauss_fit")
  {
    _ef = [this](const std::string &fitter)
      {
	std::string ename = "gauss_fit";
	
	TVirtualFitter::SetDefaultFitter(fitter.c_str() );
	TH1D * h1 = new TH1D("h1f1","Chi2 Fit",100, -5, 5. );
	TH1D * h1bis = new TH1D("h1f2","Likelihood Fit",100, -5, 5. );
	for (int i = 0; i < _n; ++i) { 
	  h1->Fill( _x.at(i) );
	  h1bis->Fill( _x.at(i) );
	}
	delete ((TF1 *)(gROOT->GetFunction("gaus")));
	TStopwatch timer;
	timer.Start();
	TFitResultPtr r1 = h1->Fit("gaus","VS0");
	timer.Stop();
	Double_t cputime1 = timer.CpuTime();
	delete ((TF1 *)(gROOT->GetFunction("gaus")));
	TStopwatch timer2;
	timer2.Start();
	TFitResultPtr r2 = h1bis->Fit("gaus","VLS0");
	timer2.Stop();
	Double_t cputime2 = timer2.CpuTime();
	
	delete h1;
	delete h1bis;
	
	expstats stats(ename,r1->NTotalParameters(),_lambda);
	stats.add_exp(r1->Status()==0,r1->MinFcnValue(),r1->Parameters(),cputime1,r1->NCalls());
	stats.add_exp(r2->Status()==0,r2->MinFcnValue(),r2->Parameters(),cputime2,r2->NCalls());
	std::cout << "gaus_fit stats: " << stats << std::endl;
	std::cout << "fmin1=" << r1->MinFcnValue() << std::endl;//" / fmin2=" << r2->MinFcnValue() << std::endl;
	return stats;
      };
  }

  ~gauss_fit_e() {}

  virtual void Setup()
  {
    std::cout << "setting up gauss_fit\n";
    for (int i=0;i<_n;i++)
      _x.push_back(_grandom.Gaus(0,1));
  }

  int _n = 1000;
  TRandom3 _grandom;
  std::vector<double> _x;
};
gauss_fit_e ggauss_fit;

/*- lorentz_fit -*/
class lorentz_fit_e : public experiment
{
public:
  lorentz_fit_e()
    :experiment("lorentz_fit")
  {
    _ef = [this](const std::string &fitter)
      {
	std::string title = "fit bench";
	std::string ename = "lorentz_fit";
	expstats stats(ename,6,_lambda);
	TVirtualFitter::SetDefaultFitter(fitter.c_str());
	
	TH1D *mhisto =new TH1D("fit",title.c_str(),200,0,3);
	for (Int_t pass=0;pass<_npass;pass++) {
	  TStopwatch timer;
	  TF1 *fitFcn = new TF1("fitFcn",lorentz_fit_e::fitFunction,0,3,6);
	  fitFcn->SetParameters(1,1,1,6,.03,1);
	  TH1D *histo = new TH1D("fit2",title.c_str(),200,0,3);
	  //histo->Print("all");
	  for (Int_t j=0;j<5000;j++)
	    {
	      histo->Fill(_xs.at(pass).at(j));
	      mhisto->Fill(_xs.at(pass).at(j));
	    }
	  /*timer.Start();
	  TFitResultPtr r = histo->Fit(fitFcn,"VS0");  // from TH1.cxx: Q: quiet, 0: do not plot 
	  timer.Stop();
	  Double_t cputime = timer.CpuTime();
	  stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());*/
	  delete histo;
	  delete fitFcn;
	}
	TStopwatch timer;
	timer.Start();
	TF1 *fitFcn = new TF1("fitFcn",lorentz_fit_e::fitFunction,0,3,6);
	fitFcn->SetParameters(1,1,1,6,.03,1);
	TFitResultPtr r = mhisto->Fit(fitFcn,"VS0");
	timer.Stop();
	double cputime = timer.CpuTime();
	stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
	printf("%s, npass=%d  : RT=%7.3f s, Cpu=%7.3f s\n",fitter.c_str(),_npass,timer.RealTime(),cputime);
	//std::cout << "lorentz_fit stats: " << stats << std::endl;
	delete mhisto;
	delete fitFcn;
	return stats;
      };
  }

  virtual ~lorentz_fit_e()
  {
    _xs.clear();
  }
  
  // Quadratic background function
  static Double_t background(Double_t *x, Double_t *par) {
    return par[0] + par[1]*x[0] + par[2]*x[0]*x[0];
  }

  // Lorenzian Peak function
  static Double_t lorentzianPeak(Double_t *x, Double_t *par) {
    return (0.5*par[0]*par[1]/TMath::Pi()) / 
      TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
  }

  // Sum of background and peak function
  static Double_t fitFunction(Double_t *x, Double_t *par) {
    return lorentz_fit_e::background(x,par) + lorentz_fit_e::lorentzianPeak(x,&par[3]);
  }

  virtual void Setup()
  {
    TF1 *fitFcn = new TF1("fitFcn",lorentz_fit_e::fitFunction,0,3,6);
    fitFcn->SetNpx(200);
    fitFcn->SetParameters(1,1,1,6,.03,1);
    fitFcn->Update();
    std::vector<double> x;
    for (int i=0;i<_npass;i++)
      {
	for (Int_t j=0;j<5000;j++) { 
	  x.push_back(fitFcn->GetRandom());
	}
	_xs.push_back(x);
      }
    delete fitFcn;
  }

  virtual void Cleanup()
  {
    _xs.clear();
  }
  
  int _npass = 20;
  //TF1 *_fitFcn;
  std::vector<std::vector<double> > _xs;
};
lorentz_fit_e glorentz_fit;

/*- gauss2D_fit -*/
class gauss2D_fit_e : public experiment
{
public:
  gauss2D_fit_e()
    :experiment("gauss2D_fit")
  {
    _ef = [this](const std::string &fitter)
      {
	std::string ename = "gauss2D_fit";
	int npass = 0;
	int n = 100000;
	TStopwatch timer;
	TVirtualFitter::SetDefaultFitter(fitter.c_str());
	TF2 *fitFcn = new TF2("fitFcn",gauss2D_fit_e::fitFunction2,-10,10,-10,10,5);
	fitFcn->SetParameters(100,0,0,2,7);
	fitFcn->Update();
	timer.Start();
	TFitResultPtr r = _histo->Fit("fitFcn","VS0");
	timer.Stop();
	Double_t cputime = timer.CpuTime();
	printf("%s : RT=%7.3f s, Cpu=%7.3f s\n",fitter.c_str(),timer.RealTime(),cputime);
	delete fitFcn;
	expstats stats(ename,r->NTotalParameters(),_lambda);
	stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
	std::cout << "gauss2D_fit stats: " << stats << std::endl;
	return stats;
      };
  }

  ~gauss2D_fit_e()
  {
    if (_histo)
      delete _histo;
  }
  
  // Quadratic background function
  static Double_t gaus2D(Double_t *x, Double_t *par) {
    double t1 =   x[0] - par[1];
    double t2 =   x[1] - par[2];
    return par[0]* exp( - 0.5 * (  t1*t1/( par[3]*par[3]) + t2*t2  /( par[4]*par[4] )  ) ) ;    
  }

  // Sum of background and peak function
  static Double_t fitFunction2(Double_t *x, Double_t *par) {
    return gaus2D(x,par);
  }

  static void fillHisto(int n,
			TH2D *histo)
  { 
    gRandom = new TRandom3();
    for (int i = 0; i < n; ++i) { 
      double x = gRandom->Gaus(2,3);
      double y = gRandom->Gaus(-1,4);
      histo->Fill(x,y,1.);
    }
  }

  virtual void Setup()
  {
    _histo = new TH2D("h2","2D Gauss",100,-10,10,100,-10,10);
    fillHisto(_n,_histo);
  }

  virtual void Cleanup()
  {
    if (_histo)
      delete _histo;
    _histo = nullptr;
  }

  int _n = 100000;
  TH2D *_histo = nullptr;
};
gauss2D_fit_e ggauss2D_fit;

/*- fit2a -*/
class fit2a_e : public experiment
{
public:
  fit2a_e()
    :experiment("fit2a")
  {
    _ef = [this](const std::string &fitter)
      {
	const Int_t npar = 15;
	TVirtualFitter::SetDefaultFitter(fitter.c_str());
	TF2 *f2 = new TF2("f2",fun2,-10,10,-10,10, npar);
	TH2F *h2 = new TH2F("h2","From f2",40,-10,10,40,-10,10);
	for (int i=0;i<_nentries;i++)
	  h2->Fill(_x.at(i),_y.at(i));

	Double_t f2params[npar] = {100,-3,3,-3,3,160,0,0.8,0,0.9,40,4,0.7,4,0.7};
	Float_t ratio = 4*_nentries/100000;
	f2params[ 0] *= ratio;
	f2params[ 5] *= ratio;
	f2params[10] *= ratio;
	f2->SetParameters(f2params);
	
	//Fit h2 with original function f2
	TStopwatch timer;
	timer.Start();
	TFitResultPtr r = h2->Fit("f2","SN0");
	timer.Stop();
	Double_t cputime = timer.CpuTime();
	expstats stats("fit2a",r->NTotalParameters(),_lambda);
	stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
	delete h2;
	delete f2;
	return stats;
      };
  }

  ~fit2a_e()
  {
    Cleanup();
  }
  
  static Double_t g2(Double_t *x, Double_t *par) {
    Double_t r1 = Double_t((x[0]-par[1])/par[2]);
    Double_t r2 = Double_t((x[1]-par[3])/par[4]);
    return par[0]*TMath::Exp(-0.5*(r1*r1+r2*r2));
  }
  
  static Double_t fun2(Double_t *x, Double_t *par) {
    Double_t *p1 = &par[0];
    Double_t *p2 = &par[5];
    Double_t *p3 = &par[10];
    Double_t result = g2(x,p1) + g2(x,p2) + g2(x,p3);
    return result;
  }

  virtual void Setup()
  {
    const Int_t npar = 15;
    Double_t f2params[npar] = {100,-3,3,-3,3,160,0,0.8,0,0.9,40,4,0.7,4,0.7};
    TF2 *f2 = new TF2("f2",fun2,-10,10,-10,10, npar);
    f2->SetParameters(f2params);
    _x = std::vector<double>(_nentries);
    _y = std::vector<double>(_nentries);
    for (int i=0;i<_nentries;i++)
      f2->GetRandom2(_x.at(i),_y.at(i));
    delete f2;
  }

  virtual void Cleanup()
  {
  }

  int _nentries = 100000;
  std::vector<double> _x,_y;
};
fit2a_e gfit2a;

/*- fit2dhist -*/
class fit2dhist_e : public experiment
{
public:
  fit2dhist_e()
    :experiment("fit2dhist")
  {
    _ef = [this](const std::string &fitter)
      {
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
	TF2 *func1 = new TF2("func",fit2dhist_e::my2Dfunc,xlow2,xup2,ylow2,yup2, 10);
	func1->SetParameters(iniParams);
	TF2 *func2 = new TF2("func",fit2dhist_e::my2Dfunc,xlow2,xup2,ylow2,yup2, 10);
	func2->SetParameters(iniParams);
	
	// fill up histograms.
	int n1 = 1000000;
	int n2 = 1000000;
	for (int i=0;i<n1;i++)
	  {
	    h1->Fill(_xr1.at(i),_yr1.at(i));
	    h2->Fill(_xr2.at(i),_yr2.at(i)); // since n1 == n2.
	  }
	
	// scale histograms to same heights (for fitting)
	double dx1 = (xup1-xlow1)/double(nbx1); 
	double dy1 = (yup1-ylow1)/double(nby1);
	double dx2 = (xup2-xlow2)/double(nbx2);
	double dy2 = (yup2-ylow2)/double(nby2);
	// scale histo 2 to scale of 1 
	h2->Sumw2();
	h2->Scale(  ( double(n1) * dx1 * dy1 )  / ( double(n2) * dx2 * dy2 ) );
	
	TStopwatch timer;
	timer.Start();
	TVirtualFitter::SetDefaultFitter(fitter.c_str());
	TFitResultPtr r1 = h1->Fit(func1,"VS0");
	timer.Stop();
	Double_t cputime1 = timer.CpuTime();
	TStopwatch timer2;
	timer2.Start();
	TFitResultPtr r2 = h2->Fit(func2,"VS0");
	timer2.Stop();
	Double_t cputime2 = timer2.CpuTime();
	expstats stats("fit2dhist",r1->NTotalParameters(),_lambda);
	stats.add_exp(r1->Status()==0,r1->MinFcnValue(),r1->Parameters(),cputime1,r1->NCalls());
	stats.add_exp(r2->Status()==0,r2->MinFcnValue(),r2->Parameters(),cputime2,r2->NCalls());
	delete h1;
	delete h2;
	delete func1;
	delete func2;
	return stats; 
      };
  }

  ~fit2dhist_e()
  {
    Cleanup();
  }

  static double gauss2D(double *x, double *par) {
    double z1 = double((x[0]-par[1])/par[2]);
    double z2 = double((x[1]-par[3])/par[4]);
    return par[0]*exp(-0.5*(z1*z1+z2*z2));
  }
  
  static double my2Dfunc(double *x, double *par) {
    return gauss2D(x,&par[0]) + gauss2D(x,&par[5]);
  }

  static void FillHisto2(std::vector<double> &xr, std::vector<double> &yr, int n, double * p)
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
    
    double x, y; 
    for (int i = 0; i < n; ++i) {
      // generate randoms with larger gaussians
      fit2dhist_e::_rndm.Rannor(x,y);
      
      double r = fit2dhist_e::_rndm.Rndm(1);
      if (r < w1) { 
	x = x*sx1 + mx1; 
	y = y*sy1 + my1; 
      }
      else { 
	x = x*sx2 + mx2; 
	y = y*sy2 + my2; 
      }
      xr.push_back(x);
      yr.push_back(y);
    }
  }

  void Setup()
  {
    double xlow1 = 0.; 
    double ylow1 = 0.; 
    double xup1 = 10.; 
    double yup1 = 10.; 
    double xlow2 = 5.; 
    double ylow2 = 5.; 
    double xup2 = 20.; 
    double yup2 = 20.; 
    double iniParams[10] = { 100, 6., 2., 7., 3, 100, 12., 3., 11., 2. };
    int n1 = 1000000;
    int n2 = 1000000;
    FillHisto2(_xr1,_yr1,n1,iniParams);
    FillHisto2(_xr2,_yr2,n2,iniParams);
  }

  void Cleanup()
  {
  }

  static TRandom3 _rndm;
  std::vector<double> _xr1,_yr1,_xr2,_yr2;
};
TRandom3 fit2dhist_e::_rndm = TRandom3();
fit2dhist_e gfit2dhist;

/*- combined_fit -*/
class combined_fit_e : public experiment
{
public:
  struct GlobalChi2 { 
    GlobalChi2(  ROOT::Math::IMultiGenFunction & f1,  
		 ROOT::Math::IMultiGenFunction & f2) : 
      fChi2_1(&f1), fChi2_2(&f2) {}
      
      // parameter vector is first background (in common 1 and 2) 
      // and then is signal (only in 2)
      double operator() (const double *par) const {
	int iparB[2] = {0,2};
	int iparSB[5] = {1,2,3,4,5};
	double p1[2];
	for (int i = 0; i < 2; ++i) p1[i] = par[iparB[i] ];
	
	double p2[5]; 
	for (int i = 0; i < 5; ++i) p2[i] = par[iparSB[i] ];
	
	return (*fChi2_1)(p1) + (*fChi2_2)(p2);
      } 
      
    const  ROOT::Math::IMultiGenFunction * fChi2_1;
    const  ROOT::Math::IMultiGenFunction * fChi2_2;
  };
  
  combined_fit_e()
    :experiment("combined_fit")
  {
    _ef = [this](const std::string &fitter)
      {
	// perform now global fit
	TF1 * fSB = new TF1("fSB","expo + gaus(2)",0,100);
	
	ROOT::Math::WrappedMultiTF1 wfB(*_fB,1);
	ROOT::Math::WrappedMultiTF1 wfSB(*fSB,1);
	
	ROOT::Fit::DataOptions opt; 
	ROOT::Fit::DataRange rangeB; 
	// set the data range
	rangeB.SetRange(10,90);
	ROOT::Fit::BinData dataB(opt,rangeB); 
	ROOT::Fit::FillData(dataB, _hB);
	
	ROOT::Fit::DataRange rangeSB; 
	rangeSB.SetRange(10,50);
	ROOT::Fit::BinData dataSB(opt,rangeSB); 
	ROOT::Fit::FillData(dataSB, _hSB);
	
	ROOT::Fit::Chi2Function chi2_B(dataB, wfB);
	ROOT::Fit::Chi2Function chi2_SB(dataSB, wfSB);
	
	GlobalChi2 globalChi2(chi2_B, chi2_SB);
	
	ROOT::Fit::Fitter rfitter;
	
	const int Npar = 6; 
	double par0[Npar] = { 5,5,-0.1,100, 30,10};
	
	// create before the parameter settings in order to fix or set range on them
	rfitter.Config().SetParamsSettings(6,par0);
	// fix 5-th parameter  
	//rfitter.Config().ParSettings(4).Fix();  // weird random crash, not yet understood: fitter tries to set variable 4 instead of 5, sometimes...
	// set limits on the third and 4-th parameter
	rfitter.Config().ParSettings(2).SetLimits(-10,-1.E-4);
	rfitter.Config().ParSettings(3).SetLimits(0,10000);
	rfitter.Config().ParSettings(3).SetStepSize(5);
	
	rfitter.Config().MinimizerOptions().SetPrintLevel(0);
	if (fitter == "Minuit2")
	  rfitter.Config().SetMinimizer("Minuit2","Migrad"); 
	else rfitter.Config().SetMinimizer("cmaes",fitter.c_str());
	
	// fit FCN function directly 
	// (specify optionally data size and flag to indicate that is a chi2 fit)
	TStopwatch timer;
	timer.Start();
	std::cout << "starting\n";
	rfitter.FitFCN(6,globalChi2,0,dataB.Size()+dataSB.Size(),true);
	timer.Stop();
	Double_t cputime = timer.CpuTime();
	ROOT::Fit::FitResult r = rfitter.Result();
	//result.Print(std::cout);
	
	delete fSB;
	expstats stats("combined",r.NTotalParameters(),_lambda);
	stats.add_exp(r.Status()==0,r.MinFcnValue(),r.Parameters(),cputime,r.NCalls());
	std::cout << "combined stats: " << stats << std::endl;
	std::cout << "fmin=" << r.MinFcnValue() << std::endl;
	return stats;
      };
  }

  ~combined_fit_e()
  {
    Cleanup();
  }

  virtual void Setup()
  {
    _hB = new TH1D("hB","histo B",100,0,100);
    _hSB = new TH1D("hSB","histo S+B",100, 0,100);
    _fB = new TF1("fB","expo",0,100);
    _fB->SetParameters(1,-0.05);
    _hB->FillRandom("fB");
    _fS = new TF1("fS","gaus",0,100);
    _fS->SetParameters(1,30,5);
    _hSB->FillRandom("fB",2000);
    _hSB->FillRandom("fS",1000);
  }

  virtual void Cleanup()
  {
    if (_hB)
      delete _hB;
    if (_hSB)
      delete _hSB;
    if (_fB)
      delete _fB;
    if (_fS)
      delete _fS;
  }

  TH1D *_hB = nullptr;
  TH1D *_hSB = nullptr;
  TF1 *_fB = nullptr;
  TF1 *_fS = nullptr;
};
combined_fit_e gcombined_fit;

/*- ex3d -*/
class ex3d_e : public experiment
{
public:
  ex3d_e()
    :experiment("ex3d")
  {
    _ef = [this](const std::string &fitter)
      {
	double ev = 0.1;
	
	// create a 3d binned data structure
	ROOT::Fit::BinData data(_n,3); 
	double xx[3];
	for(int i = 0; i < _n; ++i) {
	  xx[0] = _x[i]; 
	  xx[1] = _y[i]; 
	  xx[2] = _z[i]; 
	  // add the 3d-data coordinate, the predictor value (v[i])  and its errors
	  data.Add(xx, _v[i], ev); 
	}
	
	TF3 * f3 = new TF3("f3","[0] * sin(x) + [1] * cos(y) + [2] * z",0,10,0,10,0,10);
	f3->SetParameters(2,2,2);
	ROOT::Fit::Fitter rfitter;
	/*if (fitter.find("cmaes")!=std::string::npos)
	  rfitter.Config().SetMinimizer("cmaes","acmaes");*/
	if (fitter != "Minuit2")
	  rfitter.Config().SetMinimizer("cmaes",fitter.c_str());
	// wrapped the TF1 in a IParamMultiFunction interface for the Fitter class
	ROOT::Math::WrappedMultiTF1 wf(*f3,3);
	rfitter.SetFunction(wf); 
	TStopwatch timer;
	timer.Start();
	bool ret = rfitter.Fit(data);
	timer.Stop();
	Double_t cputime = timer.CpuTime();
	const ROOT::Fit::FitResult & res = rfitter.Result(); 
	if (ret) { 
	  // print result (should be around 1) 
	  res.Print(std::cout);
	  // copy all fit result info (values, chi2, etc..) in TF3
	  f3->SetFitResult(res);
	  // test fit p-value (chi2 probability)
	  double prob = res.Prob();
	  if (prob < 1.E-2) 
	    Error("exampleFit3D","Bad data fit - fit p-value is %f",prob);
	  else
	    std::cout << "Good fit : p-value  = " << prob << std::endl;
	}
	else Error("exampleFit3D","3D fit failed");
	expstats stats("example3D",res.NTotalParameters(),_lambda);
	stats.add_exp(res.Status()==0,res.MinFcnValue(),res.Parameters(),cputime,res.NCalls());
	delete f3;
	return stats;
      };
  }

  ~ex3d_e()
  {
  }

  virtual void Setup()
  {
    double ev = 0.1;
    TRandom2 r; 
    for (int i = 0; i < _n; ++i) { 
      _x.push_back(r.Uniform(0,10));
      _y.push_back(r.Uniform(0,10));
      _z.push_back(r.Uniform(0,10)); 
      _v.push_back(sin(_x[i] ) + cos(_y[i]) + _z[i] + r.Gaus(0,ev));         
    }
  }

  virtual void Cleanup()
  {
    _x.clear();
    _y.clear();
    _z.clear();
    _v.clear();
  }

  int _n = 1000;
  std::vector<double> _x;
  std::vector<double> _y;
  std::vector<double> _z;
  std::vector<double> _v;
};
ex3d_e gex3d;
  
/*- fit2 -*/
class fit2_e : public experiment
{
public:
  fit2_e()
    :experiment("fit2")
  {
    _ef = [this](const std::string &fitter)
      {	
	TVirtualFitter::SetDefaultFitter(fitter.c_str());	
	TF2 *f2 = new TF2("f2",fit2_e::fun22,-10,10,-10,10, _npar);
	
	Double_t f2params[15] = 
	{100,-3,3,-3,3,160,0,0.8,0,0.9,40,4,0.7,4,0.7};
	Float_t ratio = 4*_nentries/100000;
	f2params[ 0] *= ratio;
	f2params[ 5] *= ratio;
	f2params[10] *= ratio;
	f2->SetParameters(f2params);
	
	TH2F *h2 = new TH2F("h2","from f2",40,-10,10,40,-10,10);
	for (int i=0;i<_nentries;i++)
	  h2->Fill(_x.at(i),_y.at(i));

	//Fit h2 with original function f2
	TStopwatch timer;
	timer.Start();
	TFitResultPtr r = h2->Fit("f2","VS0");
	timer.Stop();
	Double_t cputime = timer.CpuTime();
	expstats stats("fit2",r->NTotalParameters(),_lambda);
	stats.add_exp(r->Status()==0,r->MinFcnValue(),r->Parameters(),cputime,r->NCalls());
	delete h2;
	delete f2;
	return stats;
      };
  }

  ~fit2_e()
  {
  }
  
  virtual void Setup()
  {
    TF2 *f2 = new TF2("f2",fit2_e::fun22,-10,10,-10,10, _npar);
    Double_t f2params[15] = 
      {100,-3,3,-3,3,160,0,0.8,0,0.9,40,4,0.7,4,0.7};
    f2->SetParameters(f2params);
    _x = std::vector<double>(_nentries);
    _y = std::vector<double>(_nentries);
    for (int i=0;i<_nentries;i++)
      f2->GetRandom2(_x.at(i),_y.at(i));
    delete f2;
  }
  
  static Double_t g22(Double_t *x, Double_t *par) {
    Double_t r1 = Double_t((x[0]-par[1])/par[2]);
    Double_t r2 = Double_t((x[1]-par[3])/par[4]);
    return par[0]*TMath::Exp(-0.5*(r1*r1+r2*r2));
  }
  
  static Double_t fun22(Double_t *x, Double_t *par) {
    Double_t *p1 = &par[0];
    Double_t *p2 = &par[5];
    Double_t *p3 = &par[10];
    Double_t result = fit2_e::g22(x,p1) + fit2_e::g22(x,p2) + fit2_e::g22(x,p3);
    return result;
  }

  Int_t _nentries = 100000;
  Int_t _npar = 15;
  std::vector<double> _x, _y;
};
fit2_e gfit2;

void cmaesFullBench(const int &n=100,
		    const int &lscaling=1)
{
  std::cout << "Proceeding with " << n << " runs on every problems\n";
  if (lscaling > 0)
    std::cout << "Linear scaling of parameters in ON\n";
  //std::vector<int> lambdas = {-1, 5, 10, 20, 40, 80, 160, 320, 640, 1280};
  std::vector<int> lambdas = {-1, 50, 200, -2, -3};
  std::vector<expstats> acmaes_stats;
  std::vector<expstats> minuit2_stats;
  std::map<std::string,experiment*> mexperiments;
  mexperiments.insert(std::pair<std::string,experiment*>(ggauss_fit._name,&ggauss_fit));
  mexperiments.insert(std::pair<std::string,experiment*>(glorentz_fit._name,&glorentz_fit));
  mexperiments.insert(std::pair<std::string,experiment*>(gfit2._name,&gfit2));
  mexperiments.insert(std::pair<std::string,experiment*>(ggauss2D_fit._name,&ggauss2D_fit));
  mexperiments.insert(std::pair<std::string,experiment*>(gfit2a._name,&gfit2a));
  mexperiments.insert(std::pair<std::string,experiment*>(gfit2dhist._name,&gfit2dhist));
  mexperiments.insert(std::pair<std::string,experiment*>(gcombined_fit._name,&gcombined_fit));
  mexperiments.insert(std::pair<std::string,experiment*>(gex3d._name,&gex3d));
  int nexp = mexperiments.size();
  int cn = 0;
  std::map<std::string,experiment*>::iterator mit = mexperiments.begin();
  while(mit!=mexperiments.end())
    {
      std::cout << "Running " << (*mit).first << std::endl;
      for (int i=0;i<n;i++)
	{
	  (*mit).second->Setup();
	  for (int j=0;j<(int)lambdas.size();j++)
	    {
	      std::string fitter_name = "acmaes";
	      if (lambdas.at(j) >= -1)
		(*mit).second->set_lambda(lambdas.at(j));
	      else if (lambdas.at(j) == -2)
		{
		  (*mit).second->set_lambda(-1);
		  (*mit).second->set_restarts(4);
		  fitter_name = "aipop";
		}
	      else if (lambdas.at(j) == -3)
		{
		  (*mit).second->set_lambda(-1);
		  (*mit).second->set_restarts(10);
		  fitter_name = "abipop";
		}
	      (*mit).second->set_lscaling(lscaling);
	      
	      if (i == 0)
		{
		  acmaes_stats.push_back((*mit).second->_ef(fitter_name));
		}
	      else
		{
		  acmaes_stats.at(cn*(lambdas.size())+j).merge((*mit).second->_ef(fitter_name));
		}
	    }
	  (*mit).second->set_lambda(-1); // N/A to Minuit2
	  if (i == 0)
	    minuit2_stats.push_back((*mit).second->_ef("Minuit2"));
	  else minuit2_stats.back().merge((*mit).second->_ef("Minuit2"));
	  (*mit).second->Cleanup();
	}
      ++cn;
      ++mit;
    }

  std::cout << "nexp=" << nexp << " / stats size=" << acmaes_stats.size() << std::endl;

  for (size_t i=0;i<minuit2_stats.size();i++)
    {
      for (size_t j=0;j<lambdas.size();j++)
	{
	  int k = i*lambdas.size()+j;
	  std::cout << "k=" << k << std::endl;
	  acmaes_stats.at(k).diff(minuit2_stats.at(i));
	  acmaes_stats.at(k).print_diff(std::cout);
	  acmaes_stats.at(k).print_avg_to_file();
	}
      minuit2_stats.at(i).diff(acmaes_stats.at(i*lambdas.size()+lambdas.size()-1));
      minuit2_stats.at(i).print_avg_to_file();
    }
}
