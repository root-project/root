/////////////////////////////////////////////////////////////////////////
//
// 'VALIDATION AND MC STUDIES' RooFit tutorial macro #802
//
// RooMCStudy: using separate fit and generator models, using the chi^2 calculator model
//
//
// 07/2008 - Wouter Verkerke
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooMCStudy.h"
#include "RooChi2MCSModule.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TDirectory.h"

using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic802 : public RooFitTestUnit
{
public:
  TestBasic802(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("MC Study with chi^2 calculator",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C r e a t e   m o d e l
  // -----------------------

  // Observables, parameters
  RooRealVar x("x","x",-10,10) ;
  x.setBins(10) ;
  RooRealVar mean("mean","mean of gaussian",0) ;
  RooRealVar sigma("sigma","width of gaussian",5,1,10) ;

  // Create Gaussian pdf
  RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;



  // C r e a t e   m a n a g e r  w i t h   c h i ^ 2   a d d - o n   m o d u l e
  // ----------------------------------------------------------------------------

  // Create study manager for binned likelihood fits of a Gaussian pdf in 10 bins
  RooMCStudy* mcs = new RooMCStudy(gauss,x,Silence(),Binned()) ;

  // Add chi^2 calculator module to mcs
  RooChi2MCSModule chi2mod ;
  mcs->addModule(chi2mod) ;

  // Generate 200 samples of 1000 events
  mcs->generateAndFit(200,1000) ;

  // Fill histograms with distributions chi2 and prob(chi2,ndf) that
  // are calculated by RooChiMCSModule

  RooRealVar* chi2 = (RooRealVar*) mcs->fitParDataSet().get()->find("chi2") ;
  RooRealVar* prob = (RooRealVar*) mcs->fitParDataSet().get()->find("prob") ;

  TH1* h_chi2  = new TH1F("h_chi2","",40,0,20) ;
  TH1* h_prob  = new TH1F("h_prob","",40,0,1) ;

  mcs->fitParDataSet().fillHistogram(h_chi2,*chi2) ;
  mcs->fitParDataSet().fillHistogram(h_prob,*prob) ;



  // C r e a t e   m a n a g e r  w i t h   s e p a r a t e   f i t   m o d e l
  // ----------------------------------------------------------------------------

  // Create alternate pdf with shifted mean
  RooRealVar mean2("mean2","mean of gaussian 2",0.5) ;
  RooGaussian gauss2("gauss2","gaussian PDF2",x,mean2,sigma) ;

  // Create study manager with separate generation and fit model. This configuration
  // is set up to generate bad fits as the fit and generator model have different means
  // and the mean parameter is not floating in the fit
  RooMCStudy* mcs2 = new RooMCStudy(gauss2,x,FitModel(gauss),Silence(),Binned()) ;

  // Add chi^2 calculator module to mcs
  RooChi2MCSModule chi2mod2 ;
  mcs2->addModule(chi2mod2) ;

  // Generate 200 samples of 1000 events
  mcs2->generateAndFit(200,1000) ;

  // Fill histograms with distributions chi2 and prob(chi2,ndf) that
  // are calculated by RooChiMCSModule

  TH1* h2_chi2  = new TH1F("h2_chi2","",40,0,20) ;
  TH1* h2_prob  = new TH1F("h2_prob","",40,0,1) ;

  mcs2->fitParDataSet().fillHistogram(h2_chi2,*chi2) ;
  mcs2->fitParDataSet().fillHistogram(h2_prob,*prob) ;

  h_chi2->SetLineColor(kRed) ;
  h_prob->SetLineColor(kRed) ;

  regTH(h_chi2,"rf802_hist_chi2") ;
  regTH(h2_chi2,"rf802_hist2_chi2") ;
  regTH(h_prob,"rf802_hist_prob") ;
  regTH(h2_prob,"rf802_hist2_prob") ;

  delete mcs ;
  delete mcs2 ;

  return true ;
  }
} ;
