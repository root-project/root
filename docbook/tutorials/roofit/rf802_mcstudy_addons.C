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
#include "RooConstVar.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooMCStudy.h"
#include "RooChi2MCSModule.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TH1.h"
#include "TDirectory.h"

using namespace RooFit ;


void rf802_mcstudy_addons()
{

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

  // Generate 1000 samples of 1000 events
  mcs->generateAndFit(2000,1000) ;
  
  // Fill histograms with distributions chi2 and prob(chi2,ndf) that
  // are calculated by RooChiMCSModule
  TH1* hist_chi2 = mcs->fitParDataSet().createHistogram("chi2") ; 
  TH1* hist_prob = mcs->fitParDataSet().createHistogram("prob") ;   



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

  // Generate 1000 samples of 1000 events
  mcs2->generateAndFit(2000,1000) ;
  
  // Fill histograms with distributions chi2 and prob(chi2,ndf) that
  // are calculated by RooChiMCSModule
  TH1* hist2_chi2 = mcs2->fitParDataSet().createHistogram("chi2") ; 
  TH1* hist2_prob = mcs2->fitParDataSet().createHistogram("prob") ;   
  hist2_chi2->SetLineColor(kRed) ;
  hist2_prob->SetLineColor(kRed) ;

  

  TCanvas* c = new TCanvas("rf802_mcstudy_addons","rf802_mcstudy_addons",800,400) ;
  c->Divide(2) ;
  c->cd(1) ; gPad->SetLeftMargin(0.15) ; hist_chi2->GetYaxis()->SetTitleOffset(1.4) ; hist_chi2->Draw() ; hist2_chi2->Draw("esame") ;
  c->cd(2) ; gPad->SetLeftMargin(0.15) ; hist_prob->GetYaxis()->SetTitleOffset(1.4) ; hist_prob->Draw() ; hist2_prob->Draw("esame") ;


  
  // Make RooMCStudy object available on command line after
  // macro finishes
  gDirectory->Add(mcs) ;
}
