/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #12
// 
// A Toy Monte Carlo study that perform cycles of
// event generation and fittting
//
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
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


void rf12_mcstudy()
{
  // (This is the pdf of tutorial rf02_composite)

  // Declare observable x
  RooRealVar x("x","x",0,10) ;

  // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their paramaters
  RooRealVar mean("mean","mean of gaussians",5,0,10) ;
  RooRealVar sigma1("sigma1","width of gaussians",0.5) ;
  RooRealVar sigma2("sigma2","width of gaussians",1) ;

  RooGaussian sig1("sig1","Signal component 1",x,mean,sigma1) ;  
  RooGaussian sig2("sig2","Signal component 2",x,mean,sigma2) ;  
  
  // Build Chebychev polynomial p.d.f.  
  RooRealVar a0("a0","a0",0.5,0.,1.) ;
  RooRealVar a1("a1","a1",-0.2,-1,1.) ;
  RooChebychev bkg("bkg","Background",x,RooArgSet(a0,a1)) ;

  // Sum the signal components into a composite signal p.d.f.
  RooRealVar sig1frac("sig1frac","fraction of component 1 in signal",0.8,0.,1.) ;
  RooAddPdf sig("sig","Signal",RooArgList(sig1,sig2),sig1frac) ;

  // Sum the composite signal and background 
  RooRealVar bkgfrac("bkgfrac","fraction of background",0.5,0.,1.) ;
  RooAddPdf  model("model","g1+g2+a",RooArgList(bkg,sig),bkgfrac) ;



  // Instantiate RooMCStudy manager on model with x as observable and given choice of fit options
  RooMCStudy mcstudy(model,x,FitOptions(Minos(kFALSE),PrintLevel(-1),Save(kTRUE))) ;

  // Generate and fit 200 samples of 100 events (suppress all messages)
  RooMsgService::instance().setGlobalKillBelow(RooMsgService::PROGRESS) ;
  mcstudy.generateAndFit(200,100) ;
  RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

  // Make plots of the distributions of mean, the error on mean and the pull of mean
  RooPlot* frame1 = mcstudy.plotParam(mean,FrameBins(40)) ;
  RooPlot* frame2 = mcstudy.plotError(mean,FrameBins(40)) ;
  RooPlot* frame3 = mcstudy.plotPull(mean,FrameBins(40),FitGauss(kTRUE)) ;

  // Draw all plots on a canvas
  TCanvas* c = new TCanvas("rf12_mcstudy","rf12_mcstudy",1200,400) ;
  c->Divide(3) ;
  c->cd(1) ; frame1->Draw() ;
  c->cd(2) ; frame2->Draw() ;
  c->cd(3) ; frame3->Draw() ;

}
