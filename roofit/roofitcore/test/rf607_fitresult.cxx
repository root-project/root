//////////////////////////////////////////////////////////////////////////
//
// 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #607
// 
// Demonstration of options of the RooFitResult class
//
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
#include "RooAddPdf.h"
#include "RooChebychev.h"
#include "RooFitResult.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "TFile.h"
#include "TStyle.h"
#include "TH2.h"

using namespace RooFit ;


class TestBasic607 : public RooFitTestUnit
{
public: 
  TestBasic607(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Fit Result functionality",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C r e a t e   p d f ,   d a t a
  // --------------------------------

  // Declare observable x
  RooRealVar x("x","x",0,10) ;

  // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
  RooRealVar mean("mean","mean of gaussians",5,-10,10) ;
  RooRealVar sigma1("sigma1","width of gaussians",0.5,0.1,10) ;
  RooRealVar sigma2("sigma2","width of gaussians",1,0.1,10) ;

  RooGaussian sig1("sig1","Signal component 1",x,mean,sigma1) ;  
  RooGaussian sig2("sig2","Signal component 2",x,mean,sigma2) ;  
  
  // Build Chebychev polynomial p.d.f.  
  RooRealVar a0("a0","a0",0.5,0.,1.) ;
  RooRealVar a1("a1","a1",-0.2) ;
  RooChebychev bkg("bkg","Background",x,RooArgSet(a0,a1)) ;

  // Sum the signal components into a composite signal p.d.f.
  RooRealVar sig1frac("sig1frac","fraction of component 1 in signal",0.8,0.,1.) ;
  RooAddPdf sig("sig","Signal",RooArgList(sig1,sig2),sig1frac) ;

  // Sum the composite signal and background 
  RooRealVar bkgfrac("bkgfrac","fraction of background",0.5,0.,1.) ;
  RooAddPdf  model("model","g1+g2+a",RooArgList(bkg,sig),bkgfrac) ;

  // Generate 1000 events
  RooDataSet* data = model.generate(x,1000) ;



  // F i t   p d f   t o   d a t a ,   s a v e   f i t r e s u l t 
  // -------------------------------------------------------------

  // Perform fit and save result
  RooFitResult* r = model.fitTo(*data,Save()) ;


  // V i s u a l i z e   c o r r e l a t i o n   m a t r i x
  // -------------------------------------------------------

  // Construct 2D color plot of correlation matrix
  gStyle->SetOptStat(0) ;
  gStyle->SetPalette(1) ;
  TH2* hcorr = r->correlationHist() ;


  // Sample dataset with parameter values according to distribution
  // of covariance matrix of fit result
  RooDataSet randPars("randPars","randPars",r->floatParsFinal()) ;
  for (Int_t i=0 ; i<10000 ; i++) {
    randPars.add(r->randomizePars()) ;    
  }

  // make histogram of 2D distribution in sigma1 vs sig1frac
  TH1* hhrand = randPars.createHistogram("hhrand",sigma1,Binning(35,0.25,0.65),YVar(sig1frac,Binning(35,0.3,1.1))) ;

  regTH(hcorr,"rf607_hcorr") ;
  regTH(hhrand,"rf607_hhand") ;

  delete data ;
  delete r ;

  return kTRUE ;

  }
} ;
