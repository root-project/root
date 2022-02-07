//////////////////////////////////////////////////////////////////////////
//
// 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #602
//
// Setting up a binning chi^2 fit
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
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooChi2Var.h"
#include "RooMinimizer.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


class TestBasic602 : public RooFitTestUnit
{
public:
  TestBasic602(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Chi2 minimization",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // S e t u p   m o d e l
  // ---------------------

  // Declare observable x
  RooRealVar x("x","x",0,10) ;

  // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
  RooRealVar mean("mean","mean of gaussians",5) ;
  RooRealVar sigma1("sigma1","width of gaussians",0.5) ;
  RooRealVar sigma2("sigma2","width of gaussians",1) ;

  RooGaussian sig1("sig1","Signal component 1",x,mean,sigma1) ;
  RooGaussian sig2("sig2","Signal component 2",x,mean,sigma2) ;

  // Build Chebychev polynomial p.d.f.
  RooRealVar a0("a0","a0",0.5,0.,1.) ;
  RooRealVar a1("a1","a1",-0.2,0.,1.) ;
  RooChebychev bkg("bkg","Background",x,RooArgSet(a0,a1)) ;

  // Sum the signal components into a composite signal p.d.f.
  RooRealVar sig1frac("sig1frac","fraction of component 1 in signal",0.8,0.,1.) ;
  RooAddPdf sig("sig","Signal",RooArgList(sig1,sig2),sig1frac) ;

  // Sum the composite signal and background
  RooRealVar bkgfrac("bkgfrac","fraction of background",0.5,0.,1.) ;
  RooAddPdf  model("model","g1+g2+a",RooArgList(bkg,sig),bkgfrac) ;


  // C r e a t e   b i n n e d   d a t a s e t
  // -----------------------------------------

  RooDataSet* d = model.generate(x,10000) ;
  RooDataHist* dh = d->binnedClone() ;


  // Construct a chi^2 of the data and the model,
  // which is the input probability density scaled
  // by the number of events in the dataset
  RooChi2Var chi2("chi2","chi2",model,*dh) ;

  // Use RooMinimizer interface to minimize chi^2
  RooMinimizer m(chi2) ;
  m.migrad() ;
  m.hesse() ;

  RooFitResult* r = m.save() ;

  regResult(r,"rf602_r") ;

  delete d ;
  delete dh ;

  return kTRUE ;
  }
} ;
