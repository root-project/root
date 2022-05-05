/////////////////////////////////////////////////////////////////////////
//
// 'VALIDATION AND MC STUDIES' RooFit tutorial macro #801
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
#include "TH2.h"
#include "RooFitResult.h"
#include "TStyle.h"
#include "TDirectory.h"

using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic801 : public RooFitTestUnit
{
public:
  TestBasic801(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Automated MC studies",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C r e a t e   m o d e l
  // -----------------------

  // Declare observable x
  RooRealVar x("x","x",0,10) ;
  x.setBins(40) ;

  // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
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
  RooRealVar nbkg("nbkg","number of background events,",150,0,1000) ;
  RooRealVar nsig("nsig","number of signal events",150,0,1000) ;
  RooAddPdf  model("model","g1+g2+a",RooArgList(bkg,sig),RooArgList(nbkg,nsig)) ;



  // C r e a t e   m a n a g e r
  // ---------------------------

  // Instantiate RooMCStudy manager on model with x as observable and given choice of fit options
  //
  // The Silence() option kills all messages below the PROGRESS level, leaving only a single message
  // per sample executed, and any error message that occur during fitting
  //
  // The Extended() option has two effects:
  //    1) The extended ML term is included in the likelihood and
  //    2) A poisson fluctuation is introduced on the number of generated events
  //
  // The FitOptions() given here are passed to the fitting stage of each toy experiment.
  // If Save() is specified, the fit result of each experiment is saved by the manager
  //
  // A Binned() option is added in this example to bin the data between generation and fitting
  // to speed up the study at the expemse of some precision

  RooMCStudy* mcstudy = new RooMCStudy(model,x,Binned(true),Silence(),Extended(),
                   FitOptions(Save(true),PrintEvalErrors(0))) ;


  // G e n e r a t e   a n d   f i t   e v e n t s
  // ---------------------------------------------

  // Generate and fit 100 samples of Poisson(nExpected) events
  mcstudy->generateAndFit(100) ;



  // E x p l o r e   r e s u l t s   o f   s t u d y
  // ------------------------------------------------

  // Make plots of the distributions of mean, the error on mean and the pull of mean
  RooPlot* frame1 = mcstudy->plotParam(mean,Bins(40)) ;
  RooPlot* frame2 = mcstudy->plotError(mean,Bins(40)) ;
  RooPlot* frame3 = mcstudy->plotPull(mean,Bins(40),FitGauss(true)) ;

  // Plot distribution of minimized likelihood
  RooPlot* frame4 = mcstudy->plotNLL(Bins(40)) ;

  regPlot(frame1,"rf801_plot1") ;
  regPlot(frame2,"rf801_plot2") ;
  regPlot(frame3,"rf801_plot3") ;
  regPlot(frame4,"rf801_plot4") ;

  delete mcstudy ;

  return true ;
  }
} ;
