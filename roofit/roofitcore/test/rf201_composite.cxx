/////////////////////////////////////////////////////////////////////////
//
// 'ADDITION AND CONVOLUTION' RooFit tutorial macro #201
// 
// Composite p.d.f with signal and background component
//
// pdf = f_bkg * bkg(x,a0,a1) + (1-fbkg) * (f_sig1 * sig1(x,m,s1 + (1-f_sig1) * sig2(x,m,s2)))
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
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


class TestBasic201 : public RooFitTestUnit
{
public: 
  TestBasic201(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Addition operator p.d.f.",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // S e t u p   c o m p o n e n t   p d f s 
    // ---------------------------------------
    
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
    
    
    ////////////////////////////////////////////////////
    // M E T H O D   1 - T w o   R o o A d d P d f s  //
    ////////////////////////////////////////////////////
    
    
    // A d d   s i g n a l   c o m p o n e n t s 
    // ------------------------------------------
    
    // Sum the signal components into a composite signal p.d.f.
    RooRealVar sig1frac("sig1frac","fraction of component 1 in signal",0.8,0.,1.) ;
    RooAddPdf sig("sig","Signal",RooArgList(sig1,sig2),sig1frac) ;
    
    
    // A d d  s i g n a l   a n d   b a c k g r o u n d
    // ------------------------------------------------
    
    // Sum the composite signal and background 
    RooRealVar bkgfrac("bkgfrac","fraction of background",0.5,0.,1.) ;
    RooAddPdf  model("model","g1+g2+a",RooArgList(bkg,sig),bkgfrac) ;
    
    
    // S a m p l e ,   f i t   a n d   p l o t   m o d e l
    // ---------------------------------------------------
    
    // Generate a data sample of 1000 events in x from model
    RooDataSet *data = model.generate(x,1000) ;
    
    // Fit model to data
    model.fitTo(*data) ;
    
    // Plot data and PDF overlaid
    RooPlot* xframe = x.frame(Title("Example of composite pdf=(sig1+sig2)+bkg")) ;
    data->plotOn(xframe) ;
    model.plotOn(xframe) ;
    
    // Overlay the background component of model with a dashed line
    model.plotOn(xframe,Components(bkg),LineStyle(kDashed)) ;
    
    // Overlay the background+sig2 components of model with a dotted line
    model.plotOn(xframe,Components(RooArgSet(bkg,sig2)),LineStyle(kDotted)) ;
    

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // M E T H O D   2 - O n e   R o o A d d P d f   w i t h   r e c u r s i v e   f r a c t i o n s  //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Construct sum of models on one go using recursive fraction interpretations
    //
    //   model2 = bkg + (sig1 + sig2)
    //
    RooAddPdf  model2("model","g1+g2+a",RooArgList(bkg,sig1,sig2),RooArgList(bkgfrac,sig1frac),kTRUE) ;    
    
    // NB: Each coefficient is interpreted as the fraction of the 
    // left-hand component of the i-th recursive sum, i.e.
    //
    //   sum4 = A + ( B + ( C + D)  with fraction fA, fB and fC expands to
    //
    //   sum4 = fA*A + (1-fA)*(fB*B + (1-fB)*(fC*C + (1-fC)*D))
    
    
    // P l o t   r e c u r s i v e   a d d i t i o n   m o d e l
    // ---------------------------------------------------------
    model2.plotOn(xframe,LineColor(kRed),LineStyle(kDashed)) ;
    model2.plotOn(xframe,Components(RooArgSet(bkg,sig2)),LineColor(kRed),LineStyle(kDashed)) ;

  
    regPlot(xframe,"rf201_plot1") ;
    
    delete data ;
    return kTRUE ;

  }

} ;

