//////////////////////////////////////////////////////////////////////////
//
// 'BASIC FUNCTIONALITY' RooFit tutorial macro #101
// 
// Fitting, plotting, toy data generation on one-dimensional p.d.f
//
// pdf = gauss(x,m,s) 
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
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic101 : public RooFitTestUnit
{
public: 
  TestBasic101(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Fitting,plotting & event generation of basic p.d.f",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // S e t u p   m o d e l 
    // ---------------------
    
    // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mean("mean","mean of gaussian",1,-10,10) ;
    RooRealVar sigma("sigma","width of gaussian",1,0.1,10) ;
    
    // Build gaussian p.d.f in terms of x,mean and sigma
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;  
    
    // Construct plot frame in 'x'
    RooPlot* xframe = x.frame(Title("Gaussian p.d.f.")) ;
    
    
    // P l o t   m o d e l   a n d   c h a n g e   p a r a m e t e r   v a l u e s
    // ---------------------------------------------------------------------------
    
    // Plot gauss in frame (i.e. in x) 
    gauss.plotOn(xframe) ;
    
    // Change the value of sigma to 3
    sigma.setVal(3) ;
    
    // Plot gauss in frame (i.e. in x) and draw frame on canvas
    gauss.plotOn(xframe,LineColor(kRed),Name("another")) ;
    
    
    // G e n e r a t e   e v e n t s 
    // -----------------------------
    
    // Generate a dataset of 1000 events in x from gauss
    RooDataSet* data = gauss.generate(x,10000) ;  
    
    // Make a second plot frame in x and draw both the 
    // data and the p.d.f in the frame
    RooPlot* xframe2 = x.frame(Title("Gaussian p.d.f. with data")) ;
    data->plotOn(xframe2) ;
    gauss.plotOn(xframe2) ;
    
    
    // F i t   m o d e l   t o   d a t a
    // -----------------------------
    
    // Fit pdf to data
    gauss.fitTo(*data) ;
        
    
    // --- Post processing for stressRooFit ---
    regPlot(xframe ,"rf101_plot1") ;
    regPlot(xframe2,"rf101_plot2") ;
    
    delete data ;
    
    return kTRUE ;
  }
} ;


