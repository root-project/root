//////////////////////////////////////////////////////////////////////////
//
// 'NUMERIC ALGORITHM TUNING' RooFit tutorial macro #903 
// 
//  Caching of slow numeric integrals and parameterizations of slow
//  numeric integrals
//
// 07/2008 - Wouter Verkerke 
// 
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooPlot.h"
#include "RooWorkspace.h"
#include "RooExpensiveObjectCache.h"
#include "TFile.h"
#include "TH1.h"

using namespace RooFit ;

RooWorkspace* getWorkspace(Int_t mode) ;

void rf903_numintcache(Int_t mode=0)
{
  // Mode = 0 : Run plain fit (slow)
  // Mode = 1 : Generate workspace with precalculated integral and store it on file (prepare for accelerated running)
  // Mode = 2 : Run fit from previously stored workspace including cached integrals (fast, requires run in mode=1 first)

  // C r e a t e ,   s a v e   o r   l o a d   w o r k s p a c e   w i t h   p . d . f . 
  // -----------------------------------------------------------------------------------

  // Make/load workspace, exit here in mode 1
  RooWorkspace* w = getWorkspace(mode) ;
  if (mode==1) {

    // Show workspace that was created
    w->Print() ;

    // Show plot of cached integral values
    RooDataHist* hhcache = (RooDataHist*) w->expensiveObjectCache().getObj(1) ;
    if (hhcache) { 

       new TCanvas("rf903_numintcache","rf903_numintcache",600,600) ;
       hhcache->createHistogram("a")->Draw() ;
    
    }
    else { 
       Error("rf903_numintcache","Cached histogram is not existing in workspace");
    }
       return ;
  }

  // U s e   p . d . f .   f r o m   w o r k s p a c e   f o r   g e n e r a t i o n   a n d   f i t t i n g 
  // -----------------------------------------------------------------------------------

  // This is always slow (need to find maximum function value empirically in 3D space)
  RooDataSet* d = w->pdf("model")->generate(RooArgSet(*w->var("x"),*w->var("y"),*w->var("z")),1000) ;

  // This is slow in mode 0, but fast in mode 1
  w->pdf("model")->fitTo(*d,Verbose(kTRUE),Timer(kTRUE)) ; 

  // Projection on x (always slow as 2D integral over Y,Z at fitted value of a is not cached)
  RooPlot* framex = w->var("x")->frame(Title("Projection of 3D model on X")) ;
  d->plotOn(framex) ;
  w->pdf("model")->plotOn(framex) ;

  // Draw x projection on canvas
  new TCanvas("rf903_numintcache","rf903_numintcache",600,600) ;
  framex->Draw() ;

  // Make workspace available on command line after macro finishes
  gDirectory->Add(w) ;

  return ;

 
}



RooWorkspace* getWorkspace(Int_t mode) 
{
  // C r e a t e ,   s a v e   o r   l o a d   w o r k s p a c e   w i t h   p . d . f . 
  // -----------------------------------------------------------------------------------
  //
  // Mode = 0 : Create workspace for plain running (no integral caching)
  // Mode = 1 : Generate workspace with precalculated integral and store it on file
  // Mode = 2 : Load previously stored workspace from file

  RooWorkspace* w(0) ;

  if (mode!=2) {

    // Create empty workspace workspace 
    w = new RooWorkspace("w",1) ;

    // Make a difficult to normalize  p.d.f. in 3 dimensions that is integrated numerically.
    w->factory("EXPR::model('1/((x-a)*(x-a)+0.01)+1/((y-a)*(y-a)+0.01)+1/((z-a)*(z-a)+0.01)',x[-1,1],y[-1,1],z[-1,1],a[-5,5])") ;
  }

  if (mode==1) {
    
    // Instruct model to precalculate normalization integral that integrate at least
    // two dimensions numerically. In this specific case the integral value for
    // all values of parameter 'a' are stored in a histogram and available for use 
    // in subsequent fitting and plotting operations (interpolation is applied)

    // w->pdf("model")->setNormValueCaching(3) ;
    w->pdf("model")->setStringAttribute("CACHEPARMINT","x:y:z");
    
    // Evaluate p.d.f. once to trigger filling of cache
    RooArgSet normSet(*w->var("x"),*w->var("y"),*w->var("z")) ;
    w->pdf("model")->getVal(&normSet) ;
    w->writeToFile("rf903_numintcache.root") ;

  } 

  if (mode==2) {    
    // Load preexisting workspace from file in mode==2
    TFile* f = new TFile("rf903_numintcache.root") ;
    w = (RooWorkspace*) f->Get("w") ;
  }

  // Return created or loaded workspace
  return w ;
}
