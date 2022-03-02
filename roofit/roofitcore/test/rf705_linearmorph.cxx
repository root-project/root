//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #705
//
// Linear interpolation between p.d.f shapes using the 'Alex Read' algorithm
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
#include "RooPolynomial.h"
#include "RooLinearMorph.h"
#include "RooNLLVar.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "TH1.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic705 : public RooFitTestUnit
{
public:

  Double_t ctol() { return 5e-2 ; } // very conservative, this is a numerically difficult test

  TestBasic705(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Linear morph operator p.d.f.",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C r e a t e   e n d   p o i n t   p d f   s h a p e s
  // ------------------------------------------------------

  // Observable
  RooRealVar x("x","x",-20,20) ;

  // Lower end point shape: a Gaussian
  RooRealVar g1mean("g1mean","g1mean",-10) ;
  RooGaussian g1("g1","g1",x,g1mean,RooConst(2)) ;

  // Upper end point shape: a Polynomial
  RooPolynomial g2("g2","g2",x,RooArgSet(RooConst(-0.03),RooConst(-0.001))) ;



  // C r e a t e   i n t e r p o l a t i n g   p d f
  // -----------------------------------------------

  // Create interpolation variable
  RooRealVar alpha("alpha","alpha",0,1.0) ;

  // Specify sampling density on observable and interpolation variable
  x.setBins(1000,"cache") ;
  alpha.setBins(50,"cache") ;

  // Construct interpolating pdf in (x,a) represent g1(x) at a=a_min
  // and g2(x) at a=a_max
  RooLinearMorph lmorph("lmorph","lmorph",g1,g2,x,alpha) ;



  // P l o t   i n t e r p o l a t i n g   p d f   a t   v a r i o u s   a l p h a
  // -----------------------------------------------------------------------------

  // Show end points as blue curves
  RooPlot* frame1 = x.frame() ;
  g1.plotOn(frame1) ;
  g2.plotOn(frame1) ;

  // Show interpolated shapes in red
  alpha.setVal(0.125) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt1")) ;
  alpha.setVal(0.25) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt2")) ;
  alpha.setVal(0.375) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt3")) ;
  alpha.setVal(0.50) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt4")) ;
  alpha.setVal(0.625) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt5")) ;
  alpha.setVal(0.75) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt6")) ;
  alpha.setVal(0.875) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt7")) ;
  alpha.setVal(0.95) ;
  lmorph.plotOn(frame1,LineColor(kRed),Name("alt8")) ;



  // S h o w   2 D   d i s t r i b u t i o n   o f   p d f ( x , a l p h a )
  // -----------------------------------------------------------------------

  // Create 2D histogram
  TH1* hh = lmorph.createHistogram("hh",x,Binning(40),YVar(alpha,Binning(40))) ;
  hh->SetLineColor(kBlue) ;


  // F i t   p d f   t o   d a t a s e t   w i t h   a l p h a = 0 . 8
  // -----------------------------------------------------------------

  // Generate a toy dataset at alpha = 0.8
  alpha=0.8 ;
  RooDataSet* data = lmorph.generate(x,1000) ;

  // Fit pdf to toy data
  lmorph.setCacheAlpha(kTRUE) ;
  lmorph.fitTo(*data) ;

  // Plot fitted pdf and data overlaid
  RooPlot* frame2 = x.frame(Bins(100)) ;
  data->plotOn(frame2) ;
  lmorph.plotOn(frame2) ;


  // S c a n   - l o g ( L )   v s   a l p h a
  // -----------------------------------------

  // Show scan -log(L) of dataset w.r.t alpha
  RooPlot* frame3 = alpha.frame(Bins(100),Range(0.5,0.9)) ;

  // Make 2D pdf of histogram
  RooNLLVar nll("nll","nll",lmorph,*data) ;
  nll.plotOn(frame3,ShiftToZero()) ;

  lmorph.setCacheAlpha(kFALSE) ;


  regPlot(frame1,"rf705_plot1") ;
  regPlot(frame2,"rf705_plot2") ;
  regPlot(frame3,"rf705_plot3") ;
  regTH(hh,"rf705_hh") ;

  delete data ;

  return kTRUE;
  }
} ;
