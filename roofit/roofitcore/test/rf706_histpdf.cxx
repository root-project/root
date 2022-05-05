//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #706
//
// Histogram based p.d.f.s and functions
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
#include "RooHistPdf.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic706 : public RooFitTestUnit
{
public:
  TestBasic706(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Histogram based p.d.f.s",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C r e a t e   p d f   f o r   s a m p l i n g
  // ---------------------------------------------

  RooRealVar x("x","x",0,20) ;
  RooPolynomial p("p","p",x,RooArgList(RooConst(0.01),RooConst(-0.01),RooConst(0.0004))) ;



  // C r e a t e   l o w   s t a t s   h i s t o g r a m
  // ---------------------------------------------------

  // Sample 500 events from p
  x.setBins(20) ;
  RooDataSet* data1 = p.generate(x,500) ;

  // Create a binned dataset with 20 bins and 500 events
  RooDataHist* hist1 = data1->binnedClone() ;

  // Represent data in dh as pdf in x
  RooHistPdf histpdf1("histpdf1","histpdf1",x,*hist1,0) ;

  // Plot unbinned data and histogram pdf overlaid
  RooPlot* frame1 = x.frame(Title("Low statistics histogram pdf"),Bins(100)) ;
  data1->plotOn(frame1) ;
  histpdf1.plotOn(frame1) ;


  // C r e a t e   h i g h   s t a t s   h i s t o g r a m
  // -----------------------------------------------------

  // Sample 100000 events from p
  x.setBins(10) ;
  RooDataSet* data2 = p.generate(x,100000) ;

  // Create a binned dataset with 10 bins and 100K events
  RooDataHist* hist2 = data2->binnedClone() ;

  // Represent data in dh as pdf in x, apply 2nd order interpolation
  RooHistPdf histpdf2("histpdf2","histpdf2",x,*hist2,2) ;

  // Plot unbinned data and histogram pdf overlaid
  RooPlot* frame2 = x.frame(Title("High stats histogram pdf with interpolation"),Bins(100)) ;
  data2->plotOn(frame2) ;
  histpdf2.plotOn(frame2) ;


  regPlot(frame1,"rf607_plot1") ;
  regPlot(frame2,"rf607_plot2") ;

  delete data1 ;
  delete hist1 ;
  delete data2 ;
  delete hist2 ;

  return true ;
  }
} ;
