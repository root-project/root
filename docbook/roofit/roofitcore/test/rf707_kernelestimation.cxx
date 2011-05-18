//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #707
// 
// Using non-parametric (multi-dimensional) kernel estimation p.d.f.s
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
#include "RooKeysPdf.h"
#include "RooNDKeysPdf.h"
#include "RooProdPdf.h"
#include "TCanvas.h"
#include "TH1.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic707 : public RooFitTestUnit
{
public: 
  TestBasic707(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Kernel estimation p.d.f.s",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C r e a t e   l o w   s t a t s   1 - D   d a t a s e t 
  // -------------------------------------------------------

  // Create a toy pdf for sampling
  RooRealVar x("x","x",0,20) ;
  RooPolynomial p("p","p",x,RooArgList(RooConst(0.01),RooConst(-0.01),RooConst(0.0004))) ;

  // Sample 500 events from p
  RooDataSet* data1 = p.generate(x,200) ;



  // C r e a t e   1 - D   k e r n e l   e s t i m a t i o n   p d f
  // ---------------------------------------------------------------

  // Create adaptive kernel estimation pdf. In this configuration the input data
  // is mirrored over the boundaries to minimize edge effects in distribution
  // that do not fall to zero towards the edges
  RooKeysPdf kest1("kest1","kest1",x,*data1,RooKeysPdf::MirrorBoth) ;

  // An adaptive kernel estimation pdf on the same data without mirroring option
  // for comparison
  RooKeysPdf kest2("kest2","kest2",x,*data1,RooKeysPdf::NoMirror) ;


  // Adaptive kernel estimation pdf with increased bandwidth scale factor
  // (promotes smoothness over detail preservation)
  RooKeysPdf kest3("kest3","kest3",x,*data1,RooKeysPdf::MirrorBoth,2) ;


  // Plot kernel estimation pdfs with and without mirroring over data
  RooPlot* frame = x.frame(Title("Adaptive kernel estimation pdf with and w/o mirroring"),Bins(20)) ;
  data1->plotOn(frame) ;
  kest1.plotOn(frame) ;    
  kest2.plotOn(frame,LineStyle(kDashed),LineColor(kRed)) ;    


  // Plot kernel estimation pdfs with regular and increased bandwidth
  RooPlot* frame2 = x.frame(Title("Adaptive kernel estimation pdf with regular, increased bandwidth")) ;
  kest1.plotOn(frame2) ;    
  kest3.plotOn(frame2,LineColor(kMagenta)) ;    



  // C r e a t e   l o w   s t a t s   2 - D   d a t a s e t 
  // -------------------------------------------------------

  // Construct a 2D toy pdf for sampleing
  RooRealVar y("y","y",0,20) ;
  RooPolynomial py("py","py",y,RooArgList(RooConst(0.01),RooConst(0.01),RooConst(-0.0004))) ;
  RooProdPdf pxy("pxy","pxy",RooArgSet(p,py)) ;
  RooDataSet* data2 = pxy.generate(RooArgSet(x,y),1000) ;



  // C r e a t e   2 - D   k e r n e l   e s t i m a t i o n   p d f
  // ---------------------------------------------------------------

  // Create 2D adaptive kernel estimation pdf with mirroring 
  RooNDKeysPdf kest4("kest4","kest4",RooArgSet(x,y),*data2,"am") ;

  // Create 2D adaptive kernel estimation pdf with mirroring and double bandwidth
  RooNDKeysPdf kest5("kest5","kest5",RooArgSet(x,y),*data2,"am",2) ;

  // Create a histogram of the data
  TH1* hh_data = data2->createHistogram("hh_data",x,Binning(10),YVar(y,Binning(10))) ;

  // Create histogram of the 2d kernel estimation pdfs
  TH1* hh_pdf = kest4.createHistogram("hh_pdf",x,Binning(25),YVar(y,Binning(25))) ;
  TH1* hh_pdf2 = kest5.createHistogram("hh_pdf2",x,Binning(25),YVar(y,Binning(25))) ;
  hh_pdf->SetLineColor(kBlue) ;
  hh_pdf2->SetLineColor(kMagenta) ;
    
  regPlot(frame,"rf707_plot1") ;
  regPlot(frame2,"rf707_plot2") ;
  regTH(hh_data,"rf707_hhdata") ;
  regTH(hh_pdf,"rf707_hhpdf") ;
  regTH(hh_pdf2,"rf707_hhpdf2") ;
  
  delete data1 ;
  delete data2 ;

  return kTRUE ;
  }
} ;
