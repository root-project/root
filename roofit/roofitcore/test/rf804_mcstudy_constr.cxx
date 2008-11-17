/////////////////////////////////////////////////////////////////////////
//
// 'VALIDATION AND MC STUDIES' RooFit tutorial macro #804
// 
// Using RooMCStudy on models with constrains
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
#include "RooAddPdf.h"
#include "RooProdPdf.h"
#include "RooMCStudy.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


class TestBasic804 : public RooFitTestUnit
{
public: 
  TestBasic804(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("MC Studies with aux. obs. constraints",refFile,writeRef,verbose) {} ;

  Double_t htol() { return 0.1 ; } // numerically very difficult test

  Bool_t testCode() {

  // C r e a t e   m o d e l   w i t h   p a r a m e t e r   c o n s t r a i n t
  // ---------------------------------------------------------------------------

  // Observable
  RooRealVar x("x","x",-10,10) ;

  // Signal component
  RooRealVar m("m","m",0,-10,10) ;
  RooRealVar s("s","s",2,0.1,10) ;
  RooGaussian g("g","g",x,m,s) ;

  // Background component
  RooPolynomial p("p","p",x) ;

  // Composite model
  RooRealVar f("f","f",0.4,0.,1.) ;
  RooAddPdf sum("sum","sum",RooArgSet(g,p),f) ;

  // Construct constraint on parameter f
  RooGaussian fconstraint("fconstraint","fconstraint",f,RooConst(0.7),RooConst(0.1)) ;

  // Multiply constraint with p.d.f
  RooProdPdf sumc("sumc","sum with constraint",RooArgSet(sum,fconstraint)) ;



  // S e t u p   t o y   s t u d y   w i t h   m o d e l
  // ---------------------------------------------------

  // Perform toy study with internal constraint on f
  RooMCStudy mcs(sumc,x,Constrain(f),Silence(),Binned(),FitOptions(PrintLevel(-1))) ;

  // Run 50 toys of 2000 events.  
  // Before each toy is generated, a value for the f is sampled from the constraint pdf and 
  // that value is used for the generation of that toy.
  mcs.generateAndFit(50,2000) ;

  // Make plot of distribution of generated value of f parameter
  RooRealVar* f_gen = (RooRealVar*) mcs.fitParDataSet().get()->find("f_gen") ;
  TH1* h_f_gen = new TH1F("h_f_gen","",40,0,1) ;
  mcs.fitParDataSet().fillHistogram(h_f_gen,*f_gen) ;

  // Make plot of distribution of fitted value of f parameter
  RooPlot* frame1  = mcs.plotParam(f,Bins(40),Range(0.4,1)) ;
  frame1->SetTitle("Distribution of fitted f values") ;

  // Make plot of pull distribution on f
  RooPlot* frame2 = mcs.plotPull(f,Bins(40),Range(-3,3)) ;
  frame1->SetTitle("Distribution of f pull values") ;

  regTH(h_f_gen,"rf804_h_f_gen") ;
  regPlot(frame1,"rf804_plot1") ;
  regPlot(frame2,"rf804_plot2") ;
  
  return kTRUE ;
  }
} ;

