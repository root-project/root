/////////////////////////////////////////////////////////////////////////
//
// 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #604
//
// Fitting with constraints
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
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


class TestBasic604 : public RooFitTestUnit
{
public:
  TestBasic604(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Auxiliary observable constraints",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C r e a t e   m o d e l  a n d   d a t a s e t
  // ----------------------------------------------

  // Construct a Gaussian p.d.f
  RooRealVar x("x","x",-10,10) ;

  RooRealVar m("m","m",0,-10,10) ;
  RooRealVar s("s","s",2,0.1,10) ;
  RooGaussian gauss("gauss","gauss(x,m,s)",x,m,s) ;

  // Construct a flat p.d.f (polynomial of 0th order)
  RooPolynomial poly("poly","poly(x)",x) ;

  // Construct model = f*gauss + (1-f)*poly
  RooRealVar f("f","f",0.5,0.,1.) ;
  RooAddPdf model("model","model",RooArgSet(gauss,poly),f) ;

  // Generate small dataset for use in fitting below
  RooDataSet* d = model.generate(x,50) ;



  // C r e a t e   c o n s t r a i n t   p d f
  // -----------------------------------------

  // Construct Gaussian constraint p.d.f on parameter f at 0.8 with resolution of 0.1
  RooGaussian fconstraint("fconstraint","fconstraint",f,RooConst(0.8),RooConst(0.1)) ;



  // M E T H O D   1   -   A d d   i n t e r n a l   c o n s t r a i n t   t o   m o d e l
  // -------------------------------------------------------------------------------------

  // Multiply constraint term with regular p.d.f using RooProdPdf
  // Specify in fitTo() that internal constraints on parameter f should be used

  // Multiply constraint with p.d.f
  RooProdPdf modelc("modelc","model with constraint",RooArgSet(model,fconstraint)) ;

  // Fit modelc without use of constraint term
  RooFitResult* r1 = modelc.fitTo(*d,Save()) ;

  // Fit modelc with constraint term on parameter f
  RooFitResult* r2 = modelc.fitTo(*d,Constrain(f),Save()) ;



  // M E T H O D   2   -     S p e c i f y   e x t e r n a l   c o n s t r a i n t   w h e n   f i t t i n g
  // -------------------------------------------------------------------------------------------------------

  // Construct another Gaussian constraint p.d.f on parameter f at 0.8 with resolution of 0.1
  RooGaussian fconstext("fconstext","fconstext",f,RooConst(0.2),RooConst(0.1)) ;

  // Fit with external constraint
  RooFitResult* r3 = model.fitTo(*d,ExternalConstraints(fconstext),Save()) ;


  regResult(r1,"rf604_r1") ;
  regResult(r2,"rf604_r2") ;
  regResult(r3,"rf604_r3") ;

  delete d ;

  return true ;
  }
} ;
