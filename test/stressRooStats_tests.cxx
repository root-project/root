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
#include "RooUnitTest.h"
#include "RooStats/NumberCountingUtils.h"
#include "RooStats/RooStatsUtils.h"

using namespace RooFit ;
using namespace RooStats ;


class TestBasic101 : public RooUnitTest
{
public: 
  TestBasic101(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooUnitTest("Zbi and Zgamma",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // Make model for prototype on/off problem
    // Pois(x | s+b) * Pois(y | tau b )
    // for Z_Gamma, use uniform prior on b.
    RooWorkspace* w = new RooWorkspace("w",true);
    w->factory("Poisson::px(x[150,0,500],sum::splusb(s[0,0,100],b[100,0,300]))");
    w->factory("Poisson::py(y[100,0,500],prod::taub(tau[1.],b))");	     
    w->factory("Uniform::prior_b(b)");

    // construct the Bayesian-averaged model (eg. a projection pdf)
    // p'(x|s) = \int db p(x|s+b) * [ p(y|b) * prior(b) ]
    w->factory("PROJ::averagedModel(PROD::foo(px|b,py,prior_b),b)") ;
    
    // plot it, blue is averaged model, red is b known exactly
    RooPlot* frame = w->var("x")->frame() ;
    w->pdf("averagedModel")->plotOn(frame) ;
    w->pdf("px")->plotOn(frame,LineColor(kRed)) ;
    
    // compare analytic calculation of Z_Bi
    // with the numerical RooFit implementation of Z_Gamma
    // for an example with x = 150, y = 100
    
    // numeric RooFit Z_Gamma
    w->var("y")->setVal(100);
    w->var("x")->setVal(150);
    RooAbsReal* cdf = w->pdf("averagedModel")->createCdf(*w->var("x"));
    cdf->getVal(); // get ugly print messages out of the way
    
    Double_t hybrid_p_value = cdf->getVal() ;
    Double_t zgamma_signif = PValueToSignificance(1-cdf->getVal()) ;

    // analytic Z_Bi
    Double_t Z_Bi = NumberCountingUtils::BinomialWithTauObsZ(150, 100, 1);

    // Register output quantities for regression test
    regPlot(frame,"rs101_zbi_zgamma") ;
    regValue(hybrid_p_value,"rs101_hybrid_p_value") ;
    regValue(zgamma_signif,"rs101_zgamma_signif") ;
    regValue(Z_Bi,"rs101_Z_Bi") ;
    
    return kTRUE ;
  }
} ;


class TestBasic102 : public RooUnitTest
{
public: 
  TestBasic102(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooUnitTest("DUMMY",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {    
    return kTRUE ;
  }
} ;

class TestBasic103 : public RooUnitTest
{
public: 
  TestBasic103(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooUnitTest("DUMMY",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {    
    return kTRUE ;
  }
} ;

class TestBasic104 : public RooUnitTest
{
public: 
  TestBasic104(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooUnitTest("DUMMY",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {    
    return kTRUE ;
  }
} ;

class TestBasic105 : public RooUnitTest
{
public: 
  TestBasic105(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooUnitTest("DUMMY",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {    
    return kTRUE ;
  }
} ;

