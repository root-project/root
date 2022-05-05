/////////////////////////////////////////////////////////////////////////
//
// 'VALIDATION AND MC STUDIES' RooFit tutorial macro #803
//
// RooMCStudy: Using the randomizer and profile likelihood add-on models
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
#include "RooMCStudy.h"
#include "RooRandomizeParamMCSModule.h"
#include "RooDLLSignificanceMCSModule.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TDirectory.h"

using namespace RooFit ;


class TestBasic803 : public RooFitTestUnit
{
public:
  TestBasic803(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("MC Study with param rand. and Z calc",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C r e a t e   m o d e l
  // -----------------------

  // Simulation of signal and background of top quark decaying into
  // 3 jets with background

  // Observable
  RooRealVar mjjj("mjjj","m(3jet) (GeV)",100,85.,350.) ;

  // Signal component (Gaussian)
  RooRealVar mtop("mtop","m(top)",162) ;
  RooRealVar wtop("wtop","m(top) resolution",15.2) ;
  RooGaussian sig("sig","top signal",mjjj,mtop,wtop) ;

  // Background component (Chebychev)
  RooRealVar c0("c0","Chebychev coefficient 0",-0.846,-1.,1.) ;
  RooRealVar c1("c1","Chebychev coefficient 1", 0.112,-1.,1.) ;
  RooRealVar c2("c2","Chebychev coefficient 2", 0.076,-1.,1.) ;
  RooChebychev bkg("bkg","combinatorial background",mjjj,RooArgList(c0,c1,c2)) ;

  // Composite model
  RooRealVar nsig("nsig","number of signal events",53,0,1e3) ;
  RooRealVar nbkg("nbkg","number of background events",103,0,5e3) ;
  RooAddPdf model("model","model",RooArgList(sig,bkg),RooArgList(nsig,nbkg)) ;



  // C r e a t e   m a n a g e r
  // ---------------------------

  // Configure manager to perform binned extended likelihood fits (Binned(),Extended()) on data generated
  // with a Poisson fluctuation on Nobs (Extended())
  RooMCStudy* mcs = new RooMCStudy(model,mjjj,Binned(),Silence(),Extended(true),
               FitOptions(Extended(true),PrintEvalErrors(-1))) ;



  // C u s t o m i z e   m a n a g e r
  // ---------------------------------

  // Add module that randomizes the summed value of nsig+nbkg
  // sampling from a uniform distribution between 0 and 1000
  //
  // In general one can randomize a single parameter, or a
  // sum of N parameters, using either a uniform or a Gaussian
  // distribution. Multiple randomization can be executed
  // by a single randomizer module

  RooRandomizeParamMCSModule randModule ;
  randModule.sampleSumUniform(RooArgSet(nsig,nbkg),50,500) ;
  mcs->addModule(randModule) ;


  // Add profile likelihood calculation of significance. Redo each
  // fit while keeping parameter nsig fixed to zero. For each toy,
  // the difference in -log(L) of both fits is stored, as well
  // a simple significance interpretation of the delta(-logL)
  // using Dnll = 0.5 sigma^2

  RooDLLSignificanceMCSModule sigModule(nsig,0) ;
  mcs->addModule(sigModule) ;



  // R u n   m a n a g e r ,   m a k e   p l o t s
  // ---------------------------------------------

  mcs->generateAndFit(50) ;

  // Make some plots
  RooRealVar* ngen    = (RooRealVar*) mcs->fitParDataSet().get()->find("ngen") ;
  RooRealVar* dll     = (RooRealVar*) mcs->fitParDataSet().get()->find("dll_nullhypo_nsig") ;
  RooRealVar* z       = (RooRealVar*) mcs->fitParDataSet().get()->find("significance_nullhypo_nsig") ;
  RooRealVar* nsigerr = (RooRealVar*) mcs->fitParDataSet().get()->find("nsigerr") ;

  TH1* dll_vs_ngen     = new TH2F("h_dll_vs_ngen"    ,"",40,0,500,40,0,50) ;
  TH1* z_vs_ngen       = new TH2F("h_z_vs_ngen"      ,"",40,0,500,40,0,10) ;
  TH1* errnsig_vs_ngen = new TH2F("h_nsigerr_vs_ngen","",40,0,500,40,0,30) ;
  TH1* errnsig_vs_nsig = new TH2F("h_nsigerr_vs_nsig","",40,0,200,40,0,30) ;

  mcs->fitParDataSet().fillHistogram(dll_vs_ngen,RooArgList(*ngen,*dll)) ;
  mcs->fitParDataSet().fillHistogram(z_vs_ngen,RooArgList(*ngen,*z)) ;
  mcs->fitParDataSet().fillHistogram(errnsig_vs_ngen,RooArgList(*ngen,*nsigerr)) ;
  mcs->fitParDataSet().fillHistogram(errnsig_vs_nsig,RooArgList(nsig,*nsigerr)) ;

  regTH(dll_vs_ngen,"rf803_dll_vs_ngen") ;
  regTH(z_vs_ngen,"rf803_z_vs_ngen") ;
  regTH(errnsig_vs_ngen,"rf803_errnsig_vs_ngen") ;
  regTH(errnsig_vs_nsig,"rf803_errnsig_vs_nsig") ;

  delete mcs ;

  return true ;

  }
} ;




