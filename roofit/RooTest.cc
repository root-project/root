#include <stdio.h>
#include <iostream.h>
#include <fstream.h>
#include "TROOT.h"
#include "TApplication.h"
#include "TMinuit.h"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooMappedCategory.hh"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooAddPdf.hh"
#include "RooFitCore/RooLinearVar.hh"
#include "RooFitCore/RooFitContext.hh"
#include "RooFitModels/RooUnblindCPAsymVar.hh"
#include "RooFitModels/RooGaussian.hh"
#include "RooFitModels/RooArgusBG.hh"
#include "RooFitCore/RooTruthModel.hh"
#include "RooFitCore/RooAddModel.hh"
#include "RooFitModels/RooGaussModel.hh"
#include "RooFitModels/RooDecay.hh"
#include "RooFitModels/RooBMixDecay.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooPdfCustomizer.hh"
#include "RooFitCore/RooThresholdCategory.hh"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooSimFitContext.hh"
#include "RooFitCore/RooTrace.hh"

main(int argc, char **argv) {
  TROOT root("root", "root"); // initialize ROOT
  TApplication app("TAppTest",&argc,argv) ;

  RooArgSet
    *mixVars, *splitPars, *mbsVars ;
  RooRealVar 
    *sigC_bias,*sigT_bias,*sigC_scfa,*sigT_scfa,*sigC_frac,*sigO_frac,
    *bkgC_bias,*bkgC_scfa,*bkgC_frac,*outl_bias,*outl_scfa,*sig_tauB,
    *bgC_tauB,*bgL_tauB,*dm,*sig_eta,*bgC_eta,*bgL_eta,*bgP_eta,*bgP_f,
    *bgC_f,*mbMean,*mbWidth,*mbMax,*argPar,*sigfrac,*zero,*zero2,*one,
    *dtErr,*dtReco,*mB,*pTagB0,*pRecB0,*runNumber ;
  RooCategory 
    *tagCatRaw,*mixState,*tagCat,*runBlock ;
  RooAbsReal  
    *mixProb ;
  RooSuperCategory 
    *fitCat ;
  RooThresholdCategory 
    *mixState_func,*runBlock_func ;
  RooMappedCategory 
    *tagCat_func ;
  RooResolutionModel 
    *bkgC_gauss,*sigC_gauss,*sigT_gauss,*outl_gauss,*sigResModel,*bkgResModel ;
  RooAbsPdf  
    *sigModel,*bgCModel,*bgLModel,*bgPModel,*sigSum,*bkgSum,*gaussPdf,*argusPdf,
    *sigGaussPdf,*bkgArgusPdf,*mixProto,*mbsProto ;
  RooSimultaneous 
    *mixModel,*mbsModel ;
  RooDataSet 
    *data ;
  RooPdfCustomizer 
    *mixCust,*mbsCust ;
  RooFitContext *c ;
  
  RooAbsPdf* k1 ;
  RooFitContext* ck1, *sig ;
  
  RooTrace::active(0) ;
  
  //--- Constants ---
  zero  = new RooRealVar("zero","zero",0.0) ;
  zero2 = new RooRealVar("zero2","zero",0.0) ;
  one   = new RooRealVar("one","one",1.0) ;
  
  //***********************************************************************
  //* Definition of ASCII data set and construction of derived quantities *
  //***********************************************************************
  
  //--- Data set variables ---
  dtErr     = new RooRealVar("dtErr","Calculated Error on Reconstructed Delta(t)",0.1,5.0) ;
  dtReco    = new RooRealVar("dtReco","Reconstructed Delta(t4",-40,20) ;
  mB        = new RooRealVar("mB","Reconstructed B0 mass",5.20,5.30,"GeV") ;
  pTagB0    = new RooRealVar("pTagB0","Tag side B0 flavour probability",0.,1.01) ;
  pRecB0    = new RooRealVar("pRecB0","Reco side B0 flavour probablity",0.,1.01) ;
  runNumber = new RooRealVar("runNumber","Run Number",0,0,30000) ;
  tagCatRaw = new RooCategory("tagCatRaw","Raw Tagging Category") ;

  //--- Tagging categories as they appear in the ascii files ---
  tagCatRaw->clearTypes();
  tagCatRaw->defineType("ElKaon",      11);
  tagCatRaw->defineType("MuKaon",      12);
  tagCatRaw->defineType("Electron",    14);
  tagCatRaw->defineType("Muon",        15);
  tagCatRaw->defineType("Kaon",        13);
  tagCatRaw->defineType("NetTagger-1", 22);
  tagCatRaw->defineType("NetTagger-2", 23);
  
  //--- Derived quantities from data set ---
  mixProb = new RooFormulaVar("mixProb","1-2*abs(pTagB0-pRecB0)",RooArgSet(*pTagB0,*pRecB0)) ;
  mixProb->setPlotRange(-1.01,1.01) ;

  mixState_func = new RooThresholdCategory("mixState","B0-B0bar Mixing State",*mixProb,"Mixed",-1) ;
  mixState_func->addThreshold(0.,"Unmixed",+1) ;  

  tagCat_func = new RooMappedCategory("tagCat","Condensed Tagging Category",*tagCatRaw,"Lep") ; 
  tagCat_func->map("El*"        ,"Lep") ;
  tagCat_func->map("Mu*"        ,"Lep") ;
  tagCat_func->map("Kaon"       ,"Kao") ;
  tagCat_func->map("NetTagger-1","NT1") ;
  tagCat_func->map("NetTagger-2","NT2") ;

  runBlock_func = new RooThresholdCategory("runBlock","Run block (I or II)",*runNumber,"Run2") ;
  runBlock_func->addThreshold(18000,"Run1") ;

  //--- Read data and precalculate derived categories as LValue objects ---
  RooArgSet dataVars(*dtReco,*dtErr,*pTagB0,*tagCatRaw,*pRecB0,*mB,*runNumber) ;
  data = RooDataSet::read("breco2.dat",dataVars,"q","") ;
  mixState = (RooCategory*) data->addColumn(*mixState_func) ;
  tagCat   = (RooCategory*) data->addColumn(*tagCat_func) ;
  runBlock = (RooCategory*) data->addColumn(*runBlock_func) ;

  //***********************************************************************
  //* Construction of proto-type mixing PDF                               *
  //***********************************************************************
    
  //--- Fit parameters ---

  // Resolution model
  sigC_bias = new RooRealVar("sigC_bias","Signal Core Bias",-0.1652,-5,+5) ;
  sigT_bias = new RooRealVar("sigT_bias","Signal Tail Bias",-0.791,-5,+5) ;
  sigC_scfa = new RooRealVar("sigC_scfa","Signal Core Scale Factor",1.0902,0.5,5) ;
  sigT_scfa = new RooRealVar("sigT_scfa","Signal Tail Scale Factor",2.004,1,10) ;
  sigC_frac = new RooRealVar("sigC_frac","Signal Core Fraction",0.838,0,1) ;
  sigO_frac = new RooRealVar("sigO_frac","Signal Outlier Fraction",0.00692,0,0.1) ;
  bkgC_bias = new RooRealVar("bkgC_bias","Backgd Core Bias",-0.1652,-5,+5) ;
  bkgC_scfa = new RooRealVar("bkgC_scfa","Backgd Core Scale Factor",1.0902,0.5,5) ;
  bkgC_frac = new RooRealVar("bkgC_frac","Backgd Core Fraction",0.838,0,1) ;
  outl_bias = new RooRealVar("outl_bias","Outlier Bias",0) ;
  outl_scfa = new RooRealVar("outl_scfa","Outlier Width",10.0) ;
  
  // Mixing, lifetime & dilutions
  sig_tauB = new RooRealVar("sig_tauB","B0 Lifetime",1.548,"ps") ;
  bgC_tauB = new RooRealVar("bgC_tauB","Bch Lifetime",1.623,"ps") ;
  bgL_tauB = new RooRealVar("bgL_tauB","B0 Lifetime",1.3,1.0,2.0,"ps") ;
  dm       = new RooRealVar("dm","B0 mass difference",0.472) ;
  sig_eta  = new RooRealVar("sig_eta","Mistag rate",0.4,0,1) ;
  bgC_eta  = new RooRealVar("bgC_eta","Mistag rate",0.4) ;
  bgL_eta  = new RooRealVar("bgL_eta","Mistag rate",0.4,0,1) ;
  bgP_eta  = new RooRealVar("bgP_eta","Mistag rate",0.4,0,1) ;
    
  // MB signal shape and composition
  bgP_f   = new RooRealVar("bgP_f","Fraction of prompt bkg",0.5,0.,1.) ;
  bgC_f   = new RooRealVar("bgC_f","Fraction of Bch bkg",0.02) ;
  mbMean  = new RooRealVar("mbMean","Fitted B0 mass",5.28,5.25,5.30,"GeV") ;
  mbWidth = new RooRealVar("mbWidth","Gaussian width of B0 mass peak",0.0027,0.002,0.004,"GeV") ;
  mbMax   = new RooRealVar("mbMax","Cutoff point for Argus",5.291,"GeV") ;
  argPar  = new RooRealVar("argPar","Slope parameter for Argus",-35.0,-100.0,0.0) ;
  sigfrac = new RooRealVar("sigfrac","MB signal gaussian fraction",0.5,0.0,1.0) ;
    
  //--- Resolution model components ---
  bkgC_gauss = new RooGaussModel("bkgC_gauss","background core gauss model",*dtReco,*bkgC_bias,*bkgC_scfa,*dtErr,*dtErr) ;
  sigC_gauss = new RooGaussModel("sigC_gauss","signal core gauss model",*dtReco,*sigC_bias,*sigC_scfa,*dtErr,*dtErr) ;
  sigT_gauss = new RooGaussModel("sigT_gauss","signal tail gauss model",*dtReco,*sigT_bias,*sigT_scfa,*dtErr,*dtErr) ;
  outl_gauss = new RooGaussModel("outl_gauss","outlier gauss model",*dtReco,*outl_bias,*outl_scfa,*one,*one) ;
  
  //--- Construct composite signal and gaussian resolution models ---
  sigResModel = new RooAddModel("sigResModel","Signal 3Gauss resolution model",*sigC_gauss,*outl_gauss,*sigT_gauss,*sigC_frac,*sigO_frac) ;
  bkgResModel = new RooAddModel("bkgResModel","Backgd 2Gauss resolution model",*bkgC_gauss,*outl_gauss,*bkgC_frac) ;
  
  //--- Construct signal and Bch,Prompt and lifetime background mixing-decay PDFs ---
  sigModel = new RooBMixDecay("sigModel","signal BMixDecay"      ,*dtReco, *mixState, *sig_tauB ,*dm  , *sig_eta,*sigResModel,RooBMixDecay::DoubleSided) ;
  bgCModel = new RooBMixDecay("bgCModel","bkg Bch BMixDecay"     ,*dtReco, *mixState, *bgC_tauB ,*zero, *bgC_eta,*sigResModel,RooBMixDecay::DoubleSided) ;
  bgLModel = new RooBMixDecay("bgLModel","bkg lifetime BMixDecay",*dtReco, *mixState, *bgL_tauB ,*zero, *bgL_eta,*bkgResModel,RooBMixDecay::DoubleSided) ;
  bgPModel = new RooBMixDecay("bgPModel","bkg prompt BMixDecay"  ,*dtReco, *mixState, *zero2    ,*zero, *bgP_eta,*bkgResModel,RooBMixDecay::DoubleSided) ;
  
  //--- proto = (sigModel+bkgCModel) X Gauss  + (bgLModel+bgPModel) X Argus ---
  sigSum      = new RooAddPdf  ("sigSum",     "signal + bkgBch BMixDecay"  ,*bgCModel,*sigModel,*bgC_f) ;
  bkgSum      = new RooAddPdf  ("bkgSum",     "life + prompt bg BMixDecay" ,*bgPModel,*bgLModel,*bgP_f) ;
  gaussPdf    = new RooGaussian("gaussPdf",   "B0 mass signal Gaussian"    ,*mB,*mbMean,*mbWidth) ;
  argusPdf    = new RooArgusBG ("argusPdf",   "B0 mass background Argus"   ,*mB,*mbMax,*argPar) ;
  sigGaussPdf = new RooProdPdf ("sigGaussPdf","gaussPdf X sigSum"          ,*gaussPdf,*sigSum,1e-15) ;
  bkgArgusPdf = new RooProdPdf ("bkgArgusPdf","argusPdf X bkgSum"          ,*argusPdf,*bkgSum,1e-15) ;
  mixProto    = new RooAddPdf  ("mixProto",   "prototype Mixing fit model" ,*sigGaussPdf,*bkgArgusPdf,*sigfrac) ;
  mbsProto    = new RooAddPdf  ("mbsProto",   "prototype MBshape fit model",*gaussPdf,*argusPdf,*sigfrac) ;

  //****************************************************************************
  //* Construct customized mixing PDFs for each fit category, build global fit *
  //****************************************************************************
  
  //--- Construct fit super category ---
  fitCat = new RooSuperCategory("fitCat","Fit super category",RooArgSet(*tagCat,*runBlock)) ;

  //--- Holder for split leafs
  splitPars = new RooArgSet("split parameters") ;

  //--- Build customized mB-shape PDFs for fit categories ---
  mbsCust = new RooPdfCustomizer(*mbsProto,*fitCat,*splitPars) ;
  mbsCust->splitArgs(RooArgSet(*mbMean,*mbWidth,*argPar,*sigfrac),*tagCat) ;

  //--- Build customized mixing PDFs for fit categories ---
  mixCust = new RooPdfCustomizer(*mixProto,*fitCat,*splitPars) ;
  mixCust->splitArg(*sigC_bias,*fitCat) ;
  mixCust->splitArgs(RooArgSet(*sig_eta,*bgC_eta,*bgL_eta,*bgP_eta,*bgP_f,*mbMean,*mbWidth,*argPar,*sigfrac),*tagCat) ;
  mixCust->splitArgs(RooArgSet(*sigC_scfa,*sigT_bias,*sigC_frac,*sigO_frac,*bkgC_bias,*bkgC_scfa,*bkgC_frac),*runBlock) ;

  //--- Combined the individual fits into a simultaneous fit ---
  mbsModel = new RooSimultaneous("mixModel","Top level Mixing PDF",*fitCat) ;
  mixModel = new RooSimultaneous("mbsModel","Top level MBshape PDF",*fitCat) ;
  TIterator* fcIter = fitCat->MakeIterator() ;
  while(fcIter->Next()) {
    mbsModel->addPdf(*mbsCust->build(fitCat->getLabel()),fitCat->getLabel()) ;
    mixModel->addPdf(*mixCust->build(fitCat->getLabel()),fitCat->getLabel()) ;
    cout << "Adding PDF for fit category " << fitCat->getLabel() << endl ;
  }
  delete fcIter ;

  //--- Read fit parameter configuration from file ---
  mixVars = mixCust->fullParamList(data->get()) ;
  ifstream cf(argv[1]) ;
  mixVars->readFromStream(cf,kFALSE) ;

  //--- Do MBshape prefit ---
  mbsModel->fitTo(*data,"mht0ooo") ;

  //--- Freeze parameters of MBshape fit and print results
  mbsVars = mbsModel->getParameters(data) ;
  mbsVars->Print("v") ;
  mbsVars->setAttribAll("Constant",kTRUE) ;

  RooTrace::dump(cout) ;

  //--- Do BMixing fit ---
  mixModel->fitTo(*data,"mlth0ooo") ;

  //--- Print final results ---
  mixModel->getParameters(data)->Print("v") ;

  RooTrace::dump(cout) ;

  return 0 ;
}
