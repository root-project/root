/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooCPMixFit.cc,v 1.2 2002/02/06 19:45:21 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   17-Oct-2001 WV Created initial version
 *
 *  NOTE: DO NOT COMMIT CHANGES TO THIS FILE WITHOUT CONSULTING THE AUTHOR
 *
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
//
// RooCPMixFit is a utility class that builds the prototype PDFs
// fit CP-golden, mixing and CP-Klong events. It also provides
// a method to read in standard charmonium/breco ascii files
// 
// See RooFitMacros/BBDecays/CPrfc/README for more details

#include "RooFitModels/RooCPMixFit.hh"

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
#include "RooFitCore/RooTruthModel.hh"
#include "RooFitCore/RooAddModel.hh"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooCustomizer.hh"
#include "RooFitCore/RooThresholdCategory.hh"
#include "RooFitCore/RooSuperCategory.hh"
#include "RooFitCore/RooSimultaneous.hh"
#include "RooFitCore/RooSimFitContext.hh"
#include "RooFitCore/RooTrace.hh"
#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooGenericPdf.hh"

#include "RooFitModels/RooUnblindCPAsymVar.hh"
#include "RooFitModels/RooUnblindPrecision.hh"
#include "RooFitModels/RooGaussian.hh"
#include "RooFitModels/RooArgusBG.hh"
#include "RooFitModels/RooGaussModel.hh"
#include "RooFitModels/RooDecay.hh"
#include "RooFitModels/RooBMixDecay.hh"
#include "RooFitModels/RooBCPEffDecay.hh"
#include "RooFitModels/RooExponential.hh"
#include "RooFitModels/RooPolynomial.hh"
#include "RooFitModels/RooGExpModel.hh"


ClassImp(RooCPMixFit)
;


RooCPMixFit::RooCPMixFit(const char* blindString, Double_t cval, Double_t sigma) :
  _blindString(blindString), _cval(cval), _sigma(sigma)
{
  initDataVars() ;
  buildPdfPrototypes() ;
}




RooCPMixFit::~RooCPMixFit() 
{
}



RooDataSet* RooCPMixFit::loadAsciiFiles(const char* asciiFileList, const char* commonPath, const char* opt)
{
  //***********************************************************************
  //* Construct RooDataSet from ASCII file                                *
  //***********************************************************************

  cout << "RooCPMixFit: Reading ascii data files" << endl ;

  // Read data and precalculate derived categories as LValue objects 
  RooDataSet* data = RooDataSet::read(asciiFileList,asciiDataVars,opt,commonPath) ;
  mixState = (RooCategory*) data->addColumn(*mixState_func) ;
  tagFlav  = (RooCategory*) data->addColumn(*tagFlav_func) ;
  tagCat   = (RooCategory*) data->addColumn(*tagCat_func) ;
  tagCatN  = (RooCategory*) data->addColumn(*tagCatN_func) ;
  runBlock = (RooCategory*) data->addColumn(*runBlock_func) ;
  physCat  = (RooCategory*) data->addColumn(*physCat_func) ;

  return data ;
}



void RooCPMixFit::initDataVars()
{
  //***********************************************************************
  //* Definition of ASCII data set and construction of derived quantities *
  //***********************************************************************

  cout << "RooCPMixFit: Initializing data variables" << endl ;
  
  // Constants 
  zero  = new RooRealVar("zero","zero",0.0) ;
  half  = new RooRealVar("half","half",0.5) ;
  one   = new RooRealVar("one","one",1.0) ;
  
  // Dataset variables
  dtReco   = new RooRealVar  ("dtReco"   ,"Delta(t) rec-tag",-20.0,20.0,"ps");           // col# 01
  dtErr    = new RooRealVar  ("dtErr"    ,"Delta(t) Per-Event Error",0.07,2.4999,"ps");  // col# 02
  zRecTrue = new RooRealVar  ("zRecTrue" ,"MC true z of reco B",-1e5,1e5,"um");          // col# 03 
  dzTrue   = new RooRealVar  ("dzTrue"   ,"MC true delta(z)",-3500,3500,"um");           // col# 04 
  zUpsTrue = new RooRealVar  ("zUpsTrue" ,"MC true z of Ups(4S)",-1e5,1e5,"um");         // col# 05 
  dtTrue   = new RooRealVar  ("dtTrue"   ,"MC true delta(t)",-1e5,1e5,"ps")  ;           // col# 06 
  onOffRes = new RooCategory ("onOffRes" ,"on/off Ups(4S) resonance") ;                  // col# 07
  field08  = new RooRealVar  ("field08"  ,"column 08",-999.,999.);                       // col# 08 
  qKstar   = new RooRealVar  ("qKstar"   ,"qKstar",-999.,999.);                          // col# 09
  qtr      = new RooRealVar  ("qtr"      ,"qtr",-999.,999.);                             // col# 10
  ftr      = new RooRealVar  ("ftr"      ,"ftr",-999.,999.);                             // col# 11
  pTagB0   = new RooRealVar  ("pTagB0"   ,"Tag side B0 flavour probability",0.,1.01) ;   // col# 12
  pTagB0Tru= new RooRealVar  ("pTagB0Tru","True tag side B0 flavour probability",0.,1000);//col# 13 
  tagCatRaw= new RooCategory ("tagCatRaw","Raw Tagging Category") ;                      // col# 14
  pRecB0   = new RooRealVar  ("pRecB0"   ,"Reco side B0 flavour probablity",0.,1.01) ;   // col# 15
  pRecB0Tru= new RooRealVar  ("pRecB0Tru","True reco side B0 flavour probablity",0.,1000);//col# 16 
  field17  = new RooRealVar  ("field17"  ,"column 17",-999.,999.);                       // col# 17
  brecMode = new RooCategory ("brecMode" ,"Decay mode of reco B");                       // col# 18
  pSig     = new RooRealVar  ("pSig"     ,"prob that the event is signal",0.,1.);        // col# 19
  pType1   = new RooRealVar  ("pType1"   ,"prob that the event is of BG type 1",0.,1.);  // col# 20
  pType2   = new RooRealVar  ("pType2"   ,"prob that the event is of BG type 2",-20.,20.); // col# 21
  pType3   = new RooRealVar  ("pType3"   ,"prob that the event is of BG type 3",0.,1.);  // col# 22
  pType4   = new RooRealVar  ("pType4"   ,"prob that the event is of BG type 4",0.,1.);  // col# 23
  pType5   = new RooRealVar  ("pType5"   ,"prob that the event is of BG type 5",0.,1.);  // col# 24
  pType6   = new RooRealVar  ("pType6"   ,"prob that the event is of BG type 6",0.,1.);  // col# 25
  pType7   = new RooRealVar  ("pType7"   ,"prob that the event is of BG type 7",0.,1.);  // col# 26
  pType8   = new RooRealVar  ("pType8"   ,"prob that the event is of BG type 8",0.,1.);  // col# 27
  deltaE   = new RooRealVar  ("deltaE"   ,"Delta(E)_ES",-0.1,0.1,"GeV");                 // col# 28
  mB       = new RooRealVar  ("mB"       ,"E_Beam Substituted Mass",5.200001,5.30,"GeV");// col# 29
  runNumber= new RooRealVar  ("runNumber","run number",0,0,30000);                       // col# 30
  tstamph  = new RooStringVar("tstamph"  ,"time stamp upper ID","");                     // col# 31
  tstampl  = new RooStringVar("tstampl"  ,"time stamp lower ID","");                     // col# 32

  // On-off resonance states
  onOffRes->defineType("1.0",1) ;
  onOffRes->defineType("Unknown",0) ;
  onOffRes->defineType("OnRes", 1);
  onOffRes->defineType("OffRes",2);
  
  // Tagging categories as they appear in the ascii files 
  tagCatRaw->clearTypes();
  tagCatRaw->defineType("NoTag",        0);
  tagCatRaw->defineType("ElKaon",      11);
  tagCatRaw->defineType("MuKaon",      12);
  tagCatRaw->defineType("Electron",    14);
  tagCatRaw->defineType("Muon",        15);
  tagCatRaw->defineType("Kaon",        13);
  tagCatRaw->defineType("NetTagger-1", 22);
  tagCatRaw->defineType("NetTagger-2", 23);
  
  // B decay modes as they appear in the ascii files
  defineBDecayModes(*brecMode) ;
  
  // Derived quantities from data set 
  mixProb = new RooFormulaVar("mixProb","1-2*abs(pTagB0-pRecB0)",RooArgSet(*pTagB0,*pRecB0)) ;
  
  mixState_func = new RooThresholdCategory("mixState","B0-B0bar Mixing State",*mixProb,"Mixed",-1) ;
  mixState_func->addThreshold(0.,"Unmixed",+1) ;  

  tagFlav_func = new RooThresholdCategory("tagFlav","B0 Tagged Flavour State",*pTagB0,"B0",1) ;
  tagFlav_func->addThreshold(0.5,"B0bar",-1) ;  
  
  tagCat_func = new RooMappedCategory("tagCat","Condensed Tagging Category",*tagCatRaw,"Lep") ; 
  tagCat_func->map("El*"        ,"Lep") ;
  tagCat_func->map("Mu*"        ,"Lep") ;
  tagCat_func->map("Kaon"       ,"Kao") ;
  tagCat_func->map("NetTagger-1","NT1") ;
  tagCat_func->map("NetTagger-2","NT2") ;
  tagCat = (RooCategory*) tagCat_func->createFundamental() ;

  tagCatL = new RooMappedCategory("tagCatL","Lepton/Non-lepton tagging category",*tagCat,"NoLep") ;
  tagCatL->map("Lep","Lep") ;

  tagCatN_func = (RooMappedCategory*) tagCat_func->clone("tagCatN") ;
  tagCatN_func->map("NoTag","Non") ;

  physCat_func = new RooMappedCategory("physCat","Physics category",*brecMode,"Unknown") ;
  physCat_func->map("B0->D*"            ,"BMix") ;
  physCat_func->map("B0->JPsiKst0*"     ,"BMix") ;
  physCat_func->map("B0->JPsiKs*"       ,"Gold") ;
  physCat_func->map("B0->Psi2sKs*"      ,"Gold") ;
  physCat_func->map("B0->Chic1Ks*"      ,"Gold") ;
  physCat_func->map("B0->JPsiKl, JPsi->EE, IFR"     ,"KlIfrE") ;
  physCat_func->map("B0->JPsiKl, JPsi->EE, IFREMC"  ,"KlIfrE") ;
  physCat_func->map("B0->JPsiKl, JPsi->MuMu, IFR"   ,"KlIfrM") ;
  physCat_func->map("B0->JPsiKl, JPsi->MuMu, IFREMC","KlIfrM") ;
  physCat_func->map("B0->JPsiKl, JPsi->EE, EMC"     ,"KlEmcE") ;
  physCat_func->map("B0->JPsiKl, JPsi->MuMu, EMC"   ,"KlEmcM") ;
  
  runBlock_func = new RooThresholdCategory("runBlock","Run block (I or II)",*runNumber,"Run2") ;
  runBlock_func->addThreshold(18000,"Run1") ;

  mixState = (RooCategory*) mixState_func->createFundamental() ;
  tagFlav  = (RooCategory*) tagFlav_func->createFundamental() ;
  tagCatN  = (RooCategory*) tagCatN_func->createFundamental() ;
  runBlock = (RooCategory*) runBlock_func->createFundamental() ;
  physCat  = (RooCategory*) physCat_func->createFundamental() ;

  // Take ownership of all created objects 
  _owned.addOwned(RooArgList(*dtErr,*dtReco,*mB,*pTagB0,*pRecB0,*runNumber,*tagCatRaw)) ;
  _owned.addOwned(RooArgList(*mixProb,*mixState,*tagCat_func,*runBlock_func,*mixState,*tagCat,*runBlock)) ;

  // Build list of ascii data variables
  asciiDataVars.add(RooArgSet(*dtReco,*dtErr,*zRecTrue,*dzTrue,*zUpsTrue,*dtTrue,*onOffRes,*field08)) ;
  asciiDataVars.add(RooArgSet(*qKstar,*qtr,*ftr,*pTagB0,*pTagB0Tru,*tagCatRaw,*pRecB0,*pRecB0Tru)) ;
  asciiDataVars.add(RooArgSet(*field17,*brecMode,*pSig,*pType1,*pType2,*pType3,*pType4,*pType5)) ;
  asciiDataVars.add(RooArgSet(*pType6,*pType7,*pType8,*deltaE,*mB,*runNumber,*tstamph,*tstampl)) ;

  // Build useful subset of ascii variables
  selDataVars.add(RooArgSet(*dtReco,*dtErr,*onOffRes,*pTagB0,*tagCatRaw,*pRecB0,*brecMode,*deltaE)) ;
  selDataVars.add(RooArgSet(*mB,*runNumber,*tagCat,*tagCatN,*runBlock,*mixState,*tagFlav,*physCat)) ;
}



const RooArgSet& RooCPMixFit::protoPdfSet() 
{
  //***********************************************************************
  //* Return set of proto-type PDFs                                       *
  //***********************************************************************

  return _pdfProtoSet ;
}




  
void RooCPMixFit::buildDeltatPdfs() 
{
  //***********************************************************************
  //* Construction of DeltaT components of physics model PDFs             *
  //***********************************************************************

  cout << "RooCPMixFit: Building deltaT PDF components" << endl ;

  // Fit parameters 

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
  kbgC_scfa = new RooRealVar("kbgC_scfa","JPsiKL Backgd Core Scale Factor",1.0902,0.5,5) ;
  kbgC_bias = new RooRealVar("kbgC_bias","JPsiKL Backgd Core Bias",-0.1652,-5,+5) ;
  kbgC_frac = new RooRealVar("kbgC_frac","JpsiKL Backgd Core Fraction",0.838,0,1) ;
  kbgT_scfa = new RooRealVar("kbgT_scfa","JPsiKL Backgd Tail Scale Factor",1.0902,0.5,5) ;
  kbgT_bias = new RooRealVar("kbgT_bias","JPsiKL Backgd Tail Bias",-0.1652,-5,+5) ;
  kbgO_frac = new RooRealVar("kbgO_frac","JpsiKL Backgd Outlier Fraction",0.838,0,1) ;
  outl_bias = new RooRealVar("outl_bias","Outlier Bias",0) ;
  outl_scfa = new RooRealVar("outl_scfa","Outlier Width",10.0) ;
  
  // Mixing, lifetime & dilutions
  sig_tauB  = new RooRealVar("sig_tauB","B0 Lifetime",1.548,"ps") ;
  bgC_tauB  = new RooRealVar("bgC_tauB","Bch Lifetime",1.623,"ps") ;
  bgL_tauB  = new RooRealVar("bgL_tauB","B0 Lifetime",1.3,1.0,2.0,"ps") ;
  bgKl_tauB = new RooRealVar("bgKl_tauB","B0 Lifetime",1.3,1.0,2.0,"ps") ;
  dm        = new RooRealVar("dm","B0 mass difference",0.472) ;
 
  sig_eta  = new RooRealVar("sig_eta","Mistag rate",0.4,0,1) ;
  sig_deta = new RooRealVar("sig_deta","Delta mistag rate",0.) ;
  bgC_eta  = new RooRealVar("bgC_eta","Mistag rate",0.4) ;
  bgL_eta  = new RooRealVar("bgL_eta","Mistag rate",0.4,0,1) ;
  bgP_eta  = new RooRealVar("bgP_eta","Mistag rate",0.4,0,1) ;
  
  // CP parameters
  sin2b         = new RooRealVar("sin2b","sin(2*beta)",-1.0,4.0) ;
  alambda       = new RooRealVar("alambda","abs(lambda)",1.0) ;
  gold_cpev     = new RooRealVar("gold_cpev","CP eigenvalue of golden modes",-1.0) ;
  psikl_cpev    = new RooRealVar("psikl_cpev","CP eigenvalue of golden modes", 1.0) ;
  psikstar_cpev = new RooRealVar("psikstar_cpev","CP eigenvalue of golden modes",-0.68) ;
  effratio      = new RooRealVar("effratio","B0/B0Bar efficiency ratio",1.0) ;
  
  // Sin2beta blinding implementation
  s2b_bs   = new RooCategory("s2b_bs","sin2beta blinding state") ;
  sin2b_ub = new RooUnblindPrecision("sin2b_ub","Unblinded Sin2beta",_blindString,_cval,_sigma,*sin2b,*s2b_bs,kTRUE) ;
  s2b_bs->defineType("Unblind",0) ;
  s2b_bs->defineType("Blind",1) ;
  if (_blindString=="DummyDefaultBlindingString") {
    s2b_bs->setLabel("Unblind") ;
  } else {
    cout << "RooCPMixFit: Blinding activated with string '" << _blindString << "'" << endl ;
    s2b_bs->setLabel("Blind") ;
  }

  // Sample composition
  bgP_f     = new RooRealVar("bgP_f","Fraction of prompt bkg",0.5,0.,1.) ;
  bgC_f     = new RooRealVar("bgC_f","Fraction of Bch bkg",0.02) ;
      
  // Resolution model components 
  sigC_gauss = new RooGaussModel("sigC_gauss","signal core gauss model",*dtReco,*sigC_bias,*sigC_scfa,*dtErr) ;
  sigT_gauss = new RooGaussModel("sigT_gauss","signal tail gauss model",*dtReco,*sigT_bias,*sigT_scfa,*dtErr) ;
  bkgC_gauss = new RooGaussModel("bkgC_gauss","background core gauss model",*dtReco,*bkgC_bias,*bkgC_scfa,*dtErr) ;
  kbgC_gauss = new RooGaussModel("kbgC_gauss","JPsiKLong backgnd core gauss model",*dtReco,*kbgC_bias,*kbgC_scfa,*dtErr) ;
  kbgT_gauss = new RooGaussModel("kbgT_gauss","JPsiKLong backgnd tail gauss model",*dtReco,*kbgT_bias,*kbgT_scfa,*dtErr) ;
  outl_gauss = new RooGaussModel("outl_gauss","outlier gauss model",*dtReco,*outl_bias,*outl_scfa) ;

  // Construct composite signal and gaussian resolution models 
  sigResModel = new RooAddModel("sigResModel","Signal 3Gauss resolution model",
				RooArgList(*sigC_gauss,*outl_gauss,*sigT_gauss),RooArgList(*sigC_frac,*sigO_frac)) ;
  bkgResModel = new RooAddModel("bkgResModel","Backgd 2Gauss resolution model",
				RooArgList(*bkgC_gauss,*outl_gauss),*bkgC_frac) ;
  kbgResModel = new RooAddModel("kbgResModel","JPsiKLong Backgd 3Gauss resolution model",
				RooArgList(*kbgC_gauss,*outl_gauss,*kbgT_gauss),RooArgList(*kbgC_frac,*kbgO_frac)) ;

  // Construct signal and Bch,Prompt and lifetime background mixing-decay PDFs 
  bmixSigModel = new RooBMixDecay("bmixSigModel","BMix DeltaT signal PDF",*dtReco, *mixState, *tagFlav,
				  *sig_tauB ,*dm  , *sig_eta, *sig_deta, *sigResModel,RooBMixDecay::DoubleSided) ;
  bmixBgCModel = new RooBMixDecay("bmixBgCModel","BMix DeltaT B+/- background PDF",*dtReco, *mixState, *tagFlav,
				  *bgC_tauB ,*zero, *bgC_eta, *zero,     *sigResModel,RooBMixDecay::DoubleSided) ;
  bmixBgLModel = new RooBMixDecay("bmixBgLModel","BMix DeltaT lifetime background PDF",*dtReco, *mixState, *tagFlav,
				  *bgL_tauB ,*zero, *bgL_eta,*zero,      *bkgResModel,RooBMixDecay::DoubleSided) ;
  bmixBgPModel = new RooBMixDecay("bmixBgPModel","BMix DeltaT prompt background PDF",*dtReco, *mixState, *tagFlav,
				  *zero     ,*zero, *bgP_eta,*zero,      *bkgResModel,RooBMixDecay::DoubleSided) ;

  // Construct BMixing signal and background components
  dtSigBMix = new RooAddPdf  ("dtSigBMix",     "signal + bkgBch BMixDecay"  ,*bmixBgCModel,*bmixSigModel,*bgC_f) ;
  dtBkgBMix = new RooAddPdf  ("dtBkgBMix",     "life + prompt bg BMixDecay" ,*bmixBgPModel,*bmixBgLModel,*bgP_f) ;

  // Construct signal and Bch,Prompt and lifetime background CPGold PDFs 
  goldSigModel = new RooBCPEffDecay("goldSigModel","CPGold DeltaT signal PDF",             *dtReco,*tagFlav,*sig_tauB,*dm,  
				    *sig_eta,*gold_cpev,*alambda,*sin2b_ub,*effratio,*sig_deta,*sigResModel,RooBCPEffDecay::DoubleSided) ;
  goldBgCModel = new RooBCPEffDecay("goldBgCModel","CPGold DeltaT B+/- background PDF",    *dtReco,*tagFlav,*sig_tauB,*dm,  
				    *sig_eta,*gold_cpev,*one,    *zero ,*effratio,*sig_deta,*sigResModel,RooBCPEffDecay::DoubleSided) ;
  goldBgLModel = new RooBCPEffDecay("goldBgLModel","CPGold DeltaT lifetime background PDF",*dtReco,*tagFlav,*sig_tauB,*dm,  
				    *half,   *gold_cpev,*one,    *zero ,*effratio,*zero,    *bkgResModel,RooBCPEffDecay::DoubleSided) ;
  goldBgPModel = new RooBCPEffDecay("goldBgPModel","CPGold DeltaT prompt background PDF",  *dtReco,*tagFlav,*zero,  *zero,  
				    *half,   *gold_cpev,*one,    *zero ,*effratio,*zero,   *bkgResModel,RooBCPEffDecay::DoubleSided) ;

  // Construct CPGold signal and background components
  dtSigGold = new RooAddPdf  ("dtSigGold",     "signal + bkgBch CPGold"  ,*goldBgCModel,*goldSigModel,*bgC_f) ;
  dtBkgGold = new RooAddPdf  ("dtBkgGold",     "life + prompt bg CPGold" ,*goldBgPModel,*goldBgLModel,*bgP_f) ;

  // Construct signal and various backgrounds for JpsiKL PDFs
  dtSigKlong   = new RooBCPEffDecay("klongSigModel","JPsiKLong DeltaT signal PDF",                 *dtReco,*tagFlav,*sig_tauB,*dm,  
				       *sig_eta,*psikl_cpev   ,*one ,*sin2b_ub,*effratio,*sig_deta,*sigResModel,RooBCPEffDecay::DoubleSided) ;
  dtBgKstKlong = new RooBCPEffDecay("klongBgKstModel","JPsiKLong DeltaT JpsiK* background PDF",    *dtReco,*tagFlav,*sig_tauB,*dm,  
				       *sig_eta,*psikstar_cpev,*one, *sin2b_ub,*effratio,*sig_deta,*sigResModel,RooBCPEffDecay::DoubleSided) ;
  dtBgKshKlong = new RooBCPEffDecay("klongBgKshModel","JPsiKLong DeltaT JpsiKs background PDF",    *dtReco,*tagFlav,*sig_tauB,*dm,  
				       *sig_eta,*gold_cpev    ,*one, *sin2b_ub,*effratio,*sig_deta,*sigResModel,RooBCPEffDecay::DoubleSided) ;
  dtBgNcpKlong = new RooBCPEffDecay("klongBgNcpModel","JPsiKLong DeltaT non-CP background PDF",    *dtReco,*tagFlav,*sig_tauB,*dm,  
				       *half,   *zero         ,*one, *zero, *effratio,*zero,    *sigResModel,RooBCPEffDecay::DoubleSided) ;
  dtBgLKlong   = new RooBCPEffDecay("klongBgLModel","JPsiKLong DeltaT non-psi lifetime background PDF",*dtReco,*tagFlav,*bgKl_tauB,*dm,  
				       *half,   *zero         ,*one, *zero ,*effratio,*zero,    *kbgResModel,RooBCPEffDecay::DoubleSided) ;
  dtBgPKlong   = new RooBCPEffDecay("klongBgPModel","JPsiKLong DeltaT non-psi prompt background PDF",  *dtReco,*tagFlav,*zero,  *zero,  
				       *half,   *zero         ,*one, *zero ,*effratio,*zero,    *kbgResModel,RooBCPEffDecay::DoubleSided) ;
  //                                    eta      cpev           |l|   sin2b    effr   d(eta)      rmodel


  // Take ownership of all constructed objects
  _owned.addOwned(RooArgSet(*zero,*half,*one,*bgP_f,*bgC_f)) ;
  _owned.addOwned(RooArgSet(*sigC_bias,*sigT_bias,*sigC_scfa,*sigT_scfa,*sigC_frac)) ;
  _owned.addOwned(RooArgSet(*sigO_frac,*bkgC_bias,*bkgC_scfa,*bkgC_frac,*outl_bias,*outl_scfa)) ;
  _owned.addOwned(RooArgSet(*sig_tauB,*bgC_tauB,*bgL_tauB,*dm,*sig_eta,*bgC_eta,*bgL_eta,*bgP_eta)) ;
  _owned.addOwned(RooArgSet(*bkgC_gauss,*sigC_gauss,*sigT_gauss,*outl_gauss,*sigResModel,*bkgResModel)) ;
  _owned.addOwned(RooArgSet(*bmixSigModel,*bmixBgCModel,*bmixBgLModel,*bmixBgPModel,*dtSigBMix,*dtBkgBMix)) ;

}





void RooCPMixFit::buildSelectionPdfs() 
{
  //***********************************************************************
  //* Construction of event selection components of physics model PDFs    *
  //***********************************************************************

  cout << "RooCPMixFit: Building signal event selection components" << endl ;

  // * * * MB for golden modes * * *

  // MB signal shape and composition
  mbMean    = new RooRealVar("mbMean","Fitted B0 mass",5.28,5.25,5.30,"GeV") ;
  mbWidth   = new RooRealVar("mbWidth","Gaussian width of B0 mass peak",0.0027,0.002,0.004,"GeV") ;
  mbMax     = new RooRealVar("mbMax","Cutoff point for Argus",5.291,"GeV") ;
  argPar    = new RooRealVar("argPar","Slope parameter for Argus",-35.0,-100.0,0.0) ;
  mbSigFrac = new RooRealVar("sigfrac","MB signal gaussian fraction",0.5,0.0,1.0) ;

  // MB event selection signal and background PDFs
  mbSig    = new RooGaussian("mbSig","B0 mass signal Gaussian"    ,*mB,*mbMean,*mbWidth) ;
  mbBkg    = new RooArgusBG ("mbBkg","B0 mass background Argus"   ,*mB,*mbMax,*argPar) ;
  mbPdf    = new RooAddPdf  ("mbPdf","MBshape PDF" ,*mbSig,*mbBkg,*mbSigFrac) ;


  // * * * DeltaE for JpsiKL * * *
  klSig_frac   = new RooRealVar("klSig_frac"  ,"KLong signal fraction",0.4) ;
  klBgCcK_frac = new RooRealVar("klBgCcK_frac","KLong ChicKL background fraction",0.015) ;
  klKstBg_frac = new RooRealVar("klKstBg_frac","KLong JpsiK*0 background fraction",0.091) ;
  klKshBg_frac = new RooRealVar("klKshBg_frac","KLong JpsiKS background fraction",0.064) ;
  klNPLBg_frac = new RooRealVar("klNPLBg_frac","KLong nonPsi lifetime background fraction",0.02) ;
  klNPPBg_frac = new RooRealVar("klNPPBg_frac","KLong nonPsi prompt background fraction",0.043) ;

  // DeltaE signal shapes for JpsiKL EMC/IFR signal
  deMeV         = new RooFormulaVar("deMeV","1000*deltaE",*deltaE) ;
  dePrime       = new RooFormulaVar("dePrime","5280 - 1000*deltaE",*deltaE) ;
  deGMeanSig    = new RooRealVar("deGMeanSig"  ,"Mean of Gaussian of DE signal shape",0.) ;
  deGWidthSig   = new RooRealVar("deGWidthSig" ,"Width of Gaussian of DE signal shape",3.) ;
  deG2MeanSig   = new RooRealVar("deG2MeanSig"  ,"Mean of Gaussian of DE signal shape",0.) ;
  deG2WidthSig  = new RooRealVar("deG2WidthSig" ,"Width of Gaussian of DE signal shape",3.) ;
  deACutoffSig  = new RooRealVar("deACutoffSig","Cutoff of Argus of DE signal shape",5290.) ;
  deAKappaSig   = new RooRealVar("deAKappaSig" ,"Slope of Argus of DE signal shape",-68.) ;
  deGFracSig    = new RooRealVar("deGFracSig"  ,"Fraction of gauss component in DE signal shape",0.9) ;
  deG2FracSig   = new RooRealVar("deG2FracSig"  ,"Fraction of gauss component in DE signal shape",0) ;
  deGaussSig    = new RooGaussian("deGaussSig","Gaussian component of signal DE shape",*deMeV,*deGMeanSig,*deGWidthSig) ;
  deGauss2Sig   = new RooGaussian("deGauss2Sig","2nd Gaussian component of signal DE shape",*deMeV,*deG2MeanSig,*deG2WidthSig) ;
  deArgusSig    = new RooArgusBG("deArgusSig","Argus component of signal DE shape",*dePrime,*deACutoffSig,*deAKappaSig) ;
  deSigKlongRaw = new RooAddPdf("deSigKlongRaw","signal DE shape",RooArgList(*deGaussSig,*deGauss2Sig,*deArgusSig),
				                                  RooArgList(*deGFracSig,*deG2FracSig)) ;
  deSigKlong    = new RooGenericPdf("deSigKlong","(abs(@0)<0.010001)*@1",RooArgList(*deltaE,*deSigKlongRaw)) ;

  // DeltaE signal shape  for JpsiKL EMC/IFR inclusive psi background
  dePol1IPbg     = new RooRealVar("dePol1IPbg","1st order coefficient of inclusive psi polynomial background", 2.066e-2) ;
  dePol2IPbg     = new RooRealVar("dePol2IPbg","2nd order coefficient of inclusive psi polynomial background",-3.456e-3) ;
  dePol3IPbg     = new RooRealVar("dePol3IPbg","3rd order coefficient of inclusive psi polynomial background", 6.921e-6) ;
  dePol4IPbg     = new RooRealVar("dePol4IPbg","4th order coefficient of inclusive psi polynomial background", 8.947e-7) ;
  deIPbgKlongRaw = new RooPolynomial("deIPbgKlongRaw","Inclusive Psi background DE shape",*deMeV,RooArgList(*dePol1IPbg,*dePol2IPbg,*dePol3IPbg,*dePol4IPbg)) ;
  deIPbgKlong    = new RooGenericPdf("deIPbgKlong","(abs(@0)<0.010001)*@1",RooArgList(*deltaE,*deIPbgKlongRaw)) ;

  // DeltaE signal shape  for JpsiKL EMC/IFR psi sideband background
  deACutoffSBbg  = new RooRealVar("deACutoffSBbg","Cutoff of Argus of DE psi sideband background shape",5288.9) ;
  deAKappaSBbg   = new RooRealVar("deAKappaSBbg" ,"Slope of Argus of DE psi sideband background shape",-73.0) ;
  deSBbgKlongRaw = new RooArgusBG("deSBbgKlongRaw","Psi sideband background DE shape",*dePrime,*deACutoffSBbg,*deAKappaSBbg) ;
  deSBbgKlong    = new RooGenericPdf("deSBbgKlong","(abs(@0)<0.010001)*@1",RooArgList(*deltaE,*deSBbgKlongRaw)) ;

  // DeltaE signal shape  for JpsiKL EMC/IFR psi ks -> pi0pi0 background
  deGMeanKSBg    = new RooRealVar("deGMeanKSBg"  ,"Mean of Gaussian of DE JpsiKS00 background shape",0.) ;
  deGWidthKSBg   = new RooRealVar("deGWidthKSBg" ,"Width of Gaussian of DE JpsiKS00 background shape",5.) ;
  deACutoffKSBg  = new RooRealVar("deACutoffKSBg","Cutoff of Argus of DE JpsiKS00 background shape",5290.) ;
  deAKappaKSBg   = new RooRealVar("deAKappaKSBg" ,"Slope of Argus of DE JpsiKS00 background shape",-163.) ;
  deGFracKSBg    = new RooRealVar("deGFracKSBg"  ,"Fraction of gauss component in DE JpsiKS00 background shape",0.7) ;
  deGaussKSBg    = new RooGaussian("deGaussKSBg","Gaussian component of JpsiKS00 background DE shape",*deMeV,*deGMeanKSBg,*deGWidthKSBg) ;
  deArgusKSBg    = new RooArgusBG("deArgusKSBg","Argus component of JpsiKS00 background DE shape",*dePrime,*deACutoffKSBg,*deAKappaKSBg) ;
  deKSBgKlongRaw = new RooAddPdf("deKSBgKlongRaw","JpsiKS00 background DE shape",*deGaussKSBg,*deArgusKSBg,*deGFracKSBg) ;
  deKSBgKlong    = new RooGenericPdf("deKSBgKlong","(abs(@0)<0.010001)*@1",RooArgList(*deltaE,*deKSBgKlongRaw)) ;

  // Take ownership of all constructed objects
  _owned.addOwned(RooArgSet(*mbMean,*mbWidth,*mbMax,*argPar,*mbSigFrac)) ;
  _owned.addOwned(RooArgSet(*mbSig,*mbBkg)) ;
}





void RooCPMixFit::buildPdfPrototypes() 
{
  //***********************************************************************
  //* Construction of physics model PDFs                                  *
  //***********************************************************************

  buildDeltatPdfs() ;
  buildSelectionPdfs() ;

  // build BMixing DeltaT x EvtSel PDF  
  BMixingSig  = new RooProdPdf ("BMixingSig","gaussPdf X sigSum",*mbSig,*dtSigBMix,1e-15) ;
  BMixingBkg  = new RooProdPdf ("BMixingBkg","argusPdf X bkgSum",*mbBkg,*dtBkgBMix,1e-15) ;
  BMixing     = new RooAddPdf  ("BMixing"   ,"BMixing PDF" ,*BMixingSig,*BMixingBkg,*mbSigFrac) ;  

  // build CPGold DeltaT x EvtSel PDF  
  CPGoldSig  = new RooProdPdf ("CPGoldSig","gaussPdf X sigSum",*mbSig,*dtSigGold,1e-15) ;
  CPGoldBkg  = new RooProdPdf ("CPGoldBkg","argusPdf X bkgSum",*mbBkg,*dtBkgGold,1e-15) ;
  CPGold     = new RooAddPdf  ("CPGold"   ,"CPGold PDF"  ,*CPGoldSig,*CPGoldBkg,*mbSigFrac) ;  

  // build KLong DeltaT x EvtSel PDF  
  KLongSig   = new RooProdPdf ("KLongSig"  ,"Signal klongDE x klong-Dt",*dtSigKlong,*deSigKlong) ;
  KLongCcKBg = new RooProdPdf ("KLongCcKBg","ChicKL Bkg klongDE x klong-Dt",*dtSigKlong,*deIPbgKlong) ;
  KLongKstBg = new RooProdPdf ("KLongKstBg","JpsiK* Bkg klongDE x klong-Dt",*dtBgKstKlong,*deIPbgKlong) ;
  KLongKshBg = new RooProdPdf ("KLongKshBg","JpsiKS Bkg klongDE x klong-Dt",*dtBgKshKlong,*deKSBgKlong) ;
  KLongOthBg = new RooProdPdf ("KLongOthBg","IncPsi Bkg klongDE x klong-Dt",*dtBgNcpKlong,*deIPbgKlong) ;
  KLongNPLBg = new RooProdPdf ("KLongNPLBg","nonPsi lifetime Bkg klongDE x klong-Dt",*dtBgLKlong,*deSBbgKlong) ;
  KLongNPPBg = new RooProdPdf ("KLongNPPBg","nonPsi prompt Bkg   klongDE x klong-Dt",*dtBgPKlong,*deSBbgKlong) ;
  KLong      = new RooAddPdf  ("KLong","KLong PDF",
			       RooArgList(*KLongSig  ,*KLongCcKBg  ,*KLongKstBg  ,*KLongKshBg  ,*KLongNPLBg  ,*KLongNPPBg,*KLongOthBg),
			       RooArgList(*klSig_frac,*klBgCcK_frac,*klKstBg_frac,*klKshBg_frac,*klNPLBg_frac,*klNPPBg_frac)) ;
      
  // Export complete models
  _pdfProtoSet.add(*BMixing) ;
  _pdfProtoSet.add(*CPGold) ;
  _pdfProtoSet.add(*KLong) ;
  _pdfProtoSet.add(*mbPdf) ;

  // Export models for signal-only
  _pdfProtoSet.add(*BMixingSig) ;
  _pdfProtoSet.add(*CPGoldSig) ;
  _pdfProtoSet.add(*KLongSig) ;
  _pdfProtoSet.add(*mbSig) ;
}





void RooCPMixFit::defineBDecayModes(RooCategory& cat)
{
  // Define all the exclusive B0 and B+/- decay modes
  cat.defineType("B0->JPsiKs, JPsi->EE, Ks->Pi+Pi-",1011) ;
  cat.defineType("B0->JPsiKs, JPsi->MuMu, Ks->Pi+Pi-",1012) ;
  cat.defineType("B0->JPsiKs, JPsi->EE, Ks->Pi0Pi0",1013) ;
  cat.defineType("B0->JPsiKs, JPsi->MuMu, Ks->Pi0Pi0",1014) ;
  cat.defineType("B+->JPsiK+, JPsi->EE",1015) ;
  cat.defineType("B+->JPsiK+, JPsi->MuMu",1016) ;
  cat.defineType("B0->Psi2sKs, Psi2s->EE, Ks->Pi+Pi-",1021) ;
  cat.defineType("B0->Psi2sKs, Psi2s->MuMu, Ks->Pi+Pi-",1022) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi+Pi-, JPsi->EE, Ks->Pi+Pi-",1023) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi+Pi-, JPsi->MuMu, Ks->Pi+Pi-",1024) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi0Pi0, JPsi->EE, Ks->Pi+Pi-",1025) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi0Pi0, JPsi->MuMu, Ks->Pi+Pi-",1026) ;
  cat.defineType("B0->Psi2sKs, Psi2s->EE, Ks->Pi0Pi0",1031) ;
  cat.defineType("B0->Psi2sKs, Psi2s->MuMu, Ks->Pi0Pi0",1032) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi+Pi-, JPsi->EE, Ks->Pi0Pi0",1033) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi+Pi-, JPsi->MuMu, Ks->Pi0Pi0",1034) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi0Pi0, JPsi->EE, Ks->Pi0Pi0",1035) ;
  cat.defineType("B0->Psi2sKs, Psi2s->JPsiPi0Pi0, JPsi->MuMu, Ks->Pi0Pi0",1036) ;
  cat.defineType("B+->Psi2sK+, Psi2s->EE",1041) ;
  cat.defineType("B+->Psi2sK+, Psi2s->MuMu",1042) ;
  cat.defineType("B+->Psi2sK+, Psi2s->JPsiPi+Pi-, JPsi->EE",1043) ;
  cat.defineType("B+->Psi2sK+, Psi2s->JPsiPi+Pi-, JPsi->MuMu",1044) ;
  cat.defineType("B+->Psi2sK+, Psi2s->JPsiPi0Pi0, JPsi->EE",1045) ;
  cat.defineType("B+->Psi2sK+, Psi2s->JPsiPi0Pi0, JPsi->MuMu",1046) ;
  //cat.defineType("B0->JPsiKl, JPsi->EE",1051) ;
  //cat.defineType("B0->JPsiKl, JPsi->MuMu",1052) ;
  cat.defineType("B0->JPsiKl, JPsi->EE, IFR",1051) ;
  cat.defineType("B0->JPsiKl, JPsi->MuMu, IFR",1052) ;
  cat.defineType("B0->JPsiKl, JPsi->EE, EMC",1053) ;
  cat.defineType("B0->JPsiKl, JPsi->MuMu, EMC",1054) ;
  cat.defineType("B0->JPsiKl, JPsi->EE, IFREMC",1055) ;
  cat.defineType("B0->JPsiKl, JPsi->MuMu, IFREMC",1056) ;
  cat.defineType("B0->JPsiKst0, JPsi->EE, Kst0->K+Pi-",1061) ;
  cat.defineType("B0->JPsiKst0, JPsi->MuMu, Kst0->K+Pi-",1062) ;
  cat.defineType("B0->JPsiKst0, JPsi->EE, Kst0->KsPi0",1063) ;
  cat.defineType("B0->JPsiKst0, JPsi->MuMu, Kst0->KsPi0",1064) ;
  cat.defineType("B+->JPsiK*+, JPsi->EE, K*+->K+Pi0",1065) ;
  cat.defineType("B+->JPsiK*+, JPsi->MuMu, K*+->K+Pi0",1066) ;
  cat.defineType("B+->JPsiK*+, JPsi->EE, K*+->KsPi+",1067) ;
  cat.defineType("B+->JPsiK*+, JPsi->MuMu, K*+->KsPi+",1068) ;
  cat.defineType("B0->Chic1Ks, Chic1->JPsiGamma, JPsi->EE, Ks->Pi+Pi-",1071) ;
  cat.defineType("B0->Chic1Ks, Chic1->JPsiGamma, JPsi->MuMu, Ks->Pi+Pi-",1072) ;
  cat.defineType("B0->Chic2Ks, Chic2->JPsiGamma, JPsi->EE, Ks->Pi+Pi-",1073) ;
  cat.defineType("B0->Chic2Ks, Chic2->JPsiGamma, JPsi->MuMu, Ks->Pi+Pi-",1074) ;
  cat.defineType("B+->Chic1K+, Chic1->JPsiGamma, JPsi->EE",1075) ;
  cat.defineType("B+->Chic1K+, Chic1->JPsiGamma, JPsi->MuMu",1076) ;
  cat.defineType("B+->Chic2K+, Chic2->JPsiGamma, JPsi->EE",1077) ;
  cat.defineType("B+->Chic2K+, Chic2->JPsiGamma, JPsi->MuMu",1078) ;
  cat.defineType("B0->JPsiPi0, JPsi->EE",1081) ;
  cat.defineType("B0->JPsiPi0, JPsi->MuMu",1082) ;
  cat.defineType("B+->JPsiPi+, JPsi->EE",1083) ;
  cat.defineType("B+->JPsiPi+, JPsi->MuMu",1084) ;
  cat.defineType("B0->JPsiRho0, JPsi->EE",1091) ;
  cat.defineType("B0->JPsiRho0, JPsi->MuMu",1092) ;
  cat.defineType("B+->JPsiRhop, JPsi->EE",1093) ;
  cat.defineType("B+->JPsiRhop, JPsi->MuMu",1094) ;
  cat.defineType("B0->Pi+Pi-",2000) ;
  cat.defineType("B0->K+K-",2001) ;
  cat.defineType("B0->K+Pi-",2002) ;
  cat.defineType("B0->D-Pi+, D- ->KsPi",3120) ;
  cat.defineType("B0->D-Pi+, D- ->KPiPi",3119) ;
  cat.defineType("B0->D-Rhop, D- ->KsPi",3122) ;
  cat.defineType("B0->D-Rhop, D- ->KPiPi",3121) ;
  cat.defineType("B0->D-A1p, D- ->KsPi",3124) ;
  cat.defineType("B0->D-A1p, D- ->KPiPi",3123) ;
  cat.defineType("B0->D*-Pi+, D*- ->D0bPi-, D0b->KPi",3101) ;
  cat.defineType("B0->D*-Pi+, D*- ->D0bPi-, D0b->KPiPi0",3102) ;
  cat.defineType("B0->D*-Pi+, D*- ->D0bPi-, D0b->K3Pi",3103) ;
  cat.defineType("B0->D*-Pi+, D*- ->D0bPi-, D0b->KsPi+Pi-",3104) ;
  cat.defineType("B0->D*-Pi+, D*- ->D-Pi0, D- ->KPiPi",3105) ;
  cat.defineType("B0->D*-Pi+, D*- ->D-Pi0, D- ->KsPi",3106) ;
  cat.defineType("B0->D*-Rhop, D*- ->D0bPi-, D0b->KPi",3107) ;
  cat.defineType("B0->D*-Rhop, D*- ->D0bPi-, D0b->KPiPi0",3108) ;
  cat.defineType("B0->D*-Rhop, D*- ->D0bPi-, D0b->K3Pi",3109) ;
  cat.defineType("B0->D*-Rhop, D*- ->D0bPi-, D0b->KsPi+Pi-",3110) ;
  cat.defineType("B0->D*-Rhop, D*- ->D-Pi0, D- ->KPiPi",3111) ;
  cat.defineType("B0->D*-Rhop, D*- ->D-Pi0, D- ->KsPi",3112) ;
  cat.defineType("B0->D*-A1p, D*- ->D0bPi-, D0b->KPi",3113) ;
  cat.defineType("B0->D*-A1p, D*- ->D0bPi-, D0b->KPiPi0",3114) ;
  cat.defineType("B0->D*-A1p, D*- ->D0bPi-, D0b->K3Pi",3115) ;
  cat.defineType("B0->D*-A1p, D*- ->D0bPi-, D0b->KsPi+Pi-",3116) ;
  cat.defineType("B0->D*-A1p, D*- ->D-Pi0, D- ->KPiPi",3117) ;
  cat.defineType("B0->D*-A1p, D*- ->D-Pi0, D- ->KsPi",3118) ;
  cat.defineType("B+->D0bPi+, D0b->KPi",3125) ;
  cat.defineType("B+->D0bPi+, D0b->KPiPi0",3126) ;
  cat.defineType("B+->D0bPi+, D0b->K3Pi",3127) ;
  cat.defineType("B+->D0bPi+, D0b->KPi+Pi-",3128) ;
  cat.defineType("B+->D0bRhop, D0b->KPi",3129) ;
  cat.defineType("B+->D0bRhop, D0b->KPiPi0",3130) ;
  cat.defineType("B+->D0bRhop, D0b->K3Pi",3131) ;
  cat.defineType("B+->D0bRhop, D0b->KPi+Pi-",3132) ;
  cat.defineType("B+->D0bA1p, D0b->KPi",3133) ;
  cat.defineType("B+->D0bA1p, D0b->KPiPi0",3134) ;
  cat.defineType("B+->D0bA1p, D0b->K3Pi",3135) ;
  cat.defineType("B+->D0bA1p, D0b->KPi+Pi-",3136) ;
  cat.defineType("B+->D*0bPi+, D*0b->D0bPi0, D0b->KPi",3137) ;
  cat.defineType("B+->D*0bPi+, D*0b->D0bPi0, D0b->KPiPi0",3138) ;
  cat.defineType("B+->D*0bPi+, D*0b->D0bPi0, D0b->K3Pi",3139) ;
  cat.defineType("B+->D*0bPi+, D*0b->D0bPi0, D0b->KsPi+Pi-",3140) ;
  cat.defineType("B+->D*0bRhop, D*0b->D0bPi0, D0b->KPi",3141) ;
  cat.defineType("B+->D*0bRhop, D*0b->D0bPi0, D0b->KPiPi0",3142) ;
  cat.defineType("B+->D*0bRhop, D*0b->D0bPi0, D0b->K3Pi",3143) ;
  cat.defineType("B+->D*0bRhop, D*0b->D0bPi0, D0b->KsPi+Pi-",3144) ;
  cat.defineType("B+->D*0bA1p, D*0b->D0bPi0, D0b->KPi",3145) ;
  cat.defineType("B+->D*0bA1p, D*0b->D0bPi0, D0b->KPiPi0",3146) ;
  cat.defineType("B+->D*0bA1p, D*0b->D0bPi0, D0b->K3Pi",3147) ;            
  cat.defineType("B+->D*0bA1p, D*0b->D0bPi0, D0b->KsPi+Pi-",3148) ;
}


