#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooSimPdfBuilder.h"
#include "RooSimultaneous.h"
#include "RooBifurGauss.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic16 : public RooFitTestUnit
{
public: 
  TestBasic16(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Complex slice projections"

    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar dm("dm","dm",0.472) ;
    RooRealVar tau("tau","tau",1.547) ;
    RooRealVar w("w","mistag rate",0.1) ;
    RooRealVar dw("dw","delta mistag rate",0.) ;
    RooCategory mixState("mixState","B0/B0bar mixing state") ;
    mixState.defineType("mixed",-1) ;
    mixState.defineType("unmixed",1) ;
    RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
    tagFlav.defineType("B0",1) ;
    tagFlav.defineType("B0bar",-1) ;
    
    // Build a gaussian resolution model
    RooRealVar dterr("dterr","dterr",0.1,1.0) ;
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",0.1) ;  
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1,dterr) ;
    
    gm1.advertiseFlatScaleFactorIntegral(kTRUE) ; // Pretend that b(X)gm1 is flat in dterr
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;
    
    // Define tagCat and runBlock categories
    RooCategory tagCat("tagCat","Tagging category") ;
    tagCat.defineType("Lep") ;
    tagCat.defineType("Kao") ;
    tagCat.defineType("NT1") ;
    tagCat.defineType("NT2") ;
    
    RooCategory runBlock("runBlock","Run block") ;
    runBlock.defineType("Run1") ;
    runBlock.defineType("Run2") ;
    
    // Instantiate a builder
    RooSimPdfBuilder mgr(bmix) ;
    RooArgSet& config = *mgr.createProtoBuildConfig() ;
    
    // Enter configuration data
    config.setStringValue("physModels", "bmix") ;               
    config.setStringValue("splitCats",  "tagCat runBlock") ;    
    config.setStringValue("bmix",       "tagCat          : w " 
			  "runBlock        : tau ") ;
    
    // Build the master PDF
    RooArgSet deps(dt,mixState,dterr,tagCat,runBlock) ;
    RooSimultaneous* simPdf  = (RooSimultaneous*) mgr.buildPdf(config,deps) ;
    RooArgSet& simPars = *simPdf->getParameters(deps) ;
    simPars.setRealValue("tau_Run1",1.5) ;
    simPars.setRealValue("tau_Run2",3.5) ;
    simPars.setRealValue("w_Lep",0.0) ;
    simPars.setRealValue("w_Kao",0.15) ;
    simPars.setRealValue("w_NT1",0.3) ;
    simPars.setRealValue("w_NT2",0.4) ;

    // Generate dummy per-event errors 
    RooBifurGauss gerr("gerr","error distribution",dterr,RooRealConstant::value(0.3),
		       RooRealConstant::value(0.3),RooRealConstant::value(0.8)) ;
    RooDataSet *protoData = gerr.generate(RooArgSet(dterr,tagCat,runBlock),2000) ;      
    
    // Generate other observables
    RooDataSet* data = simPdf->generate(RooArgSet(dt,mixState),*protoData) ;
    
    // Used binned or unbinned projection dataset according to users choice
    RooAbsData* projData = new RooDataHist("projDataWMixH","projDataWMixH",
					   RooArgSet(tagCat,runBlock,dterr,mixState),*data) ;
    
    // Plot the whole dataset
    RooPlot* frame = dt.frame() ;
    data->plotOn(frame) ;
    simPdf->plotOn(frame,ProjWData(dterr,*projData)) ;
    
    // Plot the mixed slice
    RooPlot* frame2 = dt.frame(25) ;
    data->plotOn(frame2,Cut("mixState==mixState::mixed")) ;
    mixState.setLabel("mixed") ;
    simPdf->plotOn(frame2,Slice(mixState),ProjWData(RooArgSet(dterr,mixState),*projData)) ; // Use projData with mixState here
    // Note: if the projected dataset contains value for the sliced observable(s) [as done above], 
    //       only the relevant subset of events is used in the projection 
    //       (i.e events matching the current slice coordinates)
    
    // Plot the {Run1;Lep} slice
    RooPlot* frame3 = dt.frame(25) ;
    data->plotOn(frame3,Cut("runBlock==runBlock::Run1&&tagCat==tagCat::Lep")) ;
    runBlock.setLabel("Run1") ;
    tagCat.setLabel("Lep") ;
    simPdf->plotOn(frame3,Slice(RooArgSet(runBlock,tagCat)),ProjWData(dterr,*projData)) ;
    
    // Plot the {Run1} slice
    RooPlot* frame4 = dt.frame(25) ;
    data->plotOn(frame4,Cut("runBlock==runBlock::Run1")) ;
    runBlock.setLabel("Run1") ;
    simPdf->plotOn(frame4,Slice(runBlock),ProjWData(dterr,*projData)) ;
    
    // Plot the {Run1}-mixed slice
    RooPlot* frame5 = dt.frame(25) ;
    data->plotOn(frame5,Cut("runBlock==runBlock::Run1&&mixState==mixState::mixed")) ;
    runBlock.setLabel("Run1") ;
    mixState.setLabel("mixed") ;
    simPdf->plotOn(frame5,Slice(RooArgSet(runBlock,mixState)),ProjWData(RooArgSet(mixState,dterr),*projData)) ;
    
    // Plot the {Run1;Lep}-mixed slice
    RooPlot* frame6 = dt.frame(25) ;
    data->plotOn(frame6,Cut("runBlock==runBlock::Run1&&tagCat==tagCat::Lep&&mixState==mixState::mixed")) ;
    runBlock.setLabel("Run1") ;
    mixState.setLabel("mixed") ;
    tagCat.setLabel("Lep") ;
    simPdf->plotOn(frame6,Slice(RooArgSet(runBlock,tagCat,mixState)),ProjWData(RooArgSet(dterr,mixState),*projData)) ;

    regPlot(frame, "Basic16_Plot1") ;
    regPlot(frame2,"Basic16_Plot2") ;
    regPlot(frame3,"Basic16_Plot3") ;
    regPlot(frame4,"Basic16_Plot4") ;
    regPlot(frame5,"Basic16_Plot5") ;
    regPlot(frame6,"Basic16_Plot6") ;

    delete &config ;
    delete &simPars ;

    delete protoData ;
    delete data ;
    delete projData ;

    return kTRUE ;
  }
} ;
