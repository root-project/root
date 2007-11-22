#include "RooRealVar.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic20 : public RooFitTestUnit
{
public: 
  TestBasic20(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Use of conditional observables" 

    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar dm("dm","dm",0.472,0.1,1.0) ;
    RooRealVar tau("tau","tau",1.548,0.1,5.0) ;
    RooRealVar w("w","mistag rate",0.1,0.0,0.5) ;
    RooRealVar dw("dw","delta mistag rate",0.) ;
    RooCategory mixState("mixState","B0/B0bar mixing state") ;
    mixState.defineType("mixed",-1) ;
    mixState.defineType("unmixed",1) ;
    RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
    tagFlav.defineType("B0",1) ;
    tagFlav.defineType("B0bar",-1) ;
    
    // Build a gaussian resolution model
    RooRealVar dterr("dterr","dterr",0.1,5.0) ;
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",1) ;
    
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1,dterr) ;
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;
    
    // Generate a set of event errors
    RooBifurGauss gerr("gerr","error distribution",dterr,RooRealConstant::value(0.3),
		       RooRealConstant::value(0.3),RooRealConstant::value(0.8)) ;
    RooDataSet *errdata = gerr.generate(dterr,500) ;
    
    // Generate BMixing data with above set of event errors
    RooDataSet *data = bmix.generate(RooArgSet(dt,mixState,tagFlav),*errdata,500) ;
    
    RooFitResult* r = bmix.fitTo(*data,ConditionalObservables(dterr),FitOptions("mhr")) ;

    
    RooPlot* dtframe1 = dt.frame() ;
    data->plotOn(dtframe1) ;
    bmix.plotOn(dtframe1,ProjWData(dterr,*data)) ; // automatically projects (sums) over mixState
    
    RooPlot* dtframe2 = dt.frame() ;
    data->plotOn(dtframe2,Cut("mixState==mixState::mixed")) ;
    mixState.setLabel("mixed") ;
    bmix.plotOn(dtframe2,Slice(mixState),ProjWData(RooArgSet(mixState,dterr),*data)) ; // projects slice in mixState with mixState=mixed
    
    RooPlot* dtframe3 = dt.frame() ;
    data->plotOn(dtframe3,Cut("mixState==mixState::unmixed")) ;
    mixState.setLabel("unmixed") ;
    bmix.plotOn(dtframe3,Slice(mixState),ProjWData(RooArgSet(mixState,dterr),*data)) ;// projects slice in mixState with mixState=mixed

    regResult(r,"Basic20_Result1") ;
    regPlot(dtframe1,"Basic20_Plot1") ;
    regPlot(dtframe2,"Basic20_Plot2") ;
    regPlot(dtframe3,"Basic20_Plot3") ;

    delete data ;
    delete errdata ;
    
    return kTRUE ;
  }
} ;
