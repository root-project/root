#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooGaussModel.h"
#include "RooTruthModel.h"
#include "RooDecay.h"
#include "RooAddModel.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic5 : public RooFitTestUnit
{
public: 
  TestBasic5(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Analytically convolved p.d.f"
    
    // Build a simple decay PDF
    RooRealVar dt("dt","dt",-20,20) ;
    RooRealVar tau("tau","tau",1.548) ;
    
    // Build a truth resolution model (delta function)
    RooTruthModel tm("tm","truth model",dt) ;
    
    // Construct a simple unsmeared decay PDF
    RooDecay decay_tm("decay_tm","decay",dt,tau,tm,RooDecay::DoubleSided) ;

    RooPlot* frame1 = dt.frame() ;
    decay_tm.plotOn(frame1) ;
    
    // Build a gaussian resolution model
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",1) ;
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooDecay decay_gm1("decay_gm1","decay",dt,tau,gm1,RooDecay::DoubleSided) ;
    
    RooPlot* frame2 = dt.frame() ;
    decay_gm1.plotOn(frame2) ;
    
    // Build another gaussian resolution model
    RooRealVar bias2("bias2","bias2",0) ;
    RooRealVar sigma2("sigma2","sigma2",5) ;
    RooGaussModel gm2("gm2","gauss model 2",dt,bias2,sigma2) ;
    
    // Build a composite resolution model
    RooRealVar gm1frac("gm1frac","fraction of gm1",0.5) ;
    RooAddModel gmsum("gmsum","sum of gm1 and gm2",RooArgList(gm1,gm2),gm1frac) ;
    
    // Construct a decay PDF, smeared with double gaussian resolution model
    RooDecay decay_gmsum("decay_gmsum","decay",dt,tau,gmsum,RooDecay::DoubleSided) ;
    
    RooPlot* frame3 = dt.frame() ;
    decay_gmsum.plotOn(frame3) ;
    
    regPlot(frame1,"Basic5_Plot1") ;
    regPlot(frame2,"Basic5_Plot2") ;
    regPlot(frame3,"Basic5_Plot3") ;
    

    return kTRUE ;
  }
} ;
