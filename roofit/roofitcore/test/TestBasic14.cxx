#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooBMixDecay.h"
#include "RooBinning.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic14 : public RooFitTestUnit
{
public: 
  TestBasic14(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Plotting with variable bin sizes"

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
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;
    
    // Construct a decay PDF, smeared with single gaussian resolution model
    RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,gm1,RooBMixDecay::DoubleSided) ;
    
    // Generate BMixing data with above set of event errors
    RooDataSet *data = bmix.generate(RooArgSet(dt,mixState,tagFlav),2000) ;
    
    // *** Plot mixState asymmetry with variable bin sizes ***
    
    // Create binning object with range (-10,10)
    RooBinning abins(-10,10) ;
    
    // Define bin boundaries
    abins.addBoundary(0) ;
    abins.addBoundaryPair(1) ;
    abins.addBoundaryPair(2) ;
    abins.addBoundaryPair(3) ;
    abins.addBoundaryPair(4) ;
    abins.addBoundaryPair(6) ;
    
    // Create plot frame and plot mixState asymmetry of bmix
    RooPlot* aframe = dt.frame(-10,10) ;
    bmix.plotOn(aframe,Asymmetry(mixState)) ;
    
    // Plot mixState asymmetry of data with specified binning
    data->plotOn(aframe,Asymmetry(mixState),Binning(abins)) ;
    
    aframe->SetMinimum(-1.1) ;
    aframe->SetMaximum(1.1) ;
    
    // *** Plot deltat distribution with variable bin sizes ***
    
    // Create binning object with range (-15,15)
    RooBinning tbins(-15,15) ;
    
    // Add 60 bins with uniform spacing in range (-15,0)
    tbins.addUniform(60,-15,0) ;
    
    // Make plot with specified binning
    RooPlot* dtframe = dt.frame(-15,15) ;
    data->plotOn(dtframe,Binning(tbins)) ;
    bmix.plotOn(dtframe) ;
    
    regPlot(dtframe,"Basic14_PlotDt") ;
    regPlot(aframe,"Basic14_PlotA") ;

    delete data ;

    return kTRUE ;
  }
} ;
