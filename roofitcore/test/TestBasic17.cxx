#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooNLLVar.h"
#include "RooMinuit.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic17 : public RooFitTestUnit
{
public: 
  TestBasic17(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Interactive fitting"

    // Setup a model
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mx("mx","mx",0,-0.5,0.5) ;
    RooRealVar sx("sx","sx",3,2.5,3.5) ;
    RooGaussian gx("gx","gx",x,mx,sx) ;
    
    RooRealVar y("y","y",-10,10) ;
    RooRealVar my("my","my",0,-0.5,0.5) ;
    RooRealVar sy("sy","sy",3,1,10) ;
    RooGaussian gy("gy","gy",y,my,sy) ;
    
    RooProdPdf f("f","f",RooArgSet(gx,gy)) ;
    
    // Generate a toy dataset
    RooDataSet* d = f.generate(RooArgSet(x,y),1000) ;
    
    // Construct likelihood
    RooNLLVar nll("nll","nll",f,*d) ;
    
    // Start Minuit session on nll
    RooMinuit m(nll) ;
    
    // Activate constant-term optimization (always recommended)
    m.optimizeConst(kTRUE) ;
    
    // Run HESSE (mx,my,sx,sy free)
    m.hesse() ;
    RooFitResult* r1 = m.save() ;
    
    // Freeze parameters sx,sy
    sx.setConstant(kTRUE) ;
    sy.setConstant(kTRUE) ;
    // (RooMinuit will fix sx,sy in minuit at the next commmand)
    
    // Run MIGRAD (mx,my free)
    m.migrad() ;


    RooFitResult* r2=m.save() ;
    
    // Release sx
    sx.setConstant(kFALSE) ;
    
    // Run MINOS (mx,my,sx free)
    m.migrad() ;
    m.minos() ;
    
    // Save a snapshot of the fit result
    RooFitResult* r3 = m.save() ;
    
    // Make contour plot of mx vs sx
    // m.contour(mx,my) ;
    
    regResult(r1,"Basic17_Result1") ;
    regResult(r2,"Basic17_Result2") ;
    regResult(r3,"Basic17_Result3") ;

    delete d ;
    
    return kTRUE ;
  }
} ;
