#include "RooRealVar.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic18 : public RooFitTestUnit
{
public: 
  TestBasic18(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Lagrange multipliers"

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
    
    // Construct formula with likelihood plus penalty
    RooRealVar alpha("alpha","penalty strength",1000) ;
    RooFormulaVar nllPen("nllPen",
			 "nll+alpha*abs(mx-my)",
			 RooArgList(nll,alpha,mx,my)) ;
    
    // Start Minuit session on straight NLL
    RooMinuit m(nll) ;
    m.migrad() ;
    m.hesse() ;
    RooFitResult* r1 = m.save() ;
    
    // Start Minuit session on straight NLL
    RooMinuit m2(nllPen) ;
    m2.migrad() ;
    m2.hesse() ;
    RooFitResult* r2 = m2.save() ;
    
    // Print results
    regResult(r1,"Basic18_Result1") ;
    regResult(r2,"Basic18_Result2") ;
    
    delete d ;

    return kTRUE ;
  }
} ;
