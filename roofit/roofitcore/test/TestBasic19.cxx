#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooChi2Var.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic19 : public RooFitTestUnit
{
public: 
  TestBasic19(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Chi^2 fits"

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
    RooDataSet* d = f.generate(RooArgSet(x,y),10000) ;
    
    // Bin dataset
    x.setBins(15) ;
    y.setBins(15) ;
    RooDataHist* db = new RooDataHist("db","db",RooArgSet(x,y),*d) ;
    
    // Construct binned likelihood
    RooNLLVar nll("nll","nll",f,*db) ;
    
    // Start Minuit session on NLL
    RooMinuit m(nll) ;
    m.migrad() ;
    m.hesse() ;
    RooFitResult* r1 = m.save() ;
    
    // Construct Chi2
    RooChi2Var chi2("chi2","chi2",f,*db) ;
    
    // Start Minuit session on Chi2
    RooMinuit m2(chi2) ;
    m2.migrad() ;
    m2.hesse() ;
    RooFitResult* r2 = m2.save() ;
    
    // Print results
    regResult(r1,"Basic19_ResultBLL") ;
    regResult(r2,"Basic19_ResultChi2") ;

    delete db ;
    delete d ;
    
    return kTRUE ;
  }
} ;
