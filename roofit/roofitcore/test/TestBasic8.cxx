#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooGaussian.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic8 : public RooFitTestUnit
{
public: 
  TestBasic8(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Multiple observable configurations of p.d.f.s"

    // A simple gaussian PDF has 3 variables: x,mean,sigma
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mean("mean","mean of gaussian",-1,-10,10) ;
    RooRealVar sigma("sigma","width of gaussian",3,1,20) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;
    
    // For getVal() without any arguments all variables are interpreted as parameters,
    // and no normalization is enforced
    x = 0 ;
    Double_t rawVal = gauss.getVal() ; // = exp(-[(x-mean)/sigma]^2)
    regValue(rawVal,"Basic8_Raw") ;
    // cout << "gauss(x=0,mean=-1,width=3)_raw = " << rawVal << endl ;
    
    // If we supply getVal() with the subset of its variables that should be interpreted as dependents,
    // it will apply the correct normalization for that set of dependents
    RooArgSet nset1(x) ;
    Double_t xnormVal = gauss.getVal(&nset1) ; 
    regValue(xnormVal,"Basic8_NormX") ;
    // cout << "gauss(x=0,mean=-1,width=3)_normalized_x[-10,10] = " << xnormVal << endl ;
    
    //*** gauss.getVal(x) = gauss.getVal() / Int(-10,10) gauss() dx
    
    // If we adjust the limits on x, the normalization will change accordingly
    x.setRange(-1,1) ;
    Double_t xnorm2Val = gauss.getVal(&nset1) ;
    regValue(xnorm2Val,"Basic8_NormXRange") ;
    // cout << "gauss(x=0,mean=-1,width=3)_normalized_x[-1,1] = " << xnorm2Val << endl ;
    
    //*** gauss.getVal(x) = gauss.getVal() / Int(-1,1) gauss() dx
    
    // We can also add sigma as dependent
    RooArgSet nset2(x,sigma) ;
    Double_t xsnormVal = gauss.getVal(&nset2) ;
    regValue(xsnormVal,"Basic8_NormXS") ;
    // cout << "gauss(x=0,mean=-1,width=3)_normalized_x[-1,1]_width[1,20] = " << xsnormVal << endl ;
    
    //*** gauss.getVal(RooArgSet(x,sigma)) = gauss.getVal() / Int(-1,1)(1,20) gauss() dx dsigma
    
    
    return kTRUE ;
  }
} ;
