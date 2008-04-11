#include "RooRealVar.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic9 : public RooFitTestUnit
{
public: 
  TestBasic9(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Formula expressions"

    // Build Gaussian PDF
    RooRealVar x("x","x",-10,10) ;
    RooRealVar y("y","y",0,3) ;
    
    //  g(x,m,s)
    //  m -> m(y) = m0 + m1*y
    //  g(x,m(y),s)
    
    // Build a parameterized mean variable for gauss
    RooRealVar mean0("mean0","offset of mean function",0.5) ;
    RooRealVar mean1("mean1","slope of mean function",3.0) ;
    RooFormulaVar mean("mean","parameterized mean","mean0+mean1*y",RooArgList(mean0,mean1,y)) ;
    
    RooRealVar sigma("sigma","width of gaussian",3) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;
    
    // Generate a toy MC set
    RooDataSet* data = gauss.generate(RooArgList(x,y),10000) ;
        
    // Plot x projection
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    gauss.plotOn(xframe) ; // plots f(x) = Int(dy) pdf(x,y)
    
    // Plot y projection
    RooPlot* yframe = y.frame() ;
    data->plotOn(yframe) ;
    gauss.plotOn(yframe) ; // plots f(y) = Int(dx) pdf(x,y)

    regPlot(xframe,"Basic9_PlotX") ;
    regPlot(yframe,"Basic9_PlotY") ;

    delete data ;

    return kTRUE ;
  }
} ;
