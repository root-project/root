#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooArgusBG.h"
#include "RooAddPdf.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic2 : public RooFitTestUnit
{
public: 
  TestBasic2(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Addition operator p.d.f"
    
    // Build two Gaussian PDFs
    RooRealVar x("x","x",0,10) ;
    RooRealVar mean1("mean1","mean of gaussian 1",2,-10,10) ;
    RooRealVar mean2("mean2","mean of gaussian 2",3,-10,10) ;
    RooRealVar sigma("sigma","width of gaussians",1,0.1,10) ;
    RooGaussian gauss1("gauss1","gaussian PDF",x,mean1,sigma) ;  
    RooGaussian gauss2("gauss2","gaussian PDF",x,mean2,sigma) ;  
    
    // Build Argus background PDF
    RooRealVar argpar("argpar","argus shape parameter",-1.,-40.,0.) ;
    RooRealVar cutoff("cutoff","argus cutoff",9.0) ;
    RooArgusBG argus("argus","Argus PDF",x,cutoff,argpar) ;
    
    // Add the components
    RooRealVar g1frac("g1frac","fraction of gauss1",0.5,0,0.7) ;
    RooRealVar g2frac("g2frac","fraction of gauss2",0.3) ;
    RooAddPdf  sum("sum","g1+g2+a",RooArgList(gauss1,gauss2,argus),RooArgList(g1frac,g2frac)) ;
    
    // Generate a toyMC sample
    RooDataSet *data = sum.generate(x,1000) ;

    mean1.setConstant(kTRUE) ;
    mean2.setConstant(kTRUE) ;
    RooFitResult* r = sum.fitTo(*data,"mhr") ;
  
    // Plot data and PDF overlaid
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    
    // Plot only argus and gauss1
    sum.plotOn(xframe) ;
    sum.plotOn(xframe,Components(RooArgSet(argus,gauss2)),Name("curve_Argus_plus_Gauss2")) ;
    sum.plotOn(xframe,Components(argus),Name("curve_Argus")) ;

    regResult(r,"Basic2_Result") ;
    regPlot(xframe,"Basic2_Plot") ;    

    delete data ;

    return kTRUE ;
  }
} ;
