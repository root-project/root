#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooProdPdf.h"
#include "RooGaussian.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "RooFitResult.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic3 : public RooFitTestUnit
{
public: 
  TestBasic3(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Product operator p.d.f"

    // Build two Gaussian PDFs
    RooRealVar x("x","x",-5,5) ;
    RooRealVar y("y","y",-5,5) ;
    RooRealVar meanx("mean1","mean of gaussian x",2,-2,6) ;
    RooRealVar meany("mean2","mean of gaussian y",-2) ;
    RooRealVar sigmax("sigmax","width of gaussian x",1.0,0.1,10) ;
    RooRealVar sigmay("sigmay","width of gaussian y",5.0,0.1,50) ;
    RooGaussian gaussx("gaussx","gaussian PDF",x,meanx,sigmax) ;
    RooGaussian gaussy("gaussy","gaussian PDF",y,meany,sigmay) ;
    
    // Multiply the components
    RooProdPdf  prod("gaussxy","gaussx*gaussy",RooArgList(gaussx,gaussy)) ;
    
    // Generate a toyMC sample
    RooDataSet *data = prod.generate(RooArgSet(x,y),1000) ;

    RooFitResult* r = prod.fitTo(*data,"mhr") ;

    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    prod.plotOn(xframe) ; // plots f(x) = Int(dy) pdf(x,y)
    
    RooPlot* yframe = y.frame() ;
    data->plotOn(yframe) ;
    prod.plotOn(yframe) ; // plots f(y) = Int(dx) pdf(x,y)

    regResult(r,"Basic3_Result") ;
    regPlot(xframe,"Basic3_PlotX") ;
    regPlot(yframe,"Basic3_PlotY") ;
    
    delete data ;

    return kTRUE ;
  }
} ;
