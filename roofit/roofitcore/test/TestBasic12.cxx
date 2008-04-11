#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooExtendPdf.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic12 : public RooFitTestUnit
{
public: 
  TestBasic12(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Extended likelihood constructions"


    // Build regular Gaussian PDF
    RooRealVar x("x","x",-10,10) ;
    RooRealVar mean("mean","mean of gaussian",-3,-10,10) ;
    RooRealVar sigma("sigma","width of gaussian",1,0.1,5) ;
    RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;
    
    // Make extended PDF based on gauss. n will be the expected number of events
    RooRealVar n("n","number of events",1000,0,2000) ;
    RooExtendPdf egauss("egauss","extended gaussian PDF",gauss,n) ;
    
    // Generate events from extended PDF
    // The default number of events to generate is taken from gauss.expectedEvents()
    // but can be overrided using a second argument
    RooDataSet* data = egauss.generate(x)  ;
    
    // Fit PDF to dataset in extended mode (selected by fit option "e")
    RooFitResult* r1 = egauss.fitTo(*data,"mher") ;

    
    // Plot both on a frame ;
    RooPlot* xframe = x.frame() ;
    data->plotOn(xframe) ;
    egauss.plotOn(xframe,Normalization(1.0,RooAbsReal::RelativeExpected)) ; // select intrinsic normalization

    // Make an extended gaussian PDF where the number of expected events
    // is counted in a limited region of the dependent range
    x.setRange("cut",-4,2) ;
    RooRealVar mean2("mean2","mean of gaussian",-3) ;
    RooRealVar sigma2("sigma2","width of gaussian",1) ;
    RooGaussian gauss2("gauss2","gaussian PDF 2",x,mean2,sigma2) ;
    
    RooRealVar n2("n2","number of events",1000,0,2000) ;
    RooExtendPdf egauss2("egauss2","extended gaussian PDF w limited range",gauss2,n2,"cut") ;

    RooFitResult* r2 = egauss2.fitTo(*data,"mher");

    
    // cout << "fitted number of events in data in range (-4,2) = " << n2.getVal() << endl ;
    
    // Adding two extended PDFs gives an extended sum PDF
    
    mean = 3.0 ;  sigma = 2.0 ;
    
    // Note that we omit coefficients when adding extended PDFS
    RooAddPdf sumgauss("sumgauss","sum of two extended gauss PDFs",RooArgList(egauss,egauss2)) ;
    sumgauss.plotOn(xframe,LineColor(kRed)) ; // select intrinsic normalization
    
    // Note that in the plot sumgauss does not follow the normalization of the data
    // because its expected number was intentionally chosen not to match the number of events in the data
    
    // If no special 'cut normalizations' are needed (as done in egauss2), there is a shorthand
    // way to construct an extended sumpdf:
    
    RooAddPdf sumgauss2("sumgauss2","extended sum of two gaussian PDFs",
			RooArgList(gauss,gauss2),RooArgList(n,n2)) ;
    sumgauss2.plotOn(xframe,LineColor(kGreen)) ; // select intrinsic normalization
    
    // Note that sumgauss2 looks different from sumgauss because for gauss2 the expected number
    // of event parameter n2 now applies to the entire gauss2 area, whereas in egauss2 it was
    // constructed to represent the number of events in the range (-4,-2). If we would use a separate
    // parameter n3, set to 10000, to represent the number of events for gauss2 in sumgauss2, then
    // sumgauss and sumgauss2 would be indentical.

    regResult(r1,"Basic12_Result1") ;
    regResult(r2,"Basic12_Result2") ;
    regPlot(xframe,"Basic12_Plot1") ;

    delete data ;

    return kTRUE ;
  }
} ;
