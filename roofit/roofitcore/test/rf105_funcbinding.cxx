//////////////////////////////////////////////////////////////////////////
//
// 'BASIC FUNCTIONALITY' RooFit tutorial macro #105
//
//  Demonstration of binding ROOT Math functions as RooFit functions
//  and pdfs
//
// 07/2008 - Wouter Verkerke
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "TMath.h"
#include "TF1.h"
#include "Math/DistFunc.h"
#include "RooCFunction1Binding.h"
#include "RooCFunction3Binding.h"
#include "RooTFnBinding.h"

using namespace RooFit ;

class TestBasic105 : public RooFitTestUnit
{
public:
  TestBasic105(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("C++ function binding operator p.d.f",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // B i n d   T M a t h : : E r f   C   f u n c t i o n
    // ---------------------------------------------------

    // Bind one-dimensional TMath::Erf function as RooAbsReal function
    RooRealVar x("x","x",-3,3) ;
    RooAbsReal* erf = bindFunction("erf",TMath::Erf,x) ;

    // Plot erf on frame
    RooPlot* frame1 = x.frame(Title("TMath::Erf bound as RooFit function")) ;
    erf->plotOn(frame1) ;


    // B i n d   R O O T : : M a t h : : b e t a _ p d f   C   f u n c t i o n
    // -----------------------------------------------------------------------

    // Bind pdf ROOT::Math::Beta with three variables as RooAbsPdf function
    RooRealVar x2("x2","x2",0,1) ;
    RooRealVar a("a","a",5,0,10) ;
    RooRealVar b("b","b",2,0,10) ;
    RooAbsPdf* beta = bindPdf("beta",ROOT::Math::beta_pdf,x2,a,b) ;

    // Generate some events and fit
    RooDataSet* data = beta->generate(x2,10000) ;
    beta->fitTo(*data) ;

    // Plot data and pdf on frame
    RooPlot* frame2 = x2.frame(Title("ROOT::Math::Beta bound as RooFit pdf")) ;
    data->plotOn(frame2) ;
    beta->plotOn(frame2) ;



    // B i n d   R O O T   T F 1   a s   R o o F i t   f u n c t i o n
    // ---------------------------------------------------------------

    // Create a ROOT TF1 function
    TF1 *fa1 = new TF1("fa1","sin(x)/x",0,10);

    // Create an observable
    RooRealVar x3("x3","x3",0.01,20) ;

    // Create binding of TF1 object to above observable
    RooAbsReal* rfa1 = bindFunction(fa1,x3) ;

    // Make plot frame in observable, plot TF1 binding function
    RooPlot* frame3 = x3.frame(Title("TF1 bound as RooFit function")) ;
    rfa1->plotOn(frame3) ;


    regPlot(frame1,"rf105_plot1") ;
    regPlot(frame2,"rf105_plot2") ;
    regPlot(frame3,"rf105_plot3") ;

    delete erf ;
    delete beta ;
    delete fa1 ;
    delete rfa1 ;
    delete data ;

    return kTRUE ;
  }
} ;
