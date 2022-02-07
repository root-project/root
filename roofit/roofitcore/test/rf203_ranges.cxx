/////////////////////////////////////////////////////////////////////////
//
// 'ADDITION AND CONVOLUTION' RooFit tutorial macro #203
//
// Fitting and plotting in sub ranges
//
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
#include "RooPolynomial.h"
#include "RooAddPdf.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


class TestBasic203 : public RooFitTestUnit
{
public:
  TestBasic203(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Basic fitting and plotting in ranges",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // S e t u p   m o d e l
    // ---------------------

    // Construct observables x
    RooRealVar x("x","x",-10,10) ;

    // Construct gaussx(x,mx,1)
    RooRealVar mx("mx","mx",0,-10,10) ;
    RooGaussian gx("gx","gx",x,mx,RooConst(1)) ;

    // Construct px = 1 (flat in x)
    RooPolynomial px("px","px",x) ;

    // Construct model = f*gx + (1-f)px
    RooRealVar f("f","f",0.,1.) ;
    RooAddPdf model("model","model",RooArgList(gx,px),f) ;

    // Generated 10000 events in (x,y) from p.d.f. model
    RooDataSet* modelData = model.generate(x,10000) ;

    // F i t   f u l l   r a n g e
    // ---------------------------

    // Fit p.d.f to all data
    RooFitResult* r_full = model.fitTo(*modelData,Save(kTRUE)) ;


    // F i t   p a r t i a l   r a n g e
    // ----------------------------------

    // Define "signal" range in x as [-3,3]
    x.setRange("signal",-3,3) ;

    // Fit p.d.f only to data in "signal" range
    RooFitResult* r_sig = model.fitTo(*modelData,Save(kTRUE),Range("signal")) ;


    // P l o t   /   p r i n t   r e s u l t s
    // ---------------------------------------

    // Make plot frame in x and add data and fitted model
    RooPlot* frame = x.frame(Title("Fitting a sub range")) ;
    modelData->plotOn(frame) ;
    model.plotOn(frame,Range("Full"),LineStyle(kDashed),LineColor(kRed)) ; // Add shape in full ranged dashed
    model.plotOn(frame) ; // By default only fitted range is shown

    regPlot(frame,"rf203_plot") ;
    regResult(r_full,"rf203_r_full") ;
    regResult(r_sig,"rf203_r_sig") ;

    delete modelData ;
    return kTRUE;
  }
} ;
