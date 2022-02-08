/////////////////////////////////////////////////////////////////////////
//
// 'BASIC FUNCTIONALITY' RooFit tutorial macro #110
//
// Examples on normalization of p.d.f.s,
// integration of p.d.fs, construction
// of cumulative distribution functions from p.d.f.s
// in one dimension
//
// 07/2008 - Wouter Verkerke
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooAbsReal.h"
#include "RooPlot.h"
#include "TCanvas.h"
using namespace RooFit ;

class TestBasic110 : public RooFitTestUnit
{
public:
  TestBasic110(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Normalization of p.d.f.s in 1D",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // S e t u p   m o d e l
    // ---------------------

    // Create observables x,y
    RooRealVar x("x","x",-10,10) ;

    // Create p.d.f. gaussx(x,-2,3)
    RooGaussian gx("gx","gx",x,RooConst(-2),RooConst(3)) ;


    // R e t r i e v e   r a w  &   n o r m a l i z e d   v a l u e s   o f   R o o F i t   p . d . f . s
    // --------------------------------------------------------------------------------------------------

    // Return 'raw' unnormalized value of gx
    regValue(gx.getVal(),"rf110_gx") ;

    // Return value of gx normalized over x in range [-10,10]
    RooArgSet nset(x) ;

    regValue(gx.getVal(&nset),"rf110_gx_Norm[x]") ;

    // Create object representing integral over gx
    // which is used to calculate  gx_Norm[x] == gx / gx_Int[x]
    RooAbsReal* igx = gx.createIntegral(x) ;
    regValue(igx->getVal(),"rf110_gx_Int[x]") ;


    // I n t e g r a t e   n o r m a l i z e d   p d f   o v e r   s u b r a n g e
    // ----------------------------------------------------------------------------

    // Define a range named "signal" in x from -5,5
    x.setRange("signal",-5,5) ;

    // Create an integral of gx_Norm[x] over x in range "signal"
    // This is the fraction of of p.d.f. gx_Norm[x] which is in the
    // range named "signal"
    RooAbsReal* igx_sig = gx.createIntegral(x,NormSet(x),Range("signal")) ;
    regValue(igx_sig->getVal(),"rf110_gx_Int[x|signal]_Norm[x]") ;



    // C o n s t r u c t   c u m u l a t i v e   d i s t r i b u t i o n   f u n c t i o n   f r o m   p d f
    // -----------------------------------------------------------------------------------------------------

    // Create the cumulative distribution function of gx
    // i.e. calculate Int[-10,x] gx(x') dx'
    RooAbsReal* gx_cdf = gx.createCdf(x) ;

    // Plot cdf of gx versus x
    RooPlot* frame = x.frame(Title("c.d.f of Gaussian p.d.f")) ;
    gx_cdf->plotOn(frame) ;


    regPlot(frame,"rf110_plot1") ;

    delete igx ;
    delete igx_sig ;
    delete gx_cdf ;

    return kTRUE ;
  }
} ;
