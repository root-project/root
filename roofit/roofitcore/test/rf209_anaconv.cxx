/////////////////////////////////////////////////////////////////////////
//
// 'ADDITION AND CONVOLUTION' RooFit tutorial macro #209
//
// Decay function p.d.fs with optional B physics
// effects (mixing and CP violation) that can be
// analytically convolved with e.g. Gaussian resolution
// functions
//
// pdf1 = decay(t,tau) (x) delta(t)
// pdf2 = decay(t,tau) (x) gauss(t,m,s)
// pdf3 = decay(t,tau) (x) (f*gauss1(t,m1,s1) + (1-f)*gauss2(t,m1,s1))
//
// 07/2008 - Wouter Verkerke
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussModel.h"
#include "RooAddModel.h"
#include "RooTruthModel.h"
#include "RooDecay.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;



class TestBasic209 : public RooFitTestUnit
{
public:
  TestBasic209(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Analytical convolution operator",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // B - p h y s i c s   p d f   w i t h   t r u t h   r e s o l u t i o n
    // ---------------------------------------------------------------------

    // Variables of decay p.d.f.
    RooRealVar dt("dt","dt",-10,10) ;
    RooRealVar tau("tau","tau",1.548) ;

    // Build a truth resolution model (delta function)
    RooTruthModel tm("tm","truth model",dt) ;

    // Construct decay(t) (x) delta(t)
    RooDecay decay_tm("decay_tm","decay",dt,tau,tm,RooDecay::DoubleSided) ;

    // Plot p.d.f. (dashed)
    RooPlot* frame = dt.frame(Title("Bdecay (x) resolution")) ;
    decay_tm.plotOn(frame,LineStyle(kDashed)) ;


    // B - p h y s i c s   p d f   w i t h   G a u s s i a n   r e s o l u t i o n
    // ----------------------------------------------------------------------------

    // Build a gaussian resolution model
    RooRealVar bias1("bias1","bias1",0) ;
    RooRealVar sigma1("sigma1","sigma1",1) ;
    RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;

    // Construct decay(t) (x) gauss1(t)
    RooDecay decay_gm1("decay_gm1","decay",dt,tau,gm1,RooDecay::DoubleSided) ;

    // Plot p.d.f.
    decay_gm1.plotOn(frame) ;


    // B - p h y s i c s   p d f   w i t h   d o u b l e   G a u s s i a n   r e s o l u t i o n
    // ------------------------------------------------------------------------------------------

    // Build another gaussian resolution model
    RooRealVar bias2("bias2","bias2",0) ;
    RooRealVar sigma2("sigma2","sigma2",5) ;
    RooGaussModel gm2("gm2","gauss model 2",dt,bias2,sigma2) ;

    // Build a composite resolution model f*gm1+(1-f)*gm2
    RooRealVar gm1frac("gm1frac","fraction of gm1",0.5) ;
    RooAddModel gmsum("gmsum","sum of gm1 and gm2",RooArgList(gm1,gm2),gm1frac) ;

    // Construct decay(t) (x) (f*gm1 + (1-f)*gm2)
    RooDecay decay_gmsum("decay_gmsum","decay",dt,tau,gmsum,RooDecay::DoubleSided) ;

    // Plot p.d.f. (red)
    decay_gmsum.plotOn(frame,LineColor(kRed)) ;

    regPlot(frame,"rf209_plot1") ;

    return kTRUE ;

  }
} ;
