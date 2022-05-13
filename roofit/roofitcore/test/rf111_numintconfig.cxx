//////////////////////////////////////////////////////////////////////////
//
// 'BASIC FUNCTIONALITY' RooFit tutorial macro #111
//
// Configuration and customization of how numeric (partial) integrals
// are executed
//
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
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooNumIntConfig.h"
#include "RooLandau.h"
#include "RooArgSet.h"
#include <iomanip>
using namespace RooFit ;

class TestBasic111 : public RooFitTestUnit
{
public:
  TestBasic111(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Numeric integration configuration",refFile,writeRef,verbose) {} ;
  bool testCode() {

    // A d j u s t   g l o b a l   1 D   i n t e g r a t i o n   p r e c i s i o n
    // ----------------------------------------------------------------------------

    // Example: Change global precision for 1D integrals from 1e-7 to 1e-6
    //
    // The relative epsilon (change as fraction of current best integral estimate) and
    // absolute epsilon (absolute change w.r.t last best integral estimate) can be specified
    // separately. For most p.d.f integrals the relative change criterium is the most important,
    // however for certain non-p.d.f functions that integrate out to zero a separate absolute
    // change criterium is necessary to declare convergence of the integral
    //
    // NB: This change is for illustration only. In general the precision should be at least 1e-7
    // for normalization integrals for MINUIT to succeed.
    //
    RooAbsReal::defaultIntegratorConfig()->setEpsAbs(1e-6) ;
    RooAbsReal::defaultIntegratorConfig()->setEpsRel(1e-6) ;


    // N u m e r i c   i n t e g r a t i o n   o f   l a n d a u   p d f
    // ------------------------------------------------------------------

    // Construct p.d.f without support for analytical integrator for demonstration purposes
    RooRealVar x("x","x",-10,10) ;
    RooLandau landau("landau","landau",x,RooConst(0),RooConst(0.1)) ;


    // Calculate integral over landau with default choice of numeric integrator
    RooAbsReal* intLandau = landau.createIntegral(x) ;
    double val = intLandau->getVal() ;
    regValue(val,"rf111_val1") ;


    // S a m e   w i t h   c u s t o m   c o n f i g u r a t i o n
    // -----------------------------------------------------------


    // Construct a custom configuration which uses the adaptive Gauss-Kronrod technique
    // for closed 1D integrals
    RooNumIntConfig customConfig(*RooAbsReal::defaultIntegratorConfig()) ;
    customConfig.method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D") ;


    // Calculate integral over landau with custom integral specification
    RooAbsReal* intLandau2 = landau.createIntegral(x,NumIntConfig(customConfig)) ;
    double val2 = intLandau2->getVal() ;
    regValue(val2,"rf111_val2") ;


    // A d j u s t i n g   d e f a u l t   c o n f i g   f o r   a   s p e c i f i c   p d f
    // -------------------------------------------------------------------------------------


    // Another possibility: associate custom numeric integration configuration as default for object 'landau'
    landau.setIntegratorConfig(customConfig) ;


    // Calculate integral over landau custom numeric integrator specified as object default
    RooAbsReal* intLandau3 = landau.createIntegral(x) ;
    double val3 = intLandau3->getVal() ;
    regValue(val3,"rf111_val3") ;


    delete intLandau ;
    delete intLandau2 ;
    delete intLandau3 ;

    return true ;
  }
} ;
