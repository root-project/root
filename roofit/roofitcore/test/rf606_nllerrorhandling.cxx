//////////////////////////////////////////////////////////////////////////
//
// 'LIKELIHOOD AND MINIMIZATION' RooFit tutorial macro #606
//
// Understanding and customizing error handling in likelihood evaluations
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
#include "RooArgusBG.h"
#include "RooNLLVar.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


class TestBasic606 : public RooFitTestUnit
{
public:
  TestBasic606(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("NLL error handling",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C r e a t e   m o d e l  a n d   d a t a s e t
  // ----------------------------------------------

  // Observable
  RooRealVar m("m","m",5.20,5.30) ;

  // Parameters
  RooRealVar m0("m0","m0",5.291,5.20,5.30) ;
  RooRealVar k("k","k",-30,-50,-10) ;

  // Pdf
  RooArgusBG argus("argus","argus",m,m0,k) ;

  // Sample 1000 events in m from argus
  RooDataSet* data = argus.generate(m,1000) ;



  // P l o t   m o d e l   a n d   d a t a
  // --------------------------------------

  RooPlot* frame1 = m.frame(Bins(40),Title("Argus model and data")) ;
  data->plotOn(frame1) ;
  argus.plotOn(frame1) ;



  // F i t   m o d e l   t o   d a t a
  // ---------------------------------

  argus.fitTo(*data,PrintEvalErrors(10),Warnings(kFALSE)) ;
  m0.setError(0.1) ;
  argus.fitTo(*data,PrintEvalErrors(0),EvalErrorWall(kFALSE),Warnings(kFALSE)) ;



  // P l o t   l i k e l i h o o d   a s   f u n c t i o n   o f   m 0
  // ------------------------------------------------------------------

  // Construct likelihood function of model and data
  RooNLLVar nll("nll","nll",argus,*data) ;

  // Plot likelihood in m0 in range that includes problematic values
  // In this configuration the number of errors per likelihood point
  // evaluated for the curve is shown. A positive number in PrintEvalErrors(N)
  // will show details for up to N events. By default the values for likelihood
  // evaluations with errors are shown normally (unlike fitting), but the shape
  // of the curve can be erratic in these regions.

  RooPlot* frame2 = m0.frame(Range(5.288,5.293),Title("-log(L) scan vs m0")) ;
  nll.plotOn(frame2,PrintEvalErrors(0),ShiftToZero(),LineColor(kRed),Precision(1e-4)) ;


  // Plot likelihood in m0 in range that includes problematic values
  // In this configuration no messages are printed for likelihood evaluation errors,
  // but if an likelihood value evaluates with error, the corresponding value
  // on the curve will be set to the value given in EvalErrorValue().

  RooPlot* frame3 = m0.frame(Range(5.288,5.293),Title("-log(L) scan vs m0, problematic regions masked")) ;
  nll.plotOn(frame3,PrintEvalErrors(-1),ShiftToZero(),EvalErrorValue(nll.getVal()+10),LineColor(kRed)) ;


  regPlot(frame1,"rf606_plot1") ;
  regPlot(frame2,"rf606_plot2") ;
  regPlot(frame3,"rf606_plot3") ;

  delete data ;
  return kTRUE ;

  }
} ;
