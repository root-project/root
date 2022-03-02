//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #703
//
// Using a product of an (acceptance) efficiency and a p.d.f as p.d.f.
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
#include "RooExponential.h"
#include "RooEffProd.h"
#include "RooFormulaVar.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic703 : public RooFitTestUnit
{
public:
  TestBasic703(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Efficiency product operator p.d.f",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // D e f i n e   o b s e r v a b l e s   a n d   d e c a y   p d f
  // ---------------------------------------------------------------

  // Declare observables
  RooRealVar t("t","t",0,5) ;

  // Make pdf
  RooRealVar tau("tau","tau",-1.54,-4,-0.1) ;
  RooExponential model("model","model",t,tau) ;



  // D e f i n e   e f f i c i e n c y   f u n c t i o n
  // ---------------------------------------------------

  // Use error function to simulate turn-on slope
  RooFormulaVar eff("eff","0.5*(TMath::Erf((t-1)/0.5)+1)",t) ;



  // D e f i n e   d e c a y   p d f   w i t h   e f f i c i e n c y
  // ---------------------------------------------------------------

  // Multiply pdf(t) with efficiency in t
  RooEffProd modelEff("modelEff","model with efficiency",model,eff) ;



  // P l o t   e f f i c i e n c y ,   p d f
  // ----------------------------------------

  RooPlot* frame1 = t.frame(Title("Efficiency")) ;
  eff.plotOn(frame1,LineColor(kRed)) ;

  RooPlot* frame2 = t.frame(Title("Pdf with and without efficiency")) ;

  model.plotOn(frame2,LineStyle(kDashed)) ;
  modelEff.plotOn(frame2) ;



  // G e n e r a t e   t o y   d a t a ,   f i t   m o d e l E f f   t o   d a t a
  // ------------------------------------------------------------------------------

  // Generate events. If the input pdf has an internal generator, the internal generator
  // is used and an accept/reject sampling on the efficiency is applied.
  RooDataSet* data = modelEff.generate(t,10000) ;

  // Fit pdf. The normalization integral is calculated numerically.
  modelEff.fitTo(*data) ;

  // Plot generated data and overlay fitted pdf
  RooPlot* frame3 = t.frame(Title("Fitted pdf with efficiency")) ;
  data->plotOn(frame3) ;
  modelEff.plotOn(frame3) ;


  regPlot(frame1,"rf703_plot1") ;
  regPlot(frame2,"rf703_plot2") ;
  regPlot(frame3,"rf703_plot3") ;


  delete data ;
  return kTRUE ;
  }
} ;
