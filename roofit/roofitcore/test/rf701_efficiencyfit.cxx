//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #701
//
// Unbinned maximum likelihood fit of an efficiency eff(x) function to
// a dataset D(x,cut), where cut is a category encoding a selection, of which
// the efficiency as function of x should be described by eff(x)
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
#include "RooFormulaVar.h"
#include "RooProdPdf.h"
#include "RooEfficiency.h"
#include "RooPolynomial.h"
#include "RooCategory.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic701 : public RooFitTestUnit
{
public:
  TestBasic701(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Efficiency operator p.d.f. 1D",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C o n s t r u c t   e f f i c i e n c y   f u n c t i o n   e ( x )
  // -------------------------------------------------------------------

  // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
  RooRealVar x("x","x",-10,10) ;

  // Efficiency function eff(x;a,b)
  RooRealVar a("a","a",0.4,0,1) ;
  RooRealVar b("b","b",5) ;
  RooRealVar c("c","c",-1,-10,10) ;
  RooFormulaVar effFunc("effFunc","(1-a)+a*cos((x-c)/b)",RooArgList(a,b,c,x)) ;



  // C o n s t r u c t   c o n d i t i o n a l    e f f i c i e n c y   p d f   E ( c u t | x )
  // ------------------------------------------------------------------------------------------

  // Acceptance state cut (1 or 0)
  RooCategory cut("cut","cutr") ;
  cut.defineType("accept",1) ;
  cut.defineType("reject",0) ;

  // Construct efficiency p.d.f eff(cut|x)
  RooEfficiency effPdf("effPdf","effPdf",effFunc,cut,"accept") ;



  // G e n e r a t e   d a t a   ( x ,   c u t )   f r o m   a   t o y   m o d e l
  // -----------------------------------------------------------------------------

  // Construct global shape p.d.f shape(x) and product model(x,cut) = eff(cut|x)*shape(x)
  // (These are _only_ needed to generate some toy MC here to be used later)
  RooPolynomial shapePdf("shapePdf","shapePdf",x,RooConst(-0.095)) ;
  RooProdPdf model("model","model",shapePdf,Conditional(effPdf,cut)) ;

  // Generate some toy data from model
  RooDataSet* data = model.generate(RooArgSet(x,cut),10000) ;



  // F i t   c o n d i t i o n a l   e f f i c i e n c y   p d f   t o   d a t a
  // --------------------------------------------------------------------------

  // Fit conditional efficiency p.d.f to data
  effPdf.fitTo(*data,ConditionalObservables(x)) ;



  // P l o t   f i t t e d ,   d a t a   e f f i c i e n c y
  // --------------------------------------------------------

  // Plot distribution of all events and accepted fraction of events on frame
  RooPlot* frame1 = x.frame(Bins(20),Title("Data (all, accepted)")) ;
  data->plotOn(frame1) ;
  data->plotOn(frame1,Cut("cut==cut::accept"),MarkerColor(kRed),LineColor(kRed)) ;

  // Plot accept/reject efficiency on data overlay fitted efficiency curve
  RooPlot* frame2 = x.frame(Bins(20),Title("Fitted efficiency")) ;
  data->plotOn(frame2,Efficiency(cut)) ; // needs ROOT version >= 5.21
  effFunc.plotOn(frame2,LineColor(kRed)) ;


  regPlot(frame1,"rf701_plot1") ;
  regPlot(frame2,"rf701_plot2") ;


  delete data ;

  return true ;
  }
} ;
