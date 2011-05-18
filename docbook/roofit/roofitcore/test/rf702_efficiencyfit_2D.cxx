//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #702
// 
// Unbinned maximum likelihood fit of an efficiency eff(x) function to 
// a dataset D(x,cut), where cut is a category encoding a selection whose 
// efficiency as function of x should be described by eff(x)
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooCategory.h"
#include "RooEfficiency.h"
#include "RooPolynomial.h"
#include "RooProdPdf.h"
#include "RooFormulaVar.h"
#include "TCanvas.h"
#include "TH1.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic702 : public RooFitTestUnit
{
public: 
  TestBasic702(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Efficiency operator p.d.f. 2D",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  Bool_t flat=kFALSE ;

  // C o n s t r u c t   e f f i c i e n c y   f u n c t i o n   e ( x , y ) 
  // -----------------------------------------------------------------------

  // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
  RooRealVar x("x","x",-10,10) ;
  RooRealVar y("y","y",-10,10) ;

  // Efficiency function eff(x;a,b) 
  RooRealVar ax("ax","ay",0.6,0,1) ;
  RooRealVar bx("bx","by",5) ;
  RooRealVar cx("cx","cy",-1,-10,10) ;

  RooRealVar ay("ay","ay",0.2,0,1) ;
  RooRealVar by("by","by",5) ;
  RooRealVar cy("cy","cy",-1,-10,10) ;

  RooFormulaVar effFunc("effFunc","((1-ax)+ax*cos((x-cx)/bx))*((1-ay)+ay*cos((y-cy)/by))",RooArgList(ax,bx,cx,x,ay,by,cy,y)) ; 

  // Acceptance state cut (1 or 0)
  RooCategory cut("cut","cutr") ;
  cut.defineType("accept",1) ;
  cut.defineType("reject",0) ;



  // C o n s t r u c t   c o n d i t i o n a l    e f f i c i e n c y   p d f   E ( c u t | x , y ) 
  // ---------------------------------------------------------------------------------------------

  // Construct efficiency p.d.f eff(cut|x)
  RooEfficiency effPdf("effPdf","effPdf",effFunc,cut,"accept") ;



  // G e n e r a t e   d a t a   ( x , y , c u t )   f r o m   a   t o y   m o d e l 
  // -------------------------------------------------------------------------------

  // Construct global shape p.d.f shape(x) and product model(x,cut) = eff(cut|x)*shape(x) 
  // (These are _only_ needed to generate some toy MC here to be used later)
  RooPolynomial shapePdfX("shapePdfX","shapePdfX",x,RooConst(flat?0:-0.095)) ;
  RooPolynomial shapePdfY("shapePdfY","shapePdfY",y,RooConst(flat?0:+0.095)) ;
  RooProdPdf shapePdf("shapePdf","shapePdf",RooArgSet(shapePdfX,shapePdfY)) ;
  RooProdPdf model("model","model",shapePdf,Conditional(effPdf,cut)) ;

  // Generate some toy data from model
  RooDataSet* data = model.generate(RooArgSet(x,y,cut),10000) ;



  // F i t   c o n d i t i o n a l   e f f i c i e n c y   p d f   t o   d a t a 
  // --------------------------------------------------------------------------

  // Fit conditional efficiency p.d.f to data
  effPdf.fitTo(*data,ConditionalObservables(RooArgSet(x,y))) ;



  // P l o t   f i t t e d ,   d a t a   e f f i c i e n c y  
  // --------------------------------------------------------

  // Make 2D histograms of all data, selected data and efficiency function
  TH1* hh_data_all = data->createHistogram("hh_data_all",x,Binning(8),YVar(y,Binning(8))) ;
  TH1* hh_data_sel = data->createHistogram("hh_data_sel",x,Binning(8),YVar(y,Binning(8)),Cut("cut==cut::accept")) ;
  TH1* hh_eff      = effFunc.createHistogram("hh_eff",x,Binning(50),YVar(y,Binning(50))) ;

  // Some adjustsment for good visualization
  hh_data_all->SetMinimum(0) ;
  hh_data_sel->SetMinimum(0) ;
  hh_eff->SetMinimum(0) ;
  hh_eff->SetLineColor(kBlue) ;

  regTH(hh_data_all,"rf702_hh_data_all") ;
  regTH(hh_data_sel,"rf702_hh_data_sel") ;
  regTH(hh_eff,"rf702_hh_eff") ;
  
  delete data ;

  return kTRUE;

  }
} ;
