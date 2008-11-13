//////////////////////////////////////////////////////////////////////////
//
// 'SPECIAL PDFS' RooFit tutorial macro #704
// 
// Using a p.d.f defined by a sum of real-valued amplitude components
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
#include "RooTruthModel.h"
#include "RooFormulaVar.h"
#include "RooRealSumPdf.h"
#include "RooPolyVar.h"
#include "RooProduct.h"
#include "TH1.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


// Elementary operations on a gaussian PDF
class TestBasic704 : public RooFitTestUnit
{
public: 
  TestBasic704(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Amplitude sum operator p.d.f",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // S e t u p   2 D   a m p l i t u d e   f u n c t i o n s
  // -------------------------------------------------------

  // Observables
  RooRealVar t("t","time",-1.,15.);
  RooRealVar cosa("cosa","cos(alpha)",-1.,1.);

  // Use RooTruthModel to obtain compiled implementation of sinh/cosh modulated decay functions
  RooRealVar tau("tau","#tau",1.5);  
  RooRealVar deltaGamma("deltaGamma","deltaGamma", 0.3);
  RooTruthModel tm("tm","tm",t) ;
  RooFormulaVar coshGBasis("coshGBasis","exp(-@0/ @1)*cosh(@0*@2/2)",RooArgList(t,tau,deltaGamma));
  RooFormulaVar sinhGBasis("sinhGBasis","exp(-@0/ @1)*sinh(@0*@2/2)",RooArgList(t,tau,deltaGamma));
  RooAbsReal* coshGConv = tm.convolution(&coshGBasis,&t);
  RooAbsReal* sinhGConv = tm.convolution(&sinhGBasis,&t);
    
  // Construct polynomial amplitudes in cos(a) 
  RooPolyVar poly1("poly1","poly1",cosa,RooArgList(RooConst(0.5),RooConst(0.2),RooConst(0.2)),0);
  RooPolyVar poly2("poly2","poly2",cosa,RooArgList(RooConst(1),RooConst(-0.2),RooConst(3)),0);

  // Construct 2D amplitude as uncorrelated product of amp(t)*amp(cosa)
  RooProduct  ampl1("ampl1","amplitude 1",RooArgSet(poly1,*coshGConv));
  RooProduct  ampl2("ampl2","amplitude 2",RooArgSet(poly2,*sinhGConv));



  // C o n s t r u c t   a m p l i t u d e   s u m   p d f 
  // -----------------------------------------------------

  // Amplitude strengths
  RooRealVar f1("f1","f1",1,0,2) ;
  RooRealVar f2("f2","f2",0.5,0,2) ;
  
  // Construct pdf
  RooRealSumPdf pdf("pdf","pdf",RooArgList(ampl1,ampl2),RooArgList(f1,f2)) ;

  // Generate some toy data from pdf
  RooDataSet* data = pdf.generate(RooArgSet(t,cosa),10000);

  // Fit pdf to toy data with only amplitude strength floating
  pdf.fitTo(*data) ;



  // P l o t   a m p l i t u d e   s u m   p d f 
  // -------------------------------------------

  // Make 2D plots of amplitudes
  TH1* hh_cos = ampl1.createHistogram("hh_cos",t,Binning(50),YVar(cosa,Binning(50))) ;
  TH1* hh_sin = ampl2.createHistogram("hh_sin",t,Binning(50),YVar(cosa,Binning(50))) ;
  hh_cos->SetLineColor(kBlue) ;
  hh_sin->SetLineColor(kBlue) ;

  
  // Make projection on t, plot data, pdf and its components
  // Note component projections may be larger than sum because amplitudes can be negative
  RooPlot* frame1 = t.frame();
  data->plotOn(frame1);
  pdf.plotOn(frame1);
  pdf.plotOn(frame1,Components(ampl1),LineStyle(kDashed));
  pdf.plotOn(frame1,Components(ampl2),LineStyle(kDashed),LineColor(kRed));
  
  // Make projection on cosa, plot data, pdf and its components
  // Note that components projection may be larger than sum because amplitudes can be negative
  RooPlot* frame2 = cosa.frame();
  data->plotOn(frame2);
  pdf.plotOn(frame2);
  pdf.plotOn(frame2,Components(ampl1),LineStyle(kDashed));
  pdf.plotOn(frame2,Components(ampl2),LineStyle(kDashed),LineColor(kRed));
  

  regPlot(frame1,"rf704_plot1") ;
  regPlot(frame2,"rf704_plot2") ;
  regTH(hh_cos,"rf704_hh_cos") ;
  regTH(hh_sin,"rf704_hh_sin") ;

  delete data ;
  delete coshGConv ;
  delete sinhGConv ;

  return kTRUE ;
  }
} ;
