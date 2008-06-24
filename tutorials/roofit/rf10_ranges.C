/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #10
// 
// Fitting and plotting in sub ranges
//
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


void rf10_ranges()
{
  ///////////////////////////////////////////////////////////
  //  Example in one dimension
  ///////////////////////////////////////////////////////////

  // Construct observables x
  RooRealVar x("x","x",-10,10) ;
  
  // Construct gaussx(x,mx,1)
  RooRealVar mx("mx","mx",0,-10,10) ;
  RooGaussian gx("gx","gx",x,mx,RooConst(1)) ;

  // Construct px = 1 (flat in x)
  RooPolynomial px("px","px",x) ;

  RooRealVar fx("fx","fx",0.,1.) ;
  RooAddPdf model("model","model",RooArgList(gx,px),fx) ;

  // Generated 10000 events in (x,y) from p.d.f. model
  RooDataSet* modelData = model.generate(x,1000) ;

  // Fit p.d.f to all data
  RooFitResult* r_full = model.fitTo(*modelData,Minos(kFALSE),Save(kTRUE)) ;

  // Define "signal" range in x as [-3,3]
  x.setRange("signal",-3,3) ;  

  // Fit p.d.f only to data in "signal" range
  RooFitResult* r_sig = model.fitTo(*modelData,Minos(kFALSE),Save(kTRUE),Range("signal")) ;

  // Make plot frame in x and add data and fitted model (only shown in fitted range)
  RooPlot* frame = x.frame() ;
  modelData->plotOn(frame) ;
  model.plotOn(frame) ;

  // Draw frame on canvas
  TCanvas* c = new TCanvas("rf10_ranges","rf10_ranges",600,600) ;
  c->cd(1) ; frame->Draw() ;

  // Print fit results 
  cout << "result of fit on all data " << endl ;
  r_full->Print() ;  
  cout << "result of fit in in signal region (note increased error on signal fraction)" << endl ;
  r_sig->Print() ;

  return ;
  
}
