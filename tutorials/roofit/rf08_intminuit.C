/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #8
// 
// Interactive minimization with MINUIT
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooProdPdf.h"
#include "RooMinuit.h"
#include "RooNLLVar.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


void rf08_intminuit()
{
  // Setup a dummy model gaussx(x,mx,sx) * gaussy(y,my,sy)
  RooRealVar x("x","x",-10,10) ;
  RooRealVar mx("mx","mx",0,-0.5,0.5) ;
  RooRealVar sx("sx","sx",3,2.5,3.5) ;
  RooGaussian gx("gx","gx",x,mx,sx) ;

  RooRealVar y("y","y",-10,10) ;
  RooRealVar my("my","my",0,-0.5,0.5) ;
  RooRealVar sy("sy","sy",3,1,10) ;
  RooGaussian gy("gy","gy",y,my,sy) ;

  RooProdPdf model("model","model",RooArgSet(gx,gy)) ;

  // Sample 1000 events in x,y from 
  RooDataSet* data = model.generate(RooArgSet(x,y),1000) ;

  // Construct a function representing -log(L) of model w.r.t. data
  RooNLLVar nll("nll","nll",model,*data) ;

  // Start Minuit session on anove nll
  RooMinuit m(nll) ;

  // Activate constant-term optimization (always recommended)
  m.optimizeConst(kTRUE) ;

  // Activate verbose logging
  m.setVerbose(kTRUE) ;

  // Run HESSE (mx,my,sx,sy free)
  m.hesse() ;

  // Deactivate verbose logging
  m.setVerbose(kFALSE) ;

  // Freeze parameters sx,sy
  sx.setConstant(kTRUE) ;
  sy.setConstant(kTRUE) ;
  // (RooMinuit will fix sx,sy in minuit at the next commmand)

  // Run MIGRAD (mx,my free) 
  m.migrad() ;
  
  // Release sx
  sx.setConstant(kFALSE) ;
 
  // Run MINOS (mx,my,sx free)
  m.minos() ;
  
  // Save a snapshot of the fit result
  RooFitResult* r = m.save() ;

  // Make contour plot of mx vs sx
  m.contour(mx,my) ;

  // Print the fit result snapshot
  r->Print("v") ;

}
