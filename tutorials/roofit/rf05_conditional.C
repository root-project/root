/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #5
// 
// Multi-dimensional p.d.f.s with conditional p.d.fs in product
// 
// pdf = gauss(x,f(y),sx | y ) * gauss(y,ms,sx)    with f(y) = a0 + a1*y
// 
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooPolyVar.h"
#include "RooProdPdf.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;



void rf05_conditional()
{
  // Create observables
  RooRealVar x("x","x",-5,5) ;
  RooRealVar y("y","y",-5,5) ;

  // Create function f(y) = a0 + a1*y
  RooRealVar a0("a0","a0",-0.5,-5,5) ;
  RooRealVar a1("a1","a1",-0.5,-1,1) ;
  RooPolyVar fy("fy","fy",y,RooArgSet(a0,a1)) ;

  // Create gaussx(x,f(y),sx)
  RooRealVar sigmax("sigma","width of gaussian",0.5) ;
  RooGaussian gaussx("gaussx","Gaussian in x with shifting mean in y",x,fy,sigmax) ;  

  // Create gaussy(y,0,5)
  RooGaussian gaussy("gaussy","Gaussian in y",y,RooConst(0),RooConst(3)) ;

  // Create gaussx(x,sx|y) * gaussy(y)
  RooProdPdf model("model","gaussx(x|y)*gaussy(y)",gaussy,Conditional(gaussx,x)) ;

  // Generate 1000 events in x and y from model
  RooDataSet *data = model.generate(RooArgSet(x,y),10000) ;

  // Plot x distribution of data and projection of model on x = Int(dy) model(x,y)
  RooPlot* xframe = x.frame() ;
  data->plotOn(xframe) ;
  model.plotOn(xframe) ; 

  // Plot x distribution of data and projection of model on y = Int(dx) model(x,y)
  RooPlot* yframe = y.frame() ;
  data->plotOn(yframe) ;
  model.plotOn(yframe) ; 

  // Make two-dimensional plot in x vs y
  TH1* hh_model = model.createHistogram("hh_model",x,Binning(50),YVar(y,Binning(50))) ;
  hh_model->SetLineColor(kBlue) ;

  // Make canvas and draw RooPlots
  TCanvas *c = new TCanvas("rf05_conditional","rf05_conditional",1200, 400);
  c->Divide(3);
  c->cd(1) ; xframe->Draw() ;
  c->cd(2) ; yframe->Draw() ;
  c->cd(3) ; hh_model->Draw("surf") ;


}



