/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #3
// 
// Simple uncorrelated multi-dimensional p.d.f.s
//
// pdf = gauss(x,mx,sx) * gauss(y,my,sy) 
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
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;



void rf03_multidim()
{
  // Create two p.d.f.s gaussx(x,meanx,sigmax) gaussy(y,meany,sigmay) and its variables
  RooRealVar x("x","x",-5,5) ;
  RooRealVar y("y","y",-5,5) ;

  RooRealVar meanx("mean1","mean of gaussian x",2) ;
  RooRealVar meany("mean2","mean of gaussian y",-2) ;
  RooRealVar sigmax("sigmax","width of gaussian x",1) ;
  RooRealVar sigmay("sigmay","width of gaussian y",5) ;

  RooGaussian gaussx("gaussx","gaussian PDF",x,meanx,sigmax) ;  
  RooGaussian gaussy("gaussy","gaussian PDF",y,meany,sigmay) ;  

  // Multiply gaussx and gaussy into a two-dimensional p.d.f. gaussxy
  RooProdPdf  gaussxy("gaussxy","gaussx*gaussy",RooArgList(gaussx,gaussy)) ;

  // Generate 10000 events in x and y from gaussxy
  RooDataSet *data = gaussxy.generate(RooArgSet(x,y),10000) ;

  // Plot x distribution of data and projection of gaussxy on x = Int(dy) gaussxy(x,y)
  RooPlot* xframe = x.frame() ;
  data->plotOn(xframe) ;
  gaussxy.plotOn(xframe) ; 

  // Plot x distribution of data and projection of gaussxy on y = Int(dx) gaussxy(x,y)
  RooPlot* yframe = y.frame() ;
  data->plotOn(yframe) ;
  gaussxy.plotOn(yframe) ; 

  // Make canvas and draw RooPlots
  TCanvas *c = new TCanvas("rf03_multidim","rf03_multidim",800, 400);
  c->Divide(2);
  c->cd(1) ; xframe->Draw() ;
  c->cd(2) ; yframe->Draw() ;

}



