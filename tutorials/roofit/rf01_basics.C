//////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #1
// 
// Fitting, plotting, toy data generation on one-dimensional p.d.f
//
// pdf = gauss(x,m,s) 
//
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
using namespace RooFit ;


void rf01_basics()
{
  // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
  RooRealVar x("x","x",-10,10) ;
  RooRealVar mean("mean","mean of gaussian",1,-10,10) ;
  RooRealVar sigma("sigma","width of gaussian",1,0.1,10) ;

  // Build gaussian p.d.f in terms of x,mean and sigma
  RooGaussian gauss("gauss","gaussian PDF",x,mean,sigma) ;  

  // Construct plot frame in 'x'
  RooPlot* xframe = x.frame() ;

  // Plot gauss in frame (i.e. in x) 
  gauss.plotOn(xframe) ;

  // Change the value of sigma to 3
  sigma.setVal(3) ;

  // Plot gauss in frame (i.e. in x) and draw frame on canvas
  gauss.plotOn(xframe,LineColor(kRed)) ;
  
  // Generate a dataset of 1000 events in x from gauss
  RooDataSet* data = gauss.generate(x,10000) ;  
  
  // Make a second plot frame in x and draw both the 
  // data and the p.d.f in the frame
  RooPlot* xframe2 = x.frame() ;
  data->plotOn(xframe2) ;
  gauss.plotOn(xframe2) ;
  
  // Fit pdf to data (no MINOS)
  gauss.fitTo(*data,Minos(kFALSE)) ;

  // Print values of mean and sigma (that now reflect fitted values and errors)
  mean.Print() ;
  sigma.Print() ;

  // Draw all frames on a canvas
  TCanvas* c = new TCanvas("rf01_basics","rf01_basics",800,400) ;
  c->Divide(2) ;
  c->cd(1) ; xframe->Draw() ;
  c->cd(2) ; xframe2->Draw() ;
  
 
}
