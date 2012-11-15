//////////////////////////////////////////////////////////////////////////
//
// 'NUMERIC ALGORITHM TUNING' RooFit tutorial macro #904
// 
//  Caching and interpolation of (numeric) pdf integrals to
//  speed up calculations and to promote numeric stability
//
//  11/2012 - Wouter Verkerke 
// 
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooPlot.h"
#include "RooVoigtian.h"
#include "TFile.h"
#include "TH1.h"

using namespace RooFit ;


void rf904_interpnorm(Bool_t cacheParamInt=kTRUE)
{
  // Report INFO messages on integration to see reporting on topic of this tutorial
  RooMsgService::instance().addStream(INFO,Topic(Caching)) ;

  // Observable
  RooRealVar m("m","m",0,100) ;


  // Parameters
  RooRealVar mass("mass","mass",50,0,100) ;
  RooRealVar sigma("sigma","sigma",1,0.1,10) ;
  RooRealVar width("width","width",1) ;

  // Voigtian is convolution of Gaussian and Breit-Wigner (with numerical normalization integral)
  RooVoigtian v("v","v",m,mass,sigma,width) ;

  // Activate parameterized integral caching for (mass,sigma) parameters
  // Set explicit binning for cache here to control precision (otherwise default binning of 100 bins is used)
  if (cacheParamInt) {
    sigma.setBins(20,"cache") ;
    mass.setBins(20,"cache") ;
    v.setStringAttribute("CACHEPARAMINT","mass:sigma") ;
  }

  // Generate a dataset
  RooDataSet* d = v.generate(m,100) ;

  // Fit model to data
  v.fitTo(*d,Verbose(1)) ;


  TCanvas* c = new TCanvas("c","c",800,400) ;
  c->Divide(2) ;
  
  // Make plot of data and fitted pdf
  RooPlot* frame = m.frame() ;
  d->plotOn(frame) ;
  v.plotOn(frame) ;
  c->cd(1) ;frame->Draw() ;

  // Plot the (cached) normalization integral in (mass,sigma) plane
  const RooAbsReal* nobj = v.getNormIntegral(m) ;
  c->cd(2) ; nobj->createHistogram("mass,sigma")->Draw("surf") ;
}



