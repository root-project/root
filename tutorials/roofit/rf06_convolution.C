/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #6
// 
// One-dimensional numeric convolution
// (require ROOT to be compiled with --enable-fftw3)
// 
// pdf = landau(t) (x) gauss(t)
// 
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooLandau.h"
#include "RooFFTConvPdf.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;



void rf06_convolution()
{
  // Construct observable
  RooRealVar t("t","t",-10,30) ;

  // Set #bins to be used for FFT sampling to 10000
  t.setBins(10000,"cache") ; 

  // Construct landau(t,ml,sl) ;
  RooRealVar ml("ml","mean landau",5.,-20,20) ;
  RooRealVar sl("sl","sigma landau",1,0.1,10) ;
  RooLandau landau("lx","lx",t,ml,sl) ;

  // Construct gauss(t,mg,sg)
  RooRealVar mg("mg","mg",0) ;
  RooRealVar sg("sg","sg",2,0.1,10) ;
  RooGaussian gauss("gauss","gauss",t,mg,sg) ;

  // Construct landau (x) gauss
  RooFFTConvPdf lxg("lxg","landau (X) gauss",t,landau,gauss) ;

  // Sample 1000 events in x from gxlx
  RooDataSet* data = lxg.generate(t,10000) ;

  // Fit gxlx to data
  lxg.fitTo(*data,Minos(kFALSE)) ;

  // Plot data, landau, landau(X)gauss
  RooPlot* frame = t.frame() ;
  data->plotOn(frame) ;
  lxg.plotOn(frame) ;
  landau.plotOn(frame,LineStyle(kDashed)) ;

  // Draw frame on canvas
  new TCanvas("c","c",600,600) ;
  frame->Draw() ;

}



