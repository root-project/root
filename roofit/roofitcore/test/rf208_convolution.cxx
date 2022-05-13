/////////////////////////////////////////////////////////////////////////
//
// 'ADDITION AND CONVOLUTION' RooFit tutorial macro #208
//
// One-dimensional numeric convolution
// (require ROOT to be compiled with --enable-fftw3)
//
// pdf = landau(t) (x) gauss(t)
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
#include "RooLandau.h"
#include "RooFFTConvPdf.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TPluginManager.h"
#include "TROOT.h"

using namespace RooFit ;



class TestBasic208 : public RooFitTestUnit
{
public:
  TestBasic208(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("FFT Convolution operator p.d.f.",refFile,writeRef,verbose) {} ;

  bool isTestAvailable() {

    TPluginHandler *h;
    if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualFFT"))) {
      if (h->LoadPlugin() == -1) {
   return false;
      } else {
   return true ;
      }
    }
    return false ;
  }

  double ctol() { return 5e-3 ; } // Account for difficult shape of Landau distribution

  bool testCode() {

    // S e t u p   c o m p o n e n t   p d f s
    // ---------------------------------------

    // Construct observable
    RooRealVar t("t","t",-10,30) ;

    // Construct landau(t,ml,sl) ;
    RooRealVar ml("ml","mean landau",5.,-20,20) ;
    RooRealVar sl("sl","sigma landau",1,0.1,10) ;
    RooLandau landau("lx","lx",t,ml,sl) ;

    // Construct gauss(t,mg,sg)
    RooRealVar mg("mg","mg",0) ;
    RooRealVar sg("sg","sg",2,0.1,10) ;
    RooGaussian gauss("gauss","gauss",t,mg,sg) ;


    // C o n s t r u c t   c o n v o l u t i o n   p d f
    // ---------------------------------------

    // Set #bins to be used for FFT sampling to 10000
    t.setBins(10000,"cache") ;

    // Construct landau (x) gauss
    RooFFTConvPdf lxg("lxg","landau (X) gauss",t,landau,gauss) ;


    // S a m p l e ,   f i t   a n d   p l o t   c o n v o l u t e d   p d f
    // ----------------------------------------------------------------------

    // Sample 1000 events in x from gxlx
    RooDataSet* data = lxg.generate(t,10000) ;

    // Fit gxlx to data
    lxg.fitTo(*data) ;

    // Plot data, landau pdf, landau (X) gauss pdf
    RooPlot* frame = t.frame(Title("landau (x) gauss convolution")) ;
    data->plotOn(frame) ;
    lxg.plotOn(frame) ;
    landau.plotOn(frame,LineStyle(kDashed)) ;

    regPlot(frame,"rf208_plot1") ;

    delete data ;
    return true ;

  }
} ;



