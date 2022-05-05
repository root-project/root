/////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #304
//
// Simple uncorrelated multi-dimensional p.d.f.s
//
// pdf = gauss(x,mx,sx) * gauss(y,my,sy)
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
#include "RooProdPdf.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;



class TestBasic304 : public RooFitTestUnit
{
public:
  TestBasic304(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Product operator p.d.f. with uncorrelated terms",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // C r e a t e   c o m p o n e n t   p d f s   i n   x   a n d   y
  // ----------------------------------------------------------------

  // Create two p.d.f.s gaussx(x,meanx,sigmax) gaussy(y,meany,sigmay) and its variables
  RooRealVar x("x","x",-5,5) ;
  RooRealVar y("y","y",-5,5) ;

  RooRealVar meanx("mean1","mean of gaussian x",2) ;
  RooRealVar meany("mean2","mean of gaussian y",-2) ;
  RooRealVar sigmax("sigmax","width of gaussian x",1) ;
  RooRealVar sigmay("sigmay","width of gaussian y",5) ;

  RooGaussian gaussx("gaussx","gaussian PDF",x,meanx,sigmax) ;
  RooGaussian gaussy("gaussy","gaussian PDF",y,meany,sigmay) ;



  // C o n s t r u c t   u n c o r r e l a t e d   p r o d u c t   p d f
  // -------------------------------------------------------------------

  // Multiply gaussx and gaussy into a two-dimensional p.d.f. gaussxy
  RooProdPdf  gaussxy("gaussxy","gaussx*gaussy",RooArgList(gaussx,gaussy)) ;



  // S a m p l e   p d f ,   p l o t   p r o j e c t i o n   o n   x   a n d   y
  // ---------------------------------------------------------------------------

  // Generate 10000 events in x and y from gaussxy
  RooDataSet *data = gaussxy.generate(RooArgSet(x,y),10000) ;

  // Plot x distribution of data and projection of gaussxy on x = Int(dy) gaussxy(x,y)
  RooPlot* xframe = x.frame(Title("X projection of gauss(x)*gauss(y)")) ;
  data->plotOn(xframe) ;
  gaussxy.plotOn(xframe) ;

  // Plot x distribution of data and projection of gaussxy on y = Int(dx) gaussxy(x,y)
  RooPlot* yframe = y.frame(Title("Y projection of gauss(x)*gauss(y)")) ;
  data->plotOn(yframe) ;
  gaussxy.plotOn(yframe) ;

  regPlot(xframe,"rf304_plot1") ;
  regPlot(yframe,"rf304_plot2") ;

  delete data ;

  return true ;
  }
} ;



