/////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #303
//
// Use of tailored p.d.f as conditional p.d.fs.s
//
// pdf = gauss(x,f(y),sx | y ) with f(y) = a0 + a1*y
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
#include "RooPolyVar.h"
#include "RooProdPdf.h"
#include "RooPlot.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


class TestBasic303 : public RooFitTestUnit
{
public:

RooDataSet* makeFakeDataXY()
{
  RooRealVar x("x","x",-10,10) ;
  RooRealVar y("y","y",-10,10) ;
  RooArgSet coord(x,y) ;

  RooDataSet* d = new RooDataSet("d","d",RooArgSet(x,y)) ;

  for (int i=0 ; i<10000 ; i++) {
    double tmpy = gRandom->Gaus(0,10) ;
    double tmpx = gRandom->Gaus(0.5*tmpy,1) ;
    if (fabs(tmpy)<10 && fabs(tmpx)<10) {
      x = tmpx ;
      y = tmpy ;
      d->add(coord) ;
    }

  }

  return d ;
}



  TestBasic303(TFile* refFile, bool writeRef, Int_t verbose) : RooFitTestUnit("Conditional use of F(x|y)",refFile,writeRef,verbose) {} ;
  bool testCode() {

  // S e t u p   c o m p o s e d   m o d e l   g a u s s ( x , m ( y ) , s )
  // -----------------------------------------------------------------------

  // Create observables
  RooRealVar x("x","x",-10,10) ;
  RooRealVar y("y","y",-10,10) ;

  // Create function f(y) = a0 + a1*y
  RooRealVar a0("a0","a0",-0.5,-5,5) ;
  RooRealVar a1("a1","a1",-0.5,-1,1) ;
  RooPolyVar fy("fy","fy",y,RooArgSet(a0,a1)) ;

  // Creat gauss(x,f(y),s)
  RooRealVar sigma("sigma","width of gaussian",0.5,0.1,2.0) ;
  RooGaussian model("model","Gaussian with shifting mean",x,fy,sigma) ;


  // Obtain fake external experimental dataset with values for x and y
  RooDataSet* expDataXY = makeFakeDataXY() ;



  // G e n e r a t e   d a t a   f r o m   c o n d i t i o n a l   p . d . f   m o d e l ( x | y )
  // ---------------------------------------------------------------------------------------------

  // Make subset of experimental data with only y values
  RooDataSet* expDataY= (RooDataSet*) expDataXY->reduce(y) ;

  // Generate 10000 events in x obtained from _conditional_ model(x|y) with y values taken from experimental data
  RooDataSet *data = model.generate(x,ProtoData(*expDataY)) ;



  // F i t   c o n d i t i o n a l   p . d . f   m o d e l ( x | y )   t o   d a t a
  // ---------------------------------------------------------------------------------------------

  model.fitTo(*expDataXY,ConditionalObservables(y)) ;



  // P r o j e c t   c o n d i t i o n a l   p . d . f   o n   x   a n d   y   d i m e n s i o n s
  // ---------------------------------------------------------------------------------------------

  // Plot x distribution of data and projection of model on x = 1/Ndata sum(data(y_i)) model(x;y_i)
  RooPlot* xframe = x.frame() ;
  expDataXY->plotOn(xframe) ;
  model.plotOn(xframe,ProjWData(*expDataY)) ;


  // Speed up (and approximate) projection by using binned clone of data for projection
  RooAbsData* binnedDataY = expDataY->binnedClone() ;
  model.plotOn(xframe,ProjWData(*binnedDataY),LineColor(kCyan),LineStyle(kDotted),Name("Alt1")) ;


  // Show effect of projection with too coarse binning
  ((RooRealVar*)expDataY->get()->find("y"))->setBins(5) ;
  RooAbsData* binnedDataY2 = expDataY->binnedClone() ;
  model.plotOn(xframe,ProjWData(*binnedDataY2),LineColor(kRed),Name("Alt2")) ;


  regPlot(xframe,"rf303_plot1") ;

  delete binnedDataY ;
  delete binnedDataY2 ;
  delete expDataXY ;
  delete expDataY ;
  delete data ;

  return true ;
  }
} ;




