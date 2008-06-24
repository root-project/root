/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #17
// 
// Examples on normalization of p.d.f.s,
// integration of p.d.fs, construction
// of cumulative distribution functions from p.d.f.s
// in two dimensions
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooProdPdf.h"
#include "RooAbsReal.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TH1.h"
using namespace RooFit ;


void rf17_normandint2d()
{
  // Create observables x,y
  RooRealVar x("x","x",-10,10) ;
  RooRealVar y("y","y",-10,10) ;

  // Create p.d.f. gaussx(x,-2,3), gaussy(y,2,2) 
  RooGaussian gx("gx","gx",x,RooConst(-2),RooConst(3)) ;
  RooGaussian gy("gy","gy",y,RooConst(+2),RooConst(2)) ;

  // Create gxy = gx(x)*gy(y)
  RooProdPdf gxy("gxy","gxy",RooArgSet(gx,gy)) ;

  ///////////////////////////////////////////////////////////
  //    Retrieve raw, normalized values of RooFit p.d.f.s
  ///////////////////////////////////////////////////////////

  // Return 'raw' unnormalized value of gx
  cout << "gxy = " << gxy.getVal() << endl ;
  
  // Return value of gxy normalized over x _and_ y in range [-10,10]
  RooArgSet nset_xy(x,y) ;
  cout << "gx_Norm[x,y] = " << gxy.getVal(&nset_xy) << endl ;

  // Create object representing integral over gx
  // which is used to calculate  gx_Norm[x,y] == gx / gx_Int[x,y]
  RooAbsReal* igxy = gxy.createIntegral(RooArgSet(x,y)) ;
  cout << "gx_Int[x,y] = " << igxy->getVal() << endl ;

  // NB: it is also possible to do the following

  // Return value of gxy normalized over x in range [-10,10] (i.e. treating y as parameter)
  RooArgSet nset_x(x) ;
  cout << "gx_Norm[x] = " << gxy.getVal(&nset_x) << endl ;

  // Return value of gxy normalized over y in range [-10,10] (i.e. treating x as parameter)
  RooArgSet nset_y(y) ;
  cout << "gx_Norm[y] = " << gxy.getVal(&nset_y) << endl ;



  ///////////////////////////////////////////////////////////
  //    Integration over ranges                  
  ///////////////////////////////////////////////////////////

  // Define a range named "signal" in x from -5,5
  x.setRange("signal",-5,5) ;
  y.setRange("signal",-3,3) ;
  
  // Create an integral of gxy_Norm[x,y] over x and y in range "signal"
  // This is the fraction of of p.d.f. gxy_Norm[x,y] which is in the
  // range named "signal"
  RooAbsReal* igxy_sig = gxy.createIntegral(RooArgSet(x,y),NormSet(RooArgSet(x,y)),Range("signal")) ;
  cout << "gx_Int[x,y|signal]_Norm[x,y] = " << igxy_sig->getVal() << endl ;



  ///////////////////////////////////////////////////////////
  //    Cumulative distribution functions
  ///////////////////////////////////////////////////////////


  // Create the cumulative distribution function of gx
  // i.e. calculate Int[-10,x] gx(x') dx'
  RooAbsReal* gxy_cdf = gxy.createCdf(RooArgSet(x,y)) ;
  
  // Plot cdf of gx versus x
  TH1* hh_cdf = gxy_cdf->createHistogram("hh_cdf",x,Binning(40),YVar(y,Binning(40))) ;
  hh_cdf->SetLineColor(kBlue) ;

  new TCanvas("rf17_normandint2d","rf17_normandint2d",600,600) ;
  hh_cdf->Draw("surf") ;

}
