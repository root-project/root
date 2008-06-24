/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #16
// 
// Examples on normalization of p.d.f.s,
// integration of p.d.fs, construction
// of cumulative distribution functions from p.d.f.s
// in one dimension
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooAbsReal.h"
#include "RooPlot.h"
#include "TCanvas.h"
using namespace RooFit ;


void rf16_normandint()
{
  // Create observables x,y
  RooRealVar x("x","x",-10,10) ;

  // Create p.d.f. gaussx(x,-2,3) 
  RooGaussian gx("gx","gx",x,RooConst(-2),RooConst(3)) ;


  ///////////////////////////////////////////////////////////
  //    Retrieve raw, normalized values of RooFit p.d.f.s
  ///////////////////////////////////////////////////////////

  // Return 'raw' unnormalized value of gx
  cout << "gx = " << gx.getVal() << endl ;
  
  // Return value of gx normalized over x in range [-10,10]
  RooArgSet nset(x) ;
  cout << "gx_Norm[x] = " << gx.getVal(&nset) << endl ;

  // Create object representing integral over gx
  // which is used to calculate  gx_Norm[x] == gx / gx_Int[x]
  RooAbsReal* igx = gx.createIntegral(x) ;
  cout << "gx_Int[x] = " << igx->getVal() << endl ;


  ///////////////////////////////////////////////////////////
  //    Integration over ranges                  
  ///////////////////////////////////////////////////////////

  // Define a range named "signal" in x from -5,5
  x.setRange("signal",-5,5) ;
  
  // Create an integral of gx_Norm[x] over x in range "signal"
  // This is the fraction of of p.d.f. gx_Norm[x] which is in the
  // range named "signal"
  RooAbsReal* igx_sig = gx.createIntegral(x,NormSet(x),Range("signal")) ;
  cout << "gx_Int[x|signal]_Norm[x] = " << igx_sig->getVal() << endl ;



  ///////////////////////////////////////////////////////////
  //    Cumulative distribution functions
  ///////////////////////////////////////////////////////////


  // Create the cumulative distribution function of gx
  // i.e. calculate Int[-10,x] gx(x') dx'
  RooAbsReal* gx_cdf = gx.createCdf(x) ;
  
  // Plot cdf of gx versus x
  RooPlot* frame = x.frame() ;
  gx_cdf->plotOn(frame) ;

  // Draw plot on canvas
  new TCanvas("rf16_integration","rf16_integration",600,600) ;
  frame->Draw() ;


}
