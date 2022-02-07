//////////////////////////////////////////////////////////////////////////
//
// 'MULTIDIMENSIONAL MODELS' RooFit tutorial macro #310
//
// Projecting p.d.f and data ranges in continuous observables
//
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
#include "RooAddPdf.h"
#include "RooPolynomial.h"
#include "TCanvas.h"
#include "RooPlot.h"
using namespace RooFit ;


class TestBasic311 : public RooFitTestUnit
{
public:
  TestBasic311(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit("Data and p.d.f projection in sub range",refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

  // C r e a t e   3 D   p d f   a n d   d a t a
  // -------------------------------------------

  // Create observables
  RooRealVar x("x","x",-5,5) ;
  RooRealVar y("y","y",-5,5) ;
  RooRealVar z("z","z",-5,5) ;

  // Create signal pdf gauss(x)*gauss(y)*gauss(z)
  RooGaussian gx("gx","gx",x,RooConst(0),RooConst(1)) ;
  RooGaussian gy("gy","gy",y,RooConst(0),RooConst(1)) ;
  RooGaussian gz("gz","gz",z,RooConst(0),RooConst(1)) ;
  RooProdPdf sig("sig","sig",RooArgSet(gx,gy,gz)) ;

  // Create background pdf poly(x)*poly(y)*poly(z)
  RooPolynomial px("px","px",x,RooArgSet(RooConst(-0.1),RooConst(0.004))) ;
  RooPolynomial py("py","py",y,RooArgSet(RooConst(0.1),RooConst(-0.004))) ;
  RooPolynomial pz("pz","pz",z) ;
  RooProdPdf bkg("bkg","bkg",RooArgSet(px,py,pz)) ;

  // Create composite pdf sig+bkg
  RooRealVar fsig("fsig","signal fraction",0.1,0.,1.) ;
  RooAddPdf model("model","model",RooArgList(sig,bkg),fsig) ;

  RooDataSet* data = model.generate(RooArgSet(x,y,z),20000) ;



  // P r o j e c t   p d f   a n d   d a t a   o n   x
  // -------------------------------------------------

  // Make plain projection of data and pdf on x observable
  RooPlot* frame = x.frame(Title("Projection of 3D data and pdf on X"),Bins(40)) ;
  data->plotOn(frame) ;
  model.plotOn(frame) ;



  // P r o j e c t   p d f   a n d   d a t a   o n   x   i n   s i g n a l   r a n g e
  // ----------------------------------------------------------------------------------

  // Define signal region in y and z observables
  y.setRange("sigRegion",-1,1) ;
  z.setRange("sigRegion",-1,1) ;

  // Make plot frame
  RooPlot* frame2 = x.frame(Title("Same projection on X in signal range of (Y,Z)"),Bins(40)) ;

  // Plot subset of data in which all observables are inside "sigRegion"
  // For observables that do not have an explicit "sigRegion" range defined (e.g. observable)
  // an implicit definition is used that is identical to the full range (i.e. [-5,5] for x)
  data->plotOn(frame2,CutRange("sigRegion")) ;

  // Project model on x, integrating projected observables (y,z) only in "sigRegion"
  model.plotOn(frame2,ProjectionRange("sigRegion")) ;


  regPlot(frame,"rf311_plot1") ;
  regPlot(frame2,"rf312_plot2") ;

  delete data ;

  return kTRUE ;
  }
} ;
