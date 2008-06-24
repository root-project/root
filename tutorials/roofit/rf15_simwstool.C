/////////////////////////////////////////////////////////////////////////
//
// RooFit tutorial macro #15
// 
// Using RooSimWSTool to construct a simultaneous p.d.f that is built
// of variations of an input p.d.f
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooPolynomial.h"
#include "RooSimultaneous.h"
#include "RooAddPdf.h"
#include "RooWorkspace.h"
#include "RooSimWSTool.h"
#include "RooPlot.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
using namespace RooFit ;


void rf15_simwstool()
{
  // Create p.d.f the be used below
  
  // Construct gauss(x,m,s)
  RooRealVar x("x","x",-10,10) ;
  RooRealVar m("m","m",0,-10,10) ;
  RooRealVar s("s","s",1,-10,10) ;
  RooGaussian gauss("g","g",x,m,s) ;

  // Construct poly(x,p0)
  RooRealVar p0("p0","p0",0.01,0.,1.) ;
  RooPolynomial poly("p","p",x,p0) ;		 

  // Construct model = f*gauss(x) + (1-f)*poly(x)
  RooRealVar f("f","f",0.5,0.,1.) ;
  RooAddPdf model("model","model",RooArgSet(gauss,poly),f) ;

  // Define two categories that can be used for splitting
  RooCategory c("c","c") ;
  c.defineType("run1") ;
  c.defineType("run2") ;

  RooCategory d("d","d") ;
  d.defineType("foo") ;
  d.defineType("bar") ;


  // Use combination of RooWorkspace and RooSimWSTool
  // to clone and customize above model into a simulatenous p.d.f.

  // Import ingredients in a workspace
  RooWorkspace w("w","w") ;
  w.import(RooArgSet(model,c,d)) ;

  // Make Sim builder tool
  RooSimWSTool sct(w) ;

  // Construct a simultaneous p.d.f with the following form
  //
  // model_run1(x) = f*gauss_run1(x,m_run1,s) + (1-f)*poly
  // model_run2(x) = f*gauss_run2(x,m_run2,s) + (1-f)*poly
  // simpdf(x,c) = model_run1(x) if c=="run1"
  //             = model_run2(x) if c=="run2"
  //
  // Returned p.d.f is owned by the workspace
  RooSimultaneous* model_sim = sct.build("model_sim","model",SplitParam("m","c")) ;

  // Print tree structure of model
  model_sim->Print("t") ;

  // Adjust model_sim parameters in workspace
  w.var("m_run1")->setVal(-3) ;
  w.var("m_run2")->setVal(+3) ;

  // Print contents of workspace
  w.Print("v") ;

  // Build another simultaneous p.d.f using a composite split in states c X d
  RooSimultaneous* model_sim2 = sct.build("model_sim2","model",SplitParam("p0","c,d")) ;

  // Print tree structure of this model
  model_sim2->Print("t") ;

}
