//////////////////////////////////////////////////////////////////////////
//
// RooStats tutorial macro #500d
// 2009/08 - Nils Ruthmann, Gregory Schott
//
// Prepare a workspace (stored in a ROOT file) containing a models,
// data and other objects needed to run statistical classes in
// RooStats.
//
// In this macro a PDF model is built assuming signal has a Gaussian
// PDF and the background a flat PDF.  The parameter of interest is
// the signal yield and we assume for it a flat prior.  It is shown
// how two types of systematics uncertainties can be expressed; those
// are a sytematic uncertainty on the background yield and another on
// one of the parameters (sigma) of the signal shape.  All needed
// objects are stored in a ROOT file (within a RooWorkspace
// container); this ROOT file can then be fed as input to various
// statistical methods.
//
// root -q -x -l 'rs500d_PrepareWorkspace_GaussOverFlat_withSystematics.C()'
//
//////////////////////////////////////////////////////////////////////////

#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooWorkspace.h"

#include "TFile.h"

using namespace RooFit;

// prepare the workspace
// type = 0 : unbinned data with total number of events fluctuating using Poisson statistics
// type = 1 : binned data with total number of events fluctuating using Poisson statistics
// type = 2 : binned data without any bin-bin fuctuation (Asimov data)

void rs500d_PrepareWorkspace_GaussOverFlat_withSystematics( TString fileName = "WS_GaussOverFlat_withSystematics.root", int type = 1 )
{
  // use a RooWorkspace to store the pdf models, prior informations, list of parameters,...
  RooWorkspace myWS("myWS");

  // Observable
  myWS.factory("mass[0,500]") ;

  // Pdf in observable 
  myWS.factory("Gaussian::sigPdf(mass,200,sigSigma[0,100])") ;
  myWS.factory("Uniform::bkgPdf(mass)") ;
  myWS.factory("SUM::model(S[1000,0,100000]*sigPdf,B[1000,0,100000]*bkgPdf") ;

  // Background only pdf
  myWS.factory("ExtendPdf::modelBkg(bkgPdf,B)") ;
  // Priors
  myWS.factory("Gaussian::prior_sigSigma(sigSigma,50,5)") ;
  myWS.factory("Gaussian::prior_B(B,1000,200)") ;
  myWS.factory("PROD::priorNuisance(prior_sigSigma,prior_B)") ;
  myWS.factory("Uniform::priorPOI(S)") ;

  // Definition of observables and parameters of interest
  myWS.defineSet("observables","mass");
  myWS.defineSet("parameters","B,sigSigma");
  myWS.defineSet("POI","S");
  
  // Generate data
  RooAbsData* data = 0;
  // unbinned data with Poisson fluctuations for total number of events
  if (type == 0) data = myWS.pdf("model")->generate(*myWS.set("observables"),Extended(),Name("data"));  
  // binned data with Poisson fluctuations for total number of events
  if (type == 1) data = myWS.pdf("model")->generateBinned(*myWS.set("observables"),Extended(),Name("data"));  
  // binned without any fluctuations (average case)
  if (type == 2) data = myWS.pdf("model")->generateBinned(*myWS.set("observables"),Name("data"),ExpectedData());

  myWS.import(*data) ;

  myWS.writeToFile(fileName);  
  std::cout << "\nRooFit model initialized and stored in " << fileName << std::endl;

  // control plot of the generated data
  RooPlot* plot = myWS.var("mass")->frame();
  data->plotOn(plot);
  plot->DrawClone();


}
