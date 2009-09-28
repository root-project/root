//////////////////////////////////////////////////////////////////////////
//
// RooStats tutorial macro #500b
// 2009/08 - Nils Ruthmann, Gregory Schott
//
// Prepare a workspace (stored in a ROOT file) containing a models,
// data and other objects needed to run statistical classes in
// RooStats.
//
// In this macro a PDF model is built for a counting analysis.  A
// certain number of events are observed (this can be enforced or left
// free) while a number of background events is expected.  It is also
// assumed there is a systematic uncertainty on the number of expected
// background events.  The parameter of interest is the signal yield
// and we assume for it a flat prior.  All needed objects are stored
// in a ROOT file (within a RooWorkspace container); this ROOT file
// can then be fed as input to various statistical methods.
//
// root -q -x -l 'rs500b_PrepareWorkspace_Poisson_withSystematics.C()'
//
//////////////////////////////////////////////////////////////////////////


#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooPlot.h"
#include "RooWorkspace.h"

#include "TFile.h"


using namespace RooFit;

// prepare the workspace
// type = 0 : binned data with fixed number of total events
// type = 1 : binned data with N with POisson fluctuations
// type = 2 : binned data without any bin-by bin fluctuations (Asimov data)

void rs500b_PrepareWorkspace_Poisson_withSystematics( TString fileName = "WS_Poisson_withSystematics.root", int type = 1 )
{

  // use a RooWorkspace to store the pdf models, prior informations, list of parameters,...
  RooWorkspace myWS("myWS");

  // Observable
  myWS.factory("x[0,0,1]") ;

  // Pdf in observable, 
  myWS.factory("Uniform::sigPdf(x)") ;
  myWS.factory("Uniform::bkgPdf(x)") ;
  myWS.factory("SUM::model(S[100,0,1500]*sigPdf,B[1000,0,3000]*bkgPdf") ;

  // Background only pdf
  myWS.factory("ExtendPdf::modelBkg(bkgPdf,B)") ;

  // Priors
  myWS.factory("Gaussian::priorNuisance(B,1000,200)") ;
  myWS.factory("Uniform::priorPOI(S)") ;

  // Definition of observables and parameters of interest
  myWS.defineSet("observables","x");
  myWS.defineSet("POI","S");
  myWS.defineSet("parameters","B") ;
  
  // Generate data
  RooAbsData* data = 0; 
  // binned data with fixed number of events
  if (type ==0) data = myWS.pdf("model")->generateBinned(*myWS.set("observables"),myWS.var("S")->getVal(),Name("data"));
  // binned data with Poisson fluctuations
  if (type ==1) data = myWS.pdf("model")->generateBinned(*myWS.set("observables"),Extended(),Name("data"));
  // Asimov data: binned data without any fluctuations (average case) 
  if (type == 2)  data = myWS.pdf("model")->generateBinned(*myWS.set("observables"),Name("data"),ExpectedData());
  myWS.import(*data) ;

  myWS.writeToFile(fileName);  
  std::cout << "\nRooFit model initialized and stored in " << fileName << std::endl;

  // control plot of the generated data
  RooPlot* plot = myWS.var("x")->frame();
  data->plotOn(plot);
  plot->DrawClone();

}
