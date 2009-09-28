//////////////////////////////////////////////////////////////////////////
//
// RooStats tutorial macro #501
// 2009/08 - Nils Ruthmann, Gregory Schott
//
// Show how to run the RooStats classes to perform specific tasks. The
// ROOT file containing a workspace holding the models, data and other
// objects needed to run can be prepared with any of the rs500*.C
// tutorial macros.
//
// Compute with ProfileLikelihoodCalculator a 95% CL upper limit on
// the parameter of interest for the given data.
//
//////////////////////////////////////////////////////////////////////////

#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodIntervalPlot.h"

#include "TFile.h"

using namespace RooFit;
using namespace RooStats;


void rs501_ProfileLikelihoodCalculator_limit( const char* fileName="WS_GaussOverFlat.root" )
{
  // Open the ROOT file and import from the workspace the objects needed for this tutorial
  TFile* file = new TFile(fileName);
  RooWorkspace* myWS = (RooWorkspace*) file->Get("myWS");
  RooAbsPdf* modelTmp = myWS->pdf("model");
  RooAbsData* data = myWS->data("data");
  RooAbsPdf* priorNuisance = myWS->pdf("priorNuisance");
  const RooArgSet* POI = myWS->set("POI");
  RooRealVar* parameterOfInterest = dynamic_cast<RooRealVar*>(POI->first());
  assert(parameterOfInterest);

  // If there are nuisance parameters, multiply their prior distribution to the full model
  RooAbsPdf* model = modelTmp;
  if( priorNuisance!=0 ) model = new RooProdPdf("constrainedModel","Model with nuisance parameters",*modelTmp,*priorNuisance);

  // Set up the ProfileLikelihoodCalculator
  ProfileLikelihoodCalculator plc(*data, *model, *POI);
  // ProfileLikelihoodCalculator usually make intervals: the 95% CL one-sided upper-limit is the same as the two-sided upper-limit of a 90% CL interval  
  plc.SetTestSize(0.10);

  // Pointer to the confidence interval
  model->fitTo(*data,SumW2Error(kFALSE)); // <-- problem
  LikelihoodInterval* interval = plc.GetInterval();

  // Compute the upper limit: a fit is needed first in order to locate the minimum of the -log(likelihood) and ease the upper limit computation
  model->fitTo(*data,SumW2Error(kFALSE)); // <-- problem
  const double upperLimit = interval->UpperLimit(*parameterOfInterest); // <-- to simplify

  file->Close();

  // Make a plot of the profile-likelihood and confidence interval
  LikelihoodIntervalPlot plot(interval);
  plot.Draw();

  std::cout << "One sided upper limit at 95% CL: "<< upperLimit << std::endl;

  delete model;
}
