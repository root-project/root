//////////////////////////////////////////////////////////////////////////
//
// RooStats tutorial macro #502
// 2009/08 - Nils Ruthmann, Gregory Schott
//
// Show how to run the RooStats classes to perform specific tasks. The
// ROOT file containing a workspace holding the models, data and other
// objects needed to run can be prepared with any of the rs500*.C
// tutorial macros.
//
// Compute with ProfileLikelihoodCalculator a 68% CL interval and the
// signal significance for the given data.
//
//////////////////////////////////////////////////////////////////////////


#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/HypoTestResult.h"


#include "TFile.h"

using namespace RooFit;
using namespace RooStats;


void rs502_ProfileLikelihoodCalculator_significance( const char* fileName="WS_GaussOverFlat.root" )
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
   //myWS->var("B")->setConstant();
  
   // Set up the ProfileLikelihoodCalculator
   ProfileLikelihoodCalculator plc(*data, *model, *POI);
   // The 68% CL ProfileLikelihoodCalculator interval correspond to test size of 0.32
   plc.SetTestSize(0.32);


   // Compute the confidence interval: a fit is needed first in order to locate the minimum of the -log(likelihood) and ease the upper limit computation

   model->fitTo(*data,SumW2Error(kFALSE));
   // Pointer to the confidence interval
   LikelihoodInterval* interval = plc.GetInterval();

   const double MLE = parameterOfInterest->getVal();
   const double lowerLimit = interval->LowerLimit(*parameterOfInterest);
   parameterOfInterest->setVal(MLE);
   const double upperLimit = interval->UpperLimit(*parameterOfInterest);
   parameterOfInterest->setVal(MLE);

   // Make a plot of the profile-likelihood and confidence interval
   LikelihoodIntervalPlot plot(interval);
   plot.Draw();

   // Get the significance using the GetHypoTest function: (a plot is not possible) 


   // create a copy of the POI parameters to set the values to zero
   RooArgSet nullparams;
   nullparams.addClone(*parameterOfInterest);
   ( (RooRealVar *) (nullparams.first()))->setVal(0);
   plc.SetNullParameters(nullparams);

   HypoTestResult* testresult=plc.GetHypoTest();
   const double significance = testresult->Significance();

//   // Another way is to use directly 
//   // the profile-log-likelihood function
//   parameterOfInterest->setVal(MLE);
//   RooAbsReal* profile = interval->GetLikelihoodRatio();
//   profile->getVal();
//   //Go to the Bkg Hypothesis
//   RooRealVar* myarg = (RooRealVar *) profile->getVariables()->find(parameterOfInterest->GetName()); // <-- cloned!
//   myarg->setVal(0);
//   Double_t delta_nll = profile->getVal();
//   double significance = sqrt(fabs(2*delta_nll));
//   if (delta_nll<0) significance *= -1;



   std::cout << "68% CL interval: [ " << lowerLimit << " ; " << upperLimit << " ]\n";
   std::cout << "significance estimation: " << significance << std::endl;

   delete model;
   // closing the file will delete the workspace
   //file->Close();
}
