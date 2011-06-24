// Standard demo of the Bayesian MCMC calculator
 
/*

Author: Kyle Cranmer
date: Dec. 2010

This is a standard demo that can be used with any ROOT file 
prepared in the standard way.  You specify:
 - name for input ROOT file
 - name of workspace inside ROOT file that holds model and data
 - name of ModelConfig that specifies details for calculator tools
 - name of dataset 

With default parameters the macro will attempt to run the 
standard hist2workspace example and read the ROOT file
that it produces.

The actual heart of the demo is only about 10 lines long.

The MCMCCalculator is a Bayesian tool that uses
the Metropolis-Hastings algorithm to efficiently integrate
in many dimensions.  It is not as accurate as the BayesianCalculator
for simple problems, but it scales to much more complicated cases.
*/

#include "TFile.h"
#include "TROOT.h"
#include "RooWorkspace.h"
#include "RooAbsData.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/MCMCIntervalPlot.h"

using namespace RooFit;
using namespace RooStats;

void StandardBayesianMCMCDemo(const char* infile = "",
		      const char* workspaceName = "combined",
		      const char* modelConfigName = "ModelConfig",
		      const char* dataName = "obsData"){

  /////////////////////////////////////////////////////////////
  // First part is just to access a user-defined file 
  // or create the standard example file if it doesn't exist
  ////////////////////////////////////////////////////////////
  const char* filename = "";
  if (!strcmp(infile,""))
    filename = "results/example_combined_GaussExample_model.root";
  else
    filename = infile;
  // Check if example input file exists
  TFile *file = TFile::Open(filename);

  // if input file was specified byt not found, quit
  if(!file && strcmp(infile,"")){
    cout <<"file not found" << endl;
    return;
  } 

  // if default file not found, try to create it
  if(!file ){
    // Normally this would be run on the command line
    cout <<"will run standard hist2workspace example"<<endl;
    gROOT->ProcessLine(".! prepareHistFactory .");
    gROOT->ProcessLine(".! hist2workspace config/example.xml");
    cout <<"\n\n---------------------"<<endl;
    cout <<"Done creating example input"<<endl;
    cout <<"---------------------\n\n"<<endl;
  }

  // now try to access the file again
  file = TFile::Open(filename);
  if(!file){
    // if it is still not there, then we can't continue
    cout << "Not able to run hist2workspace to create example input" <<endl;
    return;
  }

  
  /////////////////////////////////////////////////////////////
  // Tutorial starts here
  ////////////////////////////////////////////////////////////

  // get the workspace out of the file
  RooWorkspace* w = (RooWorkspace*) file->Get(workspaceName);
  if(!w){
    cout <<"workspace not found" << endl;
    return;
  }

  // get the modelConfig out of the file
  ModelConfig* mc = (ModelConfig*) w->obj(modelConfigName);

  // get the modelConfig out of the file
  RooAbsData* data = w->data(dataName);

  // make sure ingredients are found
  if(!data || !mc){
    w->Print();
    cout << "data or ModelConfig was not found" <<endl;
    return;
  }

  /////////////////////////////////////////////
  // create and use the MCMCCalculator
  // to find and plot the 95% credible interval
  // on the parameter of interest as specified
  // in the model config
  MCMCCalculator mcmc(*data,*mc);
  mcmc.SetConfidenceLevel(0.95); // 95% interval
  mcmc.SetNumIters(1000000);         // Metropolis-Hastings algorithm iterations
  mcmc.SetNumBurnInSteps(500);       // first N steps to be ignored as burn-in

  // default is the shortest interval.  here use central
  mcmc.SetLeftSideTailFraction(0.5); // for central interval

  MCMCInterval* interval = mcmc.GetInterval();

  // make a plot
  MCMCIntervalPlot plot(*interval);
  plot.Draw();
  
  // print out the iterval on the first Parameter of Interest
  RooRealVar* firstPOI = (RooRealVar*) mc->GetParametersOfInterest()->first();
  cout << "\n95% interval on " <<firstPOI->GetName()<<" is : ["<<
    interval->LowerLimit(*firstPOI) << ", "<<
    interval->UpperLimit(*firstPOI) <<"] "<<endl;

}
