// Standard demo of the numerical Bayesian calculator

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

The BayesianCalculator is based on Bayes's theorem
and performs the integration using ROOT's numeric integration utilities
*/

#include "TFile.h"
#include "TROOT.h"
#include "RooWorkspace.h"
#include "RooAbsData.h"
#include "RooRealVar.h"

#include "RooUniform.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"
#include "RooPlot.h"

using namespace RooFit;
using namespace RooStats;


TString integrationType = "";  // integration Type (default is adaptive (numerical integration)
                               // possible values are "TOYMC" (toy MC integration, work when nuisances have a constraints pdf)
                               //  "VEGAS" , "MISER", or "PLAIN"  (these are all possible MC integration) 
int nToys = 10000;             // number of toys used for the MC integrations - for Vegas should be probably set to an higher value 
bool scanPosterior = false;    // flag to compute interval by scanning posterior (it is more robust but maybe less precise) 
int nScanPoints = 50;          // number of points for scanning the posterior (if scanPosterior = false it is used only for plotting)
int intervalType = 1;          // type of interval (0 is shortest, 1 central, 2 upper limit) 

void StandardBayesianNumericalDemo(const char* infile = "",
		      const char* workspaceName = "combined",
		      const char* modelConfigName = "ModelConfig",
		      const char* dataName = "obsData"){

  /////////////////////////////////////////////////////////////
  // First part is just to access a user-defined file 
  // or create the standard example file if it doesn't exist
  ////////////////////////////////////////////////////////////
  TString filename = infile;
  if (filename.IsNull()) { 
    filename = "results/example_combined_GaussExample_model.root";
    std::cout << "Use standard file generated with HistFactory : " << filename << std::endl;
  }

  // Check if example input file exists
  TFile *file = TFile::Open(filename);

  // if input file was specified but not found, quit
  if(!file && !TString(infile).IsNull()){
     cout <<"file " << filename << " not found" << endl;
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

    // now try to access the file again
    file = TFile::Open(filename);
  }

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
  // create and use the BayesianCalculator
  // to find and plot the 95% credible interval
  // on the parameter of interest as specified
  // in the model config
  
  // before we do that, we must specify our prior
  // it belongs in the model config, but it may not have
  // been specified
  RooUniform prior("prior","",*mc->GetParametersOfInterest());
  w->import(prior);
  mc->SetPriorPdf(*w->pdf("prior"));

  // do without systematics
  //mc->SetNuisanceParameters(RooArgSet() );

  
  BayesianCalculator bayesianCalc(*data,*mc);
  bayesianCalc.SetConfidenceLevel(0.95); // 95% interval

  // default of the calculator is central interval.  here use shortest , central or upper limit depending on input
  // doing a shortest interval might require a longer time since it requires a scan of the posterior function
  if (intervalType == 0)  bayesianCalc.SetShortestInterval(); // for shortest interval
  if (intervalType == 1)  bayesianCalc.SetLeftSideTailFraction(0.5); // for central interval
  if (intervalType == 2)  bayesianCalc.SetLeftSideTailFraction(0.); // for upper limit

  if (!integrationType.IsNull() ) { 
     bayesianCalc.SetIntegrationType(integrationType); // set integrationType
     bayesianCalc.SetNumIters(nToys); // set number of ietrations (i.e. number of toys for MC integrations)
  }

  // compute interval by scanning the posterior function
  if (scanPosterior)   
     bayesianCalc.SetScanOfPosterior(nScanPoints);


  SimpleInterval* interval = bayesianCalc.GetInterval();

  // print out the iterval on the first Parameter of Interest
  RooRealVar* firstPOI = (RooRealVar*) mc->GetParametersOfInterest()->first();
  cout << "\n95% interval on " <<firstPOI->GetName()<<" is : ["<<
    interval->LowerLimit() << ", "<<
    interval->UpperLimit() <<"] "<<endl;


  // make a plot 
  // since plotting may take a long time (it requires evaluating 
  // the posterior in many points) this command will speed up 
  // by reducing the number of points to plot - do 50

  cout << "\nDrawing plot of posterior function....." << endl;

  bayesianCalc.SetScanOfPosterior(nScanPoints);

  RooPlot * plot = bayesianCalc.GetPosteriorPlot();
  plot->Draw();  

}
