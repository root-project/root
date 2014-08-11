// Standard demo of the Feldman-Cousins calculator
/*
StandardFeldmanCousinsDemo

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

The FeldmanCousins tools is a classical frequentist calculation
based on the Neyman Construction.  The test statistic can be
generalized for nuisance parameters by using the profile likeihood ratio.
But unlike the ProfileLikelihoodCalculator, this tool explicitly
builds the sampling distribution of the test statistic via toy Monte Carlo.
*/

#include "TFile.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TSystem.h"

#include "RooWorkspace.h"
#include "RooAbsData.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/FeldmanCousins.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/ConfidenceBelt.h"


using namespace RooFit;
using namespace RooStats;

void StandardFeldmanCousinsDemo(const char* infile = "",
                                const char* workspaceName = "combined",
                                const char* modelConfigName = "ModelConfig",
                                const char* dataName = "obsData"){

  /////////////////////////////////////////////////////////////
  // First part is just to access a user-defined file
  // or create the standard example file if it doesn't exist
  ////////////////////////////////////////////////////////////
  const char* filename = "";
  if (!strcmp(infile,"")) {
    filename = "results/example_combined_GaussExample_model.root";
     bool fileExist = !gSystem->AccessPathName(filename); // note opposite return code
     // if file does not exists generate with histfactory
     if (!fileExist) {
#ifdef _WIN32
        cout << "HistFactory file cannot be generated on Windows - exit" << endl;
        return;
#endif
        // Normally this would be run on the command line
        cout <<"will run standard hist2workspace example"<<endl;
        gROOT->ProcessLine(".! prepareHistFactory .");
        gROOT->ProcessLine(".! hist2workspace config/example.xml");
        cout <<"\n\n---------------------"<<endl;
        cout <<"Done creating example input"<<endl;
        cout <<"---------------------\n\n"<<endl;
     }

  }
  else
    filename = infile;

  // Try to open the file
  TFile *file = TFile::Open(filename);

  // if input file was specified byt not found, quit
  if(!file ){
    cout <<"StandardRooStatsDemoMacro: Input file " << filename << " is not found" << endl;
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
  // create and use the FeldmanCousins tool
  // to find and plot the 95% confidence interval
  // on the parameter of interest as specified
  // in the model config
  FeldmanCousins fc(*data,*mc);
  fc.SetConfidenceLevel(0.95); // 95% interval
  //fc.AdditionalNToysFactor(0.1); // to speed up the result
  fc.UseAdaptiveSampling(true); // speed it up a bit
  fc.SetNBins(10); // set how many points per parameter of interest to scan
  fc.CreateConfBelt(true); // save the information in the belt for plotting

  // Since this tool needs to throw toy MC the PDF needs to be
  // extended or the tool needs to know how many entries in a dataset
  // per pseudo experiment.
  // In the 'number counting form' where the entries in the dataset
  // are counts, and not values of discriminating variables, the
  // datasets typically only have one entry and the PDF is not
  // extended.
  if(!mc->GetPdf()->canBeExtended()){
    if(data->numEntries()==1)
      fc.FluctuateNumDataEntries(false);
    else
      cout <<"Not sure what to do about this model" <<endl;
  }

  // We can use PROOF to speed things along in parallel
  //  ProofConfig pc(*w, 1, "workers=4", kFALSE);
  //  ToyMCSampler*  toymcsampler = (ToyMCSampler*) fc.GetTestStatSampler();
  //  toymcsampler->SetProofConfig(&pc); // enable proof


  // Now get the interval
  PointSetInterval* interval = fc.GetInterval();
  ConfidenceBelt* belt = fc.GetConfidenceBelt();

  // print out the iterval on the first Parameter of Interest
  RooRealVar* firstPOI = (RooRealVar*) mc->GetParametersOfInterest()->first();
  cout << "\n95% interval on " <<firstPOI->GetName()<<" is : ["<<
    interval->LowerLimit(*firstPOI) << ", "<<
    interval->UpperLimit(*firstPOI) <<"] "<<endl;

  //////////////////////////////////////////////
  // No nice plots yet, so plot the belt by hand

  // Ask the calculator which points were scanned
  RooDataSet* parameterScan = (RooDataSet*) fc.GetPointsToScan();
  RooArgSet* tmpPoint;

  // make a histogram of parameter vs. threshold
  TH1F* histOfThresholds = new TH1F("histOfThresholds","",
                                    parameterScan->numEntries(),
                                    firstPOI->getMin(),
                                    firstPOI->getMax());

  // loop through the points that were tested and ask confidence belt
  // what the upper/lower thresholds were.
  // For FeldmanCousins, the lower cut off is always 0
  for(Int_t i=0; i<parameterScan->numEntries(); ++i){
    tmpPoint = (RooArgSet*) parameterScan->get(i)->clone("temp");
    double arMax = belt->GetAcceptanceRegionMax(*tmpPoint);
    double arMin = belt->GetAcceptanceRegionMax(*tmpPoint);
    double poiVal = tmpPoint->getRealValue(firstPOI->GetName()) ;
    histOfThresholds->Fill(poiVal,arMax);
  }
  histOfThresholds->SetMinimum(0);
  histOfThresholds->Draw();

}
