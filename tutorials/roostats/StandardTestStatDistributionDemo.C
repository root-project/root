/*
StandardTestStatDistributionDemo.C
author Kyle Cranmer
date: summer solstice, 2011

This simple script plots the sampling distribution of the profile likelihood 
ratio test statistic based on the input Model File.  To do this one needs to 
specify the value of the parameter of interest that will be used for evaluating 
the test statistic and the value of the parameters used for generating the toy data.
In this case, it uses the upper-limit estimated from the ProfileLikleihoodCalculator,
which assumes the asymptotic chi-square distribution for -2 log profile likleihood ratio.
Thus, the script is handy for checking to see if the asymptotic approximations are valid.
To aid, that comparison, the script overlays a chi-square distribution as well.

*/

#include "TFile.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TF1.h"

#include "RooWorkspace.h"
#include "RooAbsData.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/FeldmanCousins.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/ConfidenceBelt.h"

#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/SamplingDistPlot.h"

using namespace RooFit;
using namespace RooStats;


/////////////////////////////////////////////////////////////////////////
// The actual macro

void StandardTestStatDistributionDemo(const char* infile = "",
				      const char* workspaceName = "combined",
				      const char* modelConfigName = "ModelConfig",
				      const char* dataName = "obsData"){


  int nToyMC = 1000;

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
  // Now get the data and workspace
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

  mc->Print();
  /////////////////////////////////////////////////////////////
  // Now find the upper limit based on the asymptotic results
  ////////////////////////////////////////////////////////////
  RooRealVar* firstPOI = (RooRealVar*) mc->GetParametersOfInterest()->first();
  ProfileLikelihoodCalculator plc(*data,*mc);
  LikelihoodInterval* interval = plc.GetInterval();
  double plcUpperLimit = interval->UpperLimit(*firstPOI);
  delete interval;
  cout << "\n\n--------------------------------------"<<endl;
  cout <<"Will generate sampling distribution at " << firstPOI->GetName() << " = " << plcUpperLimit <<endl;
  int nPOI = mc->GetParametersOfInterest()->getSize();
  if(nPOI){
    cout <<"not sure what to do with other parameters of interest, but here are their values"<<endl;
    mc->GetParametersOfInterest()->Print("v");
  }

  /////////////////////////////////////////////
  // create thte test stat sampler
  ProfileLikelihoodTestStat ts(*mc->GetPdf());

  // temporary RooArgSet
  RooArgSet poi;
  poi.add(*mc->GetParametersOfInterest());

  // create and configure the ToyMCSampler
  ToyMCSampler sampler(ts,nToyMC);
  sampler.SetPdf(*mc->GetPdf());
  sampler.SetObservables(*mc->GetObservables());
  sampler.SetGlobalObservables(*mc->GetGlobalObservables());
  if(!mc->GetPdf()->canBeExtended() && (data->numEntries()==1)){
    cout << "tell it to use 1 event"<<endl;
    sampler.SetNEventsPerToy(1);
  }
  firstPOI->setVal(plcUpperLimit); // set POI value for generation
  sampler.SetParametersForTestStat(*mc->GetParametersOfInterest()); // set POI value for evaluation

  ProofConfig pc(*w, 4, "workers=4",false); 
  sampler.SetProofConfig(&pc);	// enable proof

  firstPOI->setVal(plcUpperLimit);
  RooArgSet allParameters;
  allParameters.add(*mc->GetParametersOfInterest());
  allParameters.add(*mc->GetNuisanceParameters());
  allParameters.Print("v");

  SamplingDistribution* sampDist = sampler.GetSamplingDistribution(allParameters);
  SamplingDistPlot plot;
  plot.AddSamplingDistribution(sampDist);
  plot.GetTH1F(sampDist)->GetYaxis()->SetTitle(Form("f(-log #lambda(#mu=%.2f) | #mu=%.2f)",plcUpperLimit,plcUpperLimit));
  plot.SetAxisTitle(Form("-log #lambda(#mu=%.2f)",plcUpperLimit));

  TCanvas* c1 = new TCanvas("c1");
  c1->SetLogy();
  plot.Draw();
  double min = plot.GetTH1F(sampDist)->GetXaxis()->GetXmin();
  double max = plot.GetTH1F(sampDist)->GetXaxis()->GetXmax();

  TF1* f = new TF1("f",Form("2*ROOT::Math::chisquared_pdf(2*x,%d,0)",nPOI),min,max);
  f->Draw("same");
  c1->SaveAs("standard_test_stat_distribution.pdf");

}
