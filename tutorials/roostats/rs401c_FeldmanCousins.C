/////////////////////////////////////////////////////////////////////////
//
// 'Debugging Sampling Distribution' RooStats tutorial macro #401
// author: Kyle Cranmer
// date Jan. 2009
//
// This tutorial shows usage of a distribution creator, sampling distribution,
// and the Neyman Construction.
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooStats/ConfInterval.h"
#include "RooStats/ConfidenceBelt.h"
#include "RooStats/FeldmanCousins.h"

#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooAddition.h"

#include "RooDataHist.h"

#include "RooPoisson.h"
#include "RooPlot.h"

#include "TCanvas.h"
#include "TTree.h"
#include "TH1F.h"
#include "TMarker.h"
#include "TStopwatch.h"

#include <iostream>

// use this order for safety on library loading
using namespace RooFit ;
using namespace RooStats ;


void rs401c_FeldmanCousins()
{
  // to time the macro
  TStopwatch t;
  t.Start();

  // make a simple model
  RooRealVar x("x","", 1,0,50);
  RooRealVar mu("mu","", 2.5,0, 15); // with a limit on mu>=0
  RooConstVar b("b","", 3.);
  RooAddition mean("mean","",RooArgList(mu,b));
  RooPoisson pois("pois", "", x, mean);
  RooArgSet parameters(mu);

  Int_t nEventsData = 1;

  // create a toy dataset
  RooDataSet* data = pois.generate(RooArgSet(x), nEventsData);
  
  std::cout << "This data has mean, stdev = " << data->moment(x,1,0.) << ", " << data->moment(x,2,data->moment(x,1,0.) ) << endl; 

  TCanvas* dataCanvas = new TCanvas("dataCanvas");
  RooPlot* frame = x.frame();
  data->plotOn(frame);
  frame->Draw();
  dataCanvas->Update();


  //////// show use of Feldman-Cousins
  RooStats::FeldmanCousins fc;
  // set the distribution creator, which encodes the test statistic
  fc.SetPdf(pois);
  fc.SetParameters(parameters);
  fc.SetTestSize(.05); // set size of test
  fc.SetData(*data);
  fc.UseAdaptiveSampling(true);
  fc.FluctuateNumDataEntries(false); // number counting analysis: dataset always has 1 entry with N events observed
  fc.SetNBins(30); // number of points to test per parameter

  // use the Feldman-Cousins tool
  ConfInterval* interval = fc.GetInterval();

  ConfidenceBelt* belt = 0;
  //  belt = fc.GetConfidenceBelt();

  // make a canvas for plots
  new TCanvas("intervalCanvas");
  
  std::cout << "is this point in the interval? " << 
    interval->IsInInterval(parameters) << std::endl;
  

  RooDataHist* parameterScan = (RooDataHist*) fc.GetPointsToScan();
  TH1F* hist = (TH1F*) parameterScan->createHistogram("mu",30);
  hist->Draw();

 
  RooArgSet* tmpPoint;
  // loop over points to test
  for(Int_t i=0; i<parameterScan->numEntries(); ++i){
    //    cout << "on parameter point " << i << " out of " << parameterScan->numEntries() << endl;
     // get a parameter point from the list of points to test.
    tmpPoint = (RooArgSet*) parameterScan->get(i)->clone("temp");

    if(belt){ 
      // use belt
      cout << "belt = " << belt << endl;
      cout << "belt:" << tmpPoint->getRealValue("mu")
	   << belt->GetAcceptanceRegionMin(*tmpPoint) 
	   << " - " 
	   << belt->GetAcceptanceRegionMax(*tmpPoint) 
	   << endl;
    }

    TMarker* mark = new TMarker(tmpPoint->getRealValue("mu"), 1, 25);
    if (interval->IsInInterval( *tmpPoint ) ) 
      mark->SetMarkerColor(kBlue);
    else
      mark->SetMarkerColor(kRed);

    mark->Draw("s");
    //delete tmpPoint;
    //    delete mark;
  }
  t.Stop();
  t.Print();
    

}
