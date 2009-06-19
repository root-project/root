/////////////////////////////////////////////////////////////////////////
//
// 'Limit Example' RooStats tutorial macro #101
// author: Kyle Cranmer
// date June. 2009
//
// This tutorial shows an example of creating a simple
// model for a number counting experiment with uncertainty
// on both the background rate and signal efficeincy. We then 
// use a Confidence Interval Calculator to set a limit on the signal.
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/FeldmanCousins.h"
#include "RooStats/NumberCountingPdfFactory.h"
#include "RooStats/ConfInterval.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooProfileLL.h"
#include "RooAbsPdf.h"
#include "RooStats/HypoTestResult.h"
#include "RooRealVar.h"
#include "RooPlot.h"
#include "RooDataSet.h"
#include "RooTreeDataStore.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStopwatch.h"

#include "RooStats/RooStatsUtils.h"

// use this order for safety on library loading
using namespace RooFit ;
using namespace RooStats ;


void rs101_limitexample()
{
  /////////////////////////////////////////
  // An example of setting a limit in a number counting experiment with uncertainty on background and signal
  /////////////////////////////////////////

  // to time the macro
  TStopwatch t;
  t.Start();

  /////////////////////////////////////////
  // The Model building stage
  /////////////////////////////////////////
  RooWorkspace* wspace = new RooWorkspace();
  wspace->factory("Poisson::countingModel(obs[150,0,300], sum(s[50,0,100]*ratioSigEff[1.,0,2.],b[100,0,300]*ratioBkgEff[1.,0.,2.]))"); // counting model
  wspace->factory("Gaussian::sigConstraint(ratioSigEff,1,0.05)"); // 5% signal efficiency uncertainty
  wspace->factory("Gaussian::bkgConstraint(ratioBkgEff,1,0.1)"); // 10% background efficiency uncertainty
  wspace->factory("PROD::modelWithConstraints(countingModel,sigConstraint,bkgConstraint)"); // product of terms
  wspace->Print();

  RooAbsPdf* modelWithConstraints = wspace->pdf("modelWithConstraints"); // get the model
  RooRealVar* obs = wspace->var("obs"); // get the observable
  RooRealVar* s = wspace->var("s"); // get the signal we care about
  RooRealVar* b = wspace->var("b"); // get the background and set it to a constant.  Uncertainty included in ratioBkgEff
  b->setConstant();
  RooRealVar* ratioSigEff = wspace->var("ratioSigEff"); // get uncertaint parameter to constrain
  RooRealVar* ratioBkgEff = wspace->var("ratioBkgEff"); // get uncertaint parameter to constrain
  RooArgSet constrainedParams(*ratioSigEff, *ratioBkgEff); // need to constrain these in the fit (should change default behavior)

  // Create an example dataset with 160 observed events
  obs->setVal(160.);
  RooDataSet* data = new RooDataSet("exampleData", "exampleData", RooArgSet(*obs));
  data->add(*obs);


  // not necessary
  modelWithConstraints->fitTo(*data, RooFit::Constrain(RooArgSet(*ratioSigEff, *ratioBkgEff)));

  // Now let's make some confidence intervals for s, our parameter of interest
  RooArgSet paramOfInterest(*s);

  // First, let's use a Calculator based on the Profile Likelihood Ratio
  ProfileLikelihoodCalculator plc;
  //  plc.SetWorkspace(*wspace);
  plc.SetPdf(*modelWithConstraints);
  plc.SetData(*data); 
  plc.SetParameters( paramOfInterest );
  plc.SetTestSize(.1);
  ConfInterval* lrint = plc.GetInterval();  // that was easy.


  // Second, use a Calculator based on the Feldman Cousins technique
  FeldmanCousins fc;
  //  fc.SetWorkspace(*wspace);
  fc.SetPdf(*modelWithConstraints);
  fc.SetData(*data); 
  fc.SetParameters( paramOfInterest );
  fc.UseAdaptiveSampling(true);
  fc.FluctuateNumDataEntries(false); // number counting analysis: dataset always has 1 entry with N events observed
  fc.SetNBins(100); // number of points to test per parameter
  fc.SetTestSize(.1);
  ConfInterval* fcint = 0;
  fcint = fc.GetInterval();  // that was easy.


  // Let's make a plot
  TCanvas* dataCanvas = new TCanvas("dataCanvas");
  dataCanvas->Divide(2,1);

  dataCanvas->cd(1);
  LikelihoodIntervalPlot plotInt((LikelihoodInterval*)lrint);
  plotInt.SetTitle("Parameters contour plot");
  plotInt.Draw();


  // Get Lower and Upper limits from Profile Calculator
  cout << "Profile lower limit on s = " << ((LikelihoodInterval*) lrint)->LowerLimit(*s) << endl;
  cout << "Profile upper limit on s = " << ((LikelihoodInterval*) lrint)->UpperLimit(*s) << endl;

  // Get Lower and Upper limits from FeldmanCousins with profile construction
  double fcul = ((PointSetInterval*) fcint)->UpperLimit(*s);
  double fcll = ((PointSetInterval*) fcint)->LowerLimit(*s);
  cout << "FC lower limit on s = " << fcll << endl;
  cout << "FC upper limit on s = " << fcul << endl;
  TLine* fcllLine = new TLine(fcll, 0, fcll, 1);
  TLine* fculLine = new TLine(fcul, 0, fcul, 1);
  fcllLine->SetLineColor(kRed);
  fculLine->SetLineColor(kRed);
  fcllLine->Draw("same");
  fculLine->Draw("same");
  dataCanvas->Update();

  // plot the points used in the profile construction
  dataCanvas->cd(2);
  TTree& parameterScan =  ((RooTreeDataStore*) fc.GetPointsToScan()->store())->tree();
  parameterScan.SetMarkerStyle(20);
  parameterScan.Draw("s:ratioSigEff:ratioBkgEff","","");


  delete wspace;
  delete lrint;
  delete fcint;
  delete data;

  /// print timing info
  t.Stop();
  t.Print();
}
