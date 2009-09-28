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

#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/UniformProposal.h"
#include "RooStats/FeldmanCousins.h"
#include "RooStats/NumberCountingPdfFactory.h"
#include "RooStats/ConfInterval.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"
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
  wspace->factory("Poisson::countingModel(obs[150,0,300], sum(s[50,0,120]*ratioSigEff[1.,0,2.],b[100,0,300]*ratioBkgEff[1.,0.,2.]))"); // counting model
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

  RooArgSet all(*s, *ratioBkgEff, *ratioSigEff);

  // not necessary
  modelWithConstraints->fitTo(*data, RooFit::Constrain(RooArgSet(*ratioSigEff, *ratioBkgEff)));

  // Now let's make some confidence intervals for s, our parameter of interest
  RooArgSet paramOfInterest(*s);

  // First, let's use a Calculator based on the Profile Likelihood Ratio
  ProfileLikelihoodCalculator plc(*data, *modelWithConstraints, paramOfInterest); 
  plc.SetTestSize(.1);
  ConfInterval* lrint = plc.GetInterval();  // that was easy.

  // Second, use a Calculator based on the Feldman Cousins technique
  FeldmanCousins fc;
  fc.SetPdf(*modelWithConstraints);
  fc.SetData(*data); 
  fc.SetParameters( paramOfInterest );
  fc.UseAdaptiveSampling(true);
  fc.FluctuateNumDataEntries(false); // number counting analysis: dataset always has 1 entry with N events observed
  fc.SetNBins(100); // number of points to test per parameter
  fc.SetTestSize(.1);
  //  fc.SaveBeltToFile(true); // optional
  ConfInterval* fcint = NULL;
  fcint = fc.GetInterval();  // that was easy.


  // Third, use a Calculator based on Markov Chain monte carlo
  UniformProposal up;
  MCMCCalculator mc;
  mc.SetPdf(*modelWithConstraints);
  mc.SetData(*data);
  mc.SetParameters(paramOfInterest);
  mc.SetProposalFunction(up);
  mc.SetNumIters(100000); // steps in the chain
  mc.SetTestSize(.1); // 90% CL
  mc.SetNumBins(50); // used in posterior histogram
  mc.SetNumBurnInSteps(40); // ignore first steps in chain due to "burn in"
  ConfInterval* mcmcint = NULL;
  mcmcint = mc.GetInterval();

  // Let's make a plot
  TCanvas* dataCanvas = new TCanvas("dataCanvas");
  dataCanvas->Divide(2,1);


  dataCanvas->cd(1);
  LikelihoodIntervalPlot plotInt((LikelihoodInterval*)lrint);
  plotInt.SetTitle("Profile Likelihood Ratio and Posterior for S");
  plotInt.Draw();

  // draw posterior
  TH1* posterior = ((MCMCInterval*)mcmcint)->GetPosteriorHist();  
  posterior->Scale(1/posterior->GetBinContent(posterior->GetMaximumBin())); // scale so highest bin has y=1.
  posterior->Draw("same");

  // Get Lower and Upper limits from Profile Calculator
  cout << "Profile lower limit on s = " << ((LikelihoodInterval*) lrint)->LowerLimit(*s) << endl;
  cout << "Profile upper limit on s = " << ((LikelihoodInterval*) lrint)->UpperLimit(*s) << endl;

  // Get Lower and Upper limits from FeldmanCousins with profile construction
  if (fcint != NULL) {
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
  }

  // Get Lower and Upper limits from MCMC
  double mcul = ((MCMCInterval*) mcmcint)->UpperLimit(*s);
  double mcll = ((MCMCInterval*) mcmcint)->LowerLimit(*s);
  cout << "MCMC lower limit on s = " << mcll << endl;
  cout << "MCMC upper limit on s = " << mcul << endl;
  TLine* mcllLine = new TLine(mcll, 0, mcll, 1);
  TLine* mculLine = new TLine(mcul, 0, mcul, 1);
  mcllLine->SetLineColor(kMagenta);
  mculLine->SetLineColor(kMagenta);
  mcllLine->Draw("same");
  mculLine->Draw("same");
  dataCanvas->Update();

  // 3-d plot of the parameter points
  dataCanvas->cd(2);
  // also plot the points in the markov chain
  TTree& chain =  ((RooTreeDataStore*) ((MCMCInterval*)mcmcint)->GetChainAsDataSet()->store())->tree();
  chain.SetMarkerStyle(6);
  chain.SetMarkerColor(kRed);
  chain.Draw("s:ratioSigEff:ratioBkgEff","weight_MarkovChain_local_","box"); // 3-d box proporional to posterior

  // the points used in the profile construction
  TTree& parameterScan =  ((RooTreeDataStore*) fc.GetPointsToScan()->store())->tree();
  parameterScan.SetMarkerStyle(24);
  parameterScan.Draw("s:ratioSigEff:ratioBkgEff","","same");


  delete wspace;
  delete lrint;
  if (fcint != NULL) delete fcint;
  delete data;

  /// print timing info
  t.Stop();
  t.Print();
}
// int main() { 
//    rs101_limitexample();
// }
