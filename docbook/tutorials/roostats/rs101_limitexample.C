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
#include "RooStats/ModelConfig.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/MCMCIntervalPlot.h"
#include "RooStats/ProposalFunction.h"
#include "RooStats/ProposalHelper.h"
#include "RooFitResult.h"


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
  //  wspace->factory("Gaussian::sigConstraint(ratioSigEff,1,0.05)"); // 5% signal efficiency uncertainty
  //  wspace->factory("Gaussian::bkgConstraint(ratioBkgEff,1,0.1)"); // 10% background efficiency uncertainty
  wspace->factory("Gaussian::sigConstraint(1,ratioSigEff,0.05)"); // 5% signal efficiency uncertainty
  wspace->factory("Gaussian::bkgConstraint(1,ratioBkgEff,0.1)"); // 10% background efficiency uncertainty
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

  ModelConfig modelConfig(new RooWorkspace());
  modelConfig.SetPdf(*modelWithConstraints);
  modelConfig.SetParametersOfInterest(paramOfInterest);


  // First, let's use a Calculator based on the Profile Likelihood Ratio
  //ProfileLikelihoodCalculator plc(*data, *modelWithConstraints, paramOfInterest); 
  ProfileLikelihoodCalculator plc(*data, modelConfig); 
  plc.SetTestSize(.05);
  ConfInterval* lrint = plc.GetInterval();  // that was easy.

  // Let's make a plot
  TCanvas* dataCanvas = new TCanvas("dataCanvas");
  dataCanvas->Divide(2,1);

  dataCanvas->cd(1);
  LikelihoodIntervalPlot plotInt((LikelihoodInterval*)lrint);
  plotInt.SetTitle("Profile Likelihood Ratio and Posterior for S");
  plotInt.Draw();

  // Second, use a Calculator based on the Feldman Cousins technique
  FeldmanCousins fc(*data, modelConfig);
  fc.UseAdaptiveSampling(true);
  fc.FluctuateNumDataEntries(false); // number counting analysis: dataset always has 1 entry with N events observed
  fc.SetNBins(100); // number of points to test per parameter
  fc.SetTestSize(.05);
  //  fc.SaveBeltToFile(true); // optional
  ConfInterval* fcint = NULL;
  fcint = fc.GetInterval();  // that was easy.

  RooFitResult* fit = modelWithConstraints->fitTo(*data, Save(true));

  // Third, use a Calculator based on Markov Chain monte carlo
  // Before configuring the calculator, let's make a ProposalFunction
  // that will achieve a high acceptance rate
  ProposalHelper ph;
  ph.SetVariables((RooArgSet&)fit->floatParsFinal());
  ph.SetCovMatrix(fit->covarianceMatrix());
  ph.SetUpdateProposalParameters(true);
  ph.SetCacheSize(100);
  ProposalFunction* pdfProp = ph.GetProposalFunction();  // that was easy

  MCMCCalculator mc(*data, modelConfig);
  mc.SetNumIters(20000); // steps to propose in the chain
  mc.SetTestSize(.05); // 95% CL
  mc.SetNumBurnInSteps(40); // ignore first N steps in chain as "burn in"
  mc.SetProposalFunction(*pdfProp);
  mc.SetLeftSideTailFraction(0.5);  // find a "central" interval
  MCMCInterval* mcInt = (MCMCInterval*)mc.GetInterval();  // that was easy


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

  // Plot MCMC interval and print some statistics
  MCMCIntervalPlot mcPlot(*mcInt);
  mcPlot.SetLineColor(kMagenta);
  mcPlot.SetLineWidth(2);
  mcPlot.Draw("same");

  double mcul = mcInt->UpperLimit(*s);
  double mcll = mcInt->LowerLimit(*s);
  cout << "MCMC lower limit on s = " << mcll << endl;
  cout << "MCMC upper limit on s = " << mcul << endl;
  cout << "MCMC Actual confidence level: "
     << mcInt->GetActualConfidenceLevel() << endl;

  // 3-d plot of the parameter points
  dataCanvas->cd(2);
  // also plot the points in the markov chain
  TTree& chain =  ((RooTreeDataStore*) mcInt->GetChainAsDataSet()->store())->tree();
  chain.SetMarkerStyle(6);
  chain.SetMarkerColor(kRed);
  chain.Draw("s:ratioSigEff:ratioBkgEff","weight_MarkovChain_local_","box"); // 3-d box proporional to posterior

  // the points used in the profile construction
  TTree& parameterScan =  ((RooTreeDataStore*) fc.GetPointsToScan()->store())->tree();
  parameterScan.SetMarkerStyle(24);
  parameterScan.Draw("s:ratioSigEff:ratioBkgEff","","same");

  delete wspace;
  delete lrint;
  delete mcInt;
  delete fcint;
  delete data;

  /// print timing info
  t.Stop();
  t.Print();
}
