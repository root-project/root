/// \file
/// \ingroup tutorial_roostats
/// \notebook
/// Limits: number counting experiment with uncertainty on both the background rate and signal efficiency.
///
/// The usage of a Confidence Interval Calculator to set a limit on the signal is illustrated
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Kyle Cranmer

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
#include "TGraph2D.h"

#include <cassert>

// use this order for safety on library loading
using namespace RooFit;
using namespace RooStats;

void rs101_limitexample()
{
   // --------------------------------------
   // An example of setting a limit in a number counting experiment with uncertainty on background and signal

   // to time the macro
   TStopwatch t;
   t.Start();

   // --------------------------------------
   // The Model building stage
   // --------------------------------------
   RooWorkspace wspace{};
   wspace.factory("Poisson::countingModel(obs[150,0,300], "
                   "sum(s[50,0,120]*ratioSigEff[1.,0,3.],b[100]*ratioBkgEff[1.,0.,3.]))"); // counting model
   //  wspace.factory("Gaussian::sigConstraint(ratioSigEff,1,0.05)"); // 5% signal efficiency uncertainty
   //  wspace.factory("Gaussian::bkgConstraint(ratioBkgEff,1,0.1)"); // 10% background efficiency uncertainty
   wspace.factory("Gaussian::sigConstraint(gSigEff[1,0,3],ratioSigEff,0.05)"); // 5% signal efficiency uncertainty
   wspace.factory("Gaussian::bkgConstraint(gSigBkg[1,0,3],ratioBkgEff,0.2)");  // 10% background efficiency uncertainty
   wspace.factory("PROD::modelWithConstraints(countingModel,sigConstraint,bkgConstraint)"); // product of terms
   wspace.Print();

   RooAbsPdf *modelWithConstraints = wspace.pdf("modelWithConstraints"); // get the model
   RooRealVar *obs = wspace.var("obs");                                  // get the observable
   RooRealVar *s = wspace.var("s");                                      // get the signal we care about
   RooRealVar *b =
      wspace.var("b"); // get the background and set it to a constant.  Uncertainty included in ratioBkgEff
   b->setConstant();

   RooRealVar *ratioSigEff = wspace.var("ratioSigEff"); // get uncertain parameter to constrain
   RooRealVar *ratioBkgEff = wspace.var("ratioBkgEff"); // get uncertain parameter to constrain
   RooArgSet constrainedParams(*ratioSigEff,
                               *ratioBkgEff); // need to constrain these in the fit (should change default behavior)

   RooRealVar *gSigEff = wspace.var("gSigEff"); // global observables for signal efficiency
   RooRealVar *gSigBkg = wspace.var("gSigBkg"); // global obs for background efficiency
   gSigEff->setConstant();
   gSigBkg->setConstant();

   // Create an example dataset with 160 observed events
   obs->setVal(160.);
   RooDataSet dataOrig{"exampleData", "exampleData", *obs};
   dataOrig.add(*obs);

   RooArgSet all(*s, *ratioBkgEff, *ratioSigEff);

   // not necessary
   modelWithConstraints->fitTo(dataOrig, Constrain({*ratioSigEff, *ratioBkgEff}), PrintLevel(-1));

   // Now let's make some confidence intervals for s, our parameter of interest
   RooArgSet paramOfInterest(*s);

   ModelConfig modelConfig(&wspace);
   modelConfig.SetPdf(*modelWithConstraints);
   modelConfig.SetParametersOfInterest(paramOfInterest);
   modelConfig.SetNuisanceParameters(constrainedParams);
   modelConfig.SetObservables(*obs);
   modelConfig.SetGlobalObservables(RooArgSet(*gSigEff, *gSigBkg));
   modelConfig.SetName("ModelConfig");
   wspace.import(modelConfig);
   wspace.import(dataOrig);
   wspace.SetName("w");
   // wspace.writeToFile("rs101_ws.root");

   // Make sure we reference the data in the workspace from now on
   RooDataSet &data = static_cast<RooDataSet &>(*wspace.data(dataOrig.GetName()));

   // First, let's use a Calculator based on the Profile Likelihood Ratio
   // ProfileLikelihoodCalculator plc(data, *modelWithConstraints, paramOfInterest);
   ProfileLikelihoodCalculator plc(data, modelConfig);
   plc.SetTestSize(.05);
   std::unique_ptr<LikelihoodInterval> lrinterval{static_cast<LikelihoodInterval*>(plc.GetInterval())};

   // Let's make a plot
   auto dataCanvas = new TCanvas("dataCanvas");
   dataCanvas->Divide(2, 1);

   dataCanvas->cd(1);
   LikelihoodIntervalPlot plotInt(lrinterval.get());
   plotInt.SetTitle("Profile Likelihood Ratio and Posterior for S");
   plotInt.Draw();

   // Second, use a Calculator based on the Feldman Cousins technique
   FeldmanCousins fc(data, modelConfig);
   fc.UseAdaptiveSampling(true);
   fc.FluctuateNumDataEntries(false); // number counting analysis: dataset always has 1 entry with N events observed
   fc.SetNBins(100);                  // number of points to test per parameter
   fc.SetTestSize(.05);
   //  fc.SaveBeltToFile(true); // optional
   std::unique_ptr<PointSetInterval> fcint{static_cast<PointSetInterval*>(fc.GetInterval())};

   std::unique_ptr<RooFitResult> fit{modelWithConstraints->fitTo(data, Save(true), PrintLevel(-1))};

   // Third, use a Calculator based on Markov Chain monte carlo
   // Before configuring the calculator, let's make a ProposalFunction
   // that will achieve a high acceptance rate
   ProposalHelper ph;
   ph.SetVariables((RooArgSet &)fit->floatParsFinal());
   ph.SetCovMatrix(fit->covarianceMatrix());
   ph.SetUpdateProposalParameters(true);
   ph.SetCacheSize(100);
   ProposalFunction *pdfProp = ph.GetProposalFunction();

   MCMCCalculator mc(data, modelConfig);
   mc.SetNumIters(20000);    // steps to propose in the chain
   mc.SetTestSize(.05);      // 95% CL
   mc.SetNumBurnInSteps(40); // ignore first N steps in chain as "burn in"
   mc.SetProposalFunction(*pdfProp);
   mc.SetLeftSideTailFraction(0.5);                        // find a "central" interval
   std::unique_ptr<MCMCInterval> mcInt{static_cast<MCMCInterval *>(mc.GetInterval())};

   // Get Lower and Upper limits from Profile Calculator
   std::cout << "Profile lower limit on s = " << lrinterval->LowerLimit(*s) << std::endl;
   std::cout << "Profile upper limit on s = " << lrinterval->UpperLimit(*s) << std::endl;

   // Get Lower and Upper limits from FeldmanCousins with profile construction
   if (fcint) {
      double fcul = fcint->UpperLimit(*s);
      double fcll = fcint->LowerLimit(*s);
      std::cout << "FC lower limit on s = " << fcll << std::endl;
      std::cout << "FC upper limit on s = " << fcul << std::endl;
      auto fcllLine = new TLine(fcll, 0, fcll, 1);
      auto fculLine = new TLine(fcul, 0, fcul, 1);
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
   std::cout << "MCMC lower limit on s = " << mcll << std::endl;
   std::cout << "MCMC upper limit on s = " << mcul << std::endl;
   std::cout << "MCMC Actual confidence level: " << mcInt->GetActualConfidenceLevel() << std::endl;

   // 3-d plot of the parameter points
   dataCanvas->cd(2);
   // also plot the points in the markov chain
   std::unique_ptr<RooDataSet> chainData{mcInt->GetChainAsDataSet()};

   assert(chainData);
   std::cout << "plotting the chain data - nentries = " << chainData->numEntries() << std::endl;
   TTree *chain = RooStats::GetAsTTree("chainTreeData", "chainTreeData", *chainData);
   assert(chain);
   chain->SetMarkerStyle(6);
   chain->SetMarkerColor(kRed);

   chain->Draw("s:ratioSigEff:ratioBkgEff", "nll_MarkovChain_local_", "box"); // 3-d box proportional to posterior

   // the points used in the profile construction
   RooDataSet *parScanData = (RooDataSet *)fc.GetPointsToScan();
   assert(parScanData);
   std::cout << "plotting the scanned points used in the frequentist construction - npoints = "
             << parScanData->numEntries() << std::endl;
   // getting the tree and drawing it -crashes (very strange....);
   // TTree* parameterScan =  RooStats::GetAsTTree("parScanTreeData","parScanTreeData",*parScanData);
   // assert(parameterScan);
   // parameterScan->Draw("s:ratioSigEff:ratioBkgEff","","goff");
   auto gr = new TGraph2D(parScanData->numEntries());
   for (int ievt = 0; ievt < parScanData->numEntries(); ++ievt) {
      const RooArgSet *evt = parScanData->get(ievt);
      double x = evt->getRealValue("ratioBkgEff");
      double y = evt->getRealValue("ratioSigEff");
      double z = evt->getRealValue("s");
      gr->SetPoint(ievt, x, y, z);
      // std::cout << ievt << "  " << x << "  " << y << "  " << z << std::endl;
   }
   gr->SetMarkerStyle(24);
   gr->Draw("P SAME");

   // print timing info
   t.Stop();
   t.Print();
}
