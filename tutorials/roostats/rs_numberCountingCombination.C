/// \file
/// \ingroup tutorial_roostats
/// \notebook -js
/// 'Number Counting Example' RooStats tutorial macro #100
///
/// This tutorial shows an example of a combination of
/// two searches using number counting with background uncertainty.
///
/// The macro uses a RooStats "factory" to construct a PDF
/// that represents the two number counting analyses with background
/// uncertainties.  The uncertainties are taken into account by
/// considering a sideband measurement of a size that corresponds to the
/// background uncertainty.  The problem has been studied in these references:
///  - http://arxiv.org/abs/physics/0511028
///  - http://arxiv.org/abs/physics/0702156
///  - http://cdsweb.cern.ch/record/1099969?ln=en
///
/// After using the factory to make the model, we use a RooStats
/// ProfileLikelihoodCalculator for a Hypothesis test and a confidence interval.
/// The calculator takes into account systematics by eliminating nuisance parameters
/// with the profile likelihood.  This is equivalent to the method of MINOS.
///
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Kyle Cranmer

#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/NumberCountingPdfFactory.h"
#include "RooStats/ConfInterval.h"
#include "RooStats/HypoTestResult.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooRealVar.h"

#include <cassert>

// use this order for safety on library loading
using namespace RooFit;
using namespace RooStats;

// declare three variations on the same tutorial
void rs_numberCountingCombination_expected();
void rs_numberCountingCombination_observed();
void rs_numberCountingCombination_observedWithTau();

// -------------------------------
// main driver to choose one
void rs_numberCountingCombination(int flag = 1)
{
   if (flag == 1)
      rs_numberCountingCombination_expected();
   if (flag == 2)
      rs_numberCountingCombination_observed();
   if (flag == 3)
      rs_numberCountingCombination_observedWithTau();
}

// -------------------------------
void rs_numberCountingCombination_expected()
{

   /////////////////////////////////////////
   // An example of a number counting combination with two channels.
   // We consider both hypothesis testing and the equivalent confidence interval.
   /////////////////////////////////////////

   /////////////////////////////////////////
   // The Model building stage
   /////////////////////////////////////////

   // Step 1, define arrays with signal & bkg expectations and background uncertainties
   double s[2] = {20., 10.};      // expected signal
   double b[2] = {100., 100.};    // expected background
   double db[2] = {.0100, .0100}; // fractional background uncertainty

   // Step 2, use a RooStats factory to build a PDF for a
   // number counting combination and add it to the workspace.
   // We need to give the signal expectation to relate the masterSignal
   // to the signal contribution in the individual channels.
   // The model neglects correlations in background uncertainty,
   // but they could be added without much change to the example.
   NumberCountingPdfFactory f;
   RooWorkspace wspace{};
   f.AddModel(s, 2, &wspace, "TopLevelPdf", "masterSignal");

   // Step 3, use a RooStats factory to add datasets to the workspace.
   // Step 3a.
   // Add the expected data to the workspace
   f.AddExpData(s, b, db, 2, &wspace, "ExpectedNumberCountingData");

   // see below for a printout of the workspace
   //  wspace.Print();  //uncomment to see structure of workspace

   /////////////////////////////////////////
   // The Hypothesis testing stage:
   /////////////////////////////////////////
   // Step 4, Define the null hypothesis for the calculator
   // Here you need to know the name of the variables corresponding to hypothesis.
   RooRealVar *mu = wspace.var("masterSignal");
   RooArgSet poi{*mu};
   RooArgSet nullParams{"nullParams"};
   nullParams.addClone(*mu);
   // here we explicitly set the value of the parameters for the null
   nullParams.setRealValue("masterSignal", 0);

   // Step 5, Create a calculator for doing the hypothesis test.
   // because this is a
   ProfileLikelihoodCalculator plc(*wspace.data("ExpectedNumberCountingData"), *wspace.pdf("TopLevelPdf"), poi, 0.05,
                                   &nullParams);

   // Step 6, Use the Calculator to get a HypoTestResult
   std::unique_ptr<HypoTestResult> htr{plc.GetHypoTest()};
   std::cout << "-------------------------------------------------" << std::endl;
   std::cout << "The p-value for the null is " << htr->NullPValue() << std::endl;
   std::cout << "Corresponding to a significance of " << htr->Significance() << std::endl;
   std::cout << "-------------------------------------------------\n\n" << std::endl;

   /* expected case should return:
      -------------------------------------------------
      The p-value for the null is 0.015294
      Corresponding to a significance of 2.16239
      -------------------------------------------------
   */

   //////////////////////////////////////////
   // Confidence Interval Stage

   // Step 8, Here we re-use the ProfileLikelihoodCalculator to return a confidence interval.
   // We need to specify what are our parameters of interest
   RooArgSet &paramsOfInterest = nullParams; // they are the same as before in this case
   plc.SetParameters(paramsOfInterest);
   std::unique_ptr<LikelihoodInterval> lrint{static_cast<LikelihoodInterval *>(plc.GetInterval())};
   lrint->SetConfidenceLevel(0.95);

   // Step 9, make a plot of the likelihood ratio and the interval obtained
   // paramsOfInterest.setRealValue("masterSignal",1.);
   // find limits
   double lower = lrint->LowerLimit(*mu);
   double upper = lrint->UpperLimit(*mu);

   LikelihoodIntervalPlot lrPlot(lrint.get());
   lrPlot.SetMaximum(3.);
   lrPlot.Draw();

   // Step 10a. Get upper and lower limits
   std::cout << "lower limit on master signal = " << lower << std::endl;
   std::cout << "upper limit on master signal = " << upper << std::endl;

   // Step 10b, Ask if masterSignal=0 is in the interval.
   // Note, this is equivalent to the question of a 2-sigma hypothesis test:
   // "is the parameter point masterSignal=0 inside the 95% confidence interval?"
   // Since the significance of the Hypothesis test was > 2-sigma it should not be:
   // eg. we exclude masterSignal=0 at 95% confidence.
   paramsOfInterest.setRealValue("masterSignal", 0.);
   std::cout << "-------------------------------------------------" << std::endl;
   std::cout << "Consider this parameter point:" << std::endl;
   paramsOfInterest.first()->Print();
   if (lrint->IsInInterval(paramsOfInterest))
      std::cout << "It IS in the interval." << std::endl;
   else
      std::cout << "It is NOT in the interval." << std::endl;
   std::cout << "-------------------------------------------------\n\n" << std::endl;

   // Step 10c, We also ask about the parameter point masterSignal=2, which is inside the interval.
   paramsOfInterest.setRealValue("masterSignal", 2.);
   std::cout << "-------------------------------------------------" << std::endl;
   std::cout << "Consider this parameter point:" << std::endl;
   paramsOfInterest.first()->Print();
   if (lrint->IsInInterval(paramsOfInterest))
      std::cout << "It IS in the interval." << std::endl;
   else
      std::cout << "It is NOT in the interval." << std::endl;
   std::cout << "-------------------------------------------------\n\n" << std::endl;

   /*
   // Here's an example of what is in the workspace
   //  wspace.Print();
   RooWorkspace(NumberCountingWS) Number Counting WS contents

   variables
   ---------
   (x_0,masterSignal,expected_s_0,b_0,y_0,tau_0,x_1,expected_s_1,b_1,y_1,tau_1)

   p.d.f.s
   -------
   RooProdPdf::joint[ pdfs=(sigRegion_0,sideband_0,sigRegion_1,sideband_1) ] = 2.20148e-08
   RooPoisson::sigRegion_0[ x=x_0 mean=splusb_0 ] = 0.036393
   RooPoisson::sideband_0[ x=y_0 mean=bTau_0 ] = 0.00398939
   RooPoisson::sigRegion_1[ x=x_1 mean=splusb_1 ] = 0.0380088
   RooPoisson::sideband_1[ x=y_1 mean=bTau_1 ] = 0.00398939

   functions
   --------
   RooAddition::splusb_0[ set1=(s_0,b_0) set2=() ] = 120
   RooProduct::s_0[ compRSet=(masterSignal,expected_s_0) compCSet=() ] = 20
   RooProduct::bTau_0[ compRSet=(b_0,tau_0) compCSet=() ] = 10000
   RooAddition::splusb_1[ set1=(s_1,b_1) set2=() ] = 110
   RooProduct::s_1[ compRSet=(masterSignal,expected_s_1) compCSet=() ] = 10
   RooProduct::bTau_1[ compRSet=(b_1,tau_1) compCSet=() ] = 10000

   datasets
   --------
   RooDataSet::ExpectedNumberCountingData(x_0,y_0,x_1,y_1)

   embedded pre-calculated expensive components
   -------------------------------------------
   */
}

void rs_numberCountingCombination_observed()
{

   /////////////////////////////////////////
   // The same example with observed data in a main
   // measurement and an background-only auxiliary
   // measurement with a factor tau more background
   // than in the main measurement.

   /////////////////////////////////////////
   // The Model building stage
   /////////////////////////////////////////

   // Step 1, define arrays with signal & bkg expectations and background uncertainties
   // We still need the expectation to relate signal in different channels with the master signal
   double s[2] = {20., 10.}; // expected signal

   // Step 2, use a RooStats factory to build a PDF for a
   // number counting combination and add it to the workspace.
   // We need to give the signal expectation to relate the masterSignal
   // to the signal contribution in the individual channels.
   // The model neglects correlations in background uncertainty,
   // but they could be added without much change to the example.
   NumberCountingPdfFactory f;
   RooWorkspace wspace{};
   f.AddModel(s, 2, &wspace, "TopLevelPdf", "masterSignal");

   // Step 3, use a RooStats factory to add datasets to the workspace.
   // Add the observed data to the workspace
   double mainMeas[2] = {123., 117.};   // observed main measurement
   double bkgMeas[2] = {111.23, 98.76}; // observed background
   double dbMeas[2] = {.011, .0095};    // observed fractional background uncertainty
   f.AddData(mainMeas, bkgMeas, dbMeas, 2, &wspace, "ObservedNumberCountingData");

   // see below for a printout of the workspace
   //  wspace.Print();  //uncomment to see structure of workspace

   /////////////////////////////////////////
   // The Hypothesis testing stage:
   /////////////////////////////////////////
   // Step 4, Define the null hypothesis for the calculator
   // Here you need to know the name of the variables corresponding to hypothesis.
   RooRealVar *mu = wspace.var("masterSignal");
   RooArgSet poi{*mu};
   RooArgSet nullParams{"nullParams"};
   nullParams.addClone(*mu);
   // here we explicitly set the value of the parameters for the null
   nullParams.setRealValue("masterSignal", 0);

   // Step 5, Create a calculator for doing the hypothesis test.
   // because this is a
   ProfileLikelihoodCalculator plc(*wspace.data("ObservedNumberCountingData"), *wspace.pdf("TopLevelPdf"), poi, 0.05,
                                   &nullParams);

   wspace.var("tau_0")->Print();
   wspace.var("tau_1")->Print();

   // Step 7, Use the Calculator to get a HypoTestResult
   std::unique_ptr<HypoTestResult> htr{plc.GetHypoTest()};
   std::cout << "-------------------------------------------------" << std::endl;
   std::cout << "The p-value for the null is " << htr->NullPValue() << std::endl;
   std::cout << "Corresponding to a significance of " << htr->Significance() << std::endl;
   std::cout << "-------------------------------------------------\n\n" << std::endl;

   /* observed case should return:
      -------------------------------------------------
      The p-value for the null is 0.0351669
      Corresponding to a significance of 1.80975
      -------------------------------------------------
   */

   //////////////////////////////////////////
   // Confidence Interval Stage

   // Step 8, Here we re-use the ProfileLikelihoodCalculator to return a confidence interval.
   // We need to specify what are our parameters of interest
   RooArgSet &paramsOfInterest = nullParams; // they are the same as before in this case
   plc.SetParameters(paramsOfInterest);
   std::unique_ptr<LikelihoodInterval> lrint{static_cast<LikelihoodInterval *>(plc.GetInterval())};
   lrint->SetConfidenceLevel(0.95);

   // Step 9c. Get upper and lower limits
   std::cout << "lower limit on master signal = " << lrint->LowerLimit(*mu) << std::endl;
   std::cout << "upper limit on master signal = " << lrint->UpperLimit(*mu) << std::endl;
}

void rs_numberCountingCombination_observedWithTau()
{

   /////////////////////////////////////////
   // The same example with observed data in a main
   // measurement and an background-only auxiliary
   // measurement with a factor tau more background
   // than in the main measurement.

   /////////////////////////////////////////
   // The Model building stage
   /////////////////////////////////////////

   // Step 1, define arrays with signal & bkg expectations and background uncertainties
   // We still need the expectation to relate signal in different channels with the master signal
   double s[2] = {20., 10.}; // expected signal

   // Step 2, use a RooStats factory to build a PDF for a
   // number counting combination and add it to the workspace.
   // We need to give the signal expectation to relate the masterSignal
   // to the signal contribution in the individual channels.
   // The model neglects correlations in background uncertainty,
   // but they could be added without much change to the example.
   NumberCountingPdfFactory f;
   RooWorkspace wspace{};
   f.AddModel(s, 2, &wspace, "TopLevelPdf", "masterSignal");

   // Step 3, use a RooStats factory to add datasets to the workspace.
   // Add the observed data to the workspace in the on-off problem.
   double mainMeas[2] = {123., 117.};    // observed main measurement
   double sideband[2] = {11123., 9876.}; // observed sideband
   double tau[2] = {100., 100.}; // ratio of bkg in sideband to bkg in main measurement, from experimental design.
   f.AddDataWithSideband(mainMeas, sideband, tau, 2, &wspace, "ObservedNumberCountingDataWithSideband");

   // see below for a printout of the workspace
   //  wspace.Print();  //uncomment to see structure of workspace

   /////////////////////////////////////////
   // The Hypothesis testing stage:
   /////////////////////////////////////////
   // Step 4, Define the null hypothesis for the calculator
   // Here you need to know the name of the variables corresponding to hypothesis.
   RooRealVar *mu = wspace.var("masterSignal");
   RooArgSet poi{*mu};
   RooArgSet nullParams{"nullParams"};
   nullParams.addClone(*mu);
   // here we explicitly set the value of the parameters for the null
   nullParams.setRealValue("masterSignal", 0);

   // Step 5, Create a calculator for doing the hypothesis test.
   // because this is a
   ProfileLikelihoodCalculator plc(*wspace.data("ObservedNumberCountingDataWithSideband"), *wspace.pdf("TopLevelPdf"),
                                   poi, 0.05, &nullParams);

   // Step 7, Use the Calculator to get a HypoTestResult
   std::unique_ptr<HypoTestResult> htr{plc.GetHypoTest()};
   std::cout << "-------------------------------------------------" << std::endl;
   std::cout << "The p-value for the null is " << htr->NullPValue() << std::endl;
   std::cout << "Corresponding to a significance of " << htr->Significance() << std::endl;
   std::cout << "-------------------------------------------------\n\n" << std::endl;

   /* observed case should return:
      -------------------------------------------------
      The p-value for the null is 0.0352035
      Corresponding to a significance of 1.80928
      -------------------------------------------------
   */

   //////////////////////////////////////////
   // Confidence Interval Stage

   // Step 8, Here we re-use the ProfileLikelihoodCalculator to return a confidence interval.
   // We need to specify what are our parameters of interest
   RooArgSet &paramsOfInterest = nullParams; // they are the same as before in this case
   plc.SetParameters(paramsOfInterest);
   std::unique_ptr<LikelihoodInterval> lrint{static_cast<LikelihoodInterval *>(plc.GetInterval())};
   lrint->SetConfidenceLevel(0.95);

   // Step 9c. Get upper and lower limits
   std::cout << "lower limit on master signal = " << lrint->LowerLimit(*mu) << std::endl;
   std::cout << "upper limit on master signal = " << lrint->UpperLimit(*mu) << std::endl;
}
