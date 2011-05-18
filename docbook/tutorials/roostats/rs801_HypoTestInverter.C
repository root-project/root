/////////////////////////////////////////////////////////////////////////
//
// 'Hypothesis Test Inversion' RooStats tutorial macro #801
// author: Gregory Schott
// date Sep 2009
//
// This tutorial shows an example of using the HypoTestInverter class 
//
/////////////////////////////////////////////////////////////////////////

#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "RooPolynomial.h"
#include "RooAddPdf.h"
#include "RooExtendPdf.h"

#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/HybridCalculatorOriginal.h"

#include "TGraphErrors.h"

using namespace RooFit;
using namespace RooStats;


void rs801_HypoTestInverter()
{
  // prepare the model
  RooRealVar lumi("lumi","luminosity",1);
  RooRealVar r("r","cross-section ratio",3.74,0,50);
  RooFormulaVar ns("ns","1*r*lumi",RooArgList(lumi,r));
  RooRealVar nb("nb","background yield",1);
  RooRealVar x("x","dummy observable",0,1);
  RooConstVar p0(RooFit::RooConst(0));
  RooPolynomial flatPdf("flatPdf","flat PDF",x,p0);
  RooAddPdf totPdf("totPdf","S+B model",RooArgList(flatPdf,flatPdf),RooArgList(ns,nb));
  RooExtendPdf bkgPdf("bkgPdf","B-only model",flatPdf,nb);
  RooDataSet* data = totPdf.generate(x,1);

  // prepare the calculator
  HybridCalculatorOriginal myhc(*data, totPdf, bkgPdf,0,0);
  myhc.SetTestStatistic(2);
  myhc.SetNumberOfToys(1000);
  myhc.UseNuisance(false);                            

  // run the hypothesis-test invertion
  HypoTestInverter myInverter(myhc,r);
  myInverter.SetTestSize(0.10);
  myInverter.UseCLs(true);
  // myInverter.RunFixedScan(5,1,6);
  // scan for a 95% UL
  myInverter.RunAutoScan(3.,5,myInverter.Size()/2,0.005);  
  // run an alternative autoscan algorithm 
  // myInverter.RunAutoScan(1,6,myInverter.Size()/2,0.005,1);  
  //myInverter.RunOnePoint(3.9);


  HypoTestInverterResult* results = myInverter.GetInterval();

  HypoTestInverterPlot myInverterPlot("myInverterPlot","",results);
  TGraphErrors* gr1 = myInverterPlot.MakePlot();
  gr1->Draw("ALP");

  double ulError = results->UpperLimitEstimatedError();

  double upperLimit = results->UpperLimit();
  std::cout << "The computed upper limit is: " << upperLimit << std::endl;
  std::cout << "an estimated error on this upper limit is: " << ulError << std::endl;
  // expected result: 4.10
}
int main() { 
   rs801_HypoTestInverter();
}
