/////////////////////////////////////////////////////////////////////////
//
// 'Hypothesis Test Inversion' RooStats tutorial macro #801
// author: Gregory Schott
// date Sep 2009
//
// This tutorial shows an example of using the HypoTestInvertor class 
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

#include "RooStats/HypoTestInvertor.h"
#include "RooStats/HypoTestInvertorResult.h"
#include "RooStats/HypoTestInvertorPlot.h"
#include "RooStats/HybridCalculator.h"


using namespace RooFit;
using namespace RooStats;


void rs801_HypoTestInvertor()
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
  HybridCalculator myhc("myhc","",*data, totPdf, bkgPdf,0,0);
  myhc.SetTestStatistics(2);
  myhc.SetNumberOfToys(1000);
  myhc.UseNuisance(false);                            

  // run the hypothesis-test invertion
  HypoTestInvertor myInvertor("myInvertor","",&myhc,&r);
  myInvertor.SetTestSize(0.05);
  // myInvertor.RunFixedScan(5,1,6);
  myInvertor.RunAutoScan(1,6,0.005);
  myInvertor.RunOnePoint(3.9);

  HypoTestInvertorResult* results = myInvertor.GetInterval();
  HypoTestInvertorPlot myInvertorPlot("myInvertorPlot","",results);
  TGraph* gr1 = myInvertorPlot.MakePlot();
  gr1->Draw("ALP*");

  std::cout << "The computed upper limit is: " << results->UpperLimit() << std::endl;
  // expected result: 4.10
}
