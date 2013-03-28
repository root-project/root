/////////////////////////////////////////////////////////////////////////
//
// 'Bernstein Correction' RooStats tutorial macro
// author: Kyle Cranmer
// date March. 2009
//
// This tutorial shows usage of a the BernsteinCorrection utility in RooStats.  
// The idea is that one has a distribution coming either from data or Monte Carlo 
// (called "reality" in the macro) and a nominal model that is not sufficiently 
// flexible to take into account the real distribution.  One wants to take into
// account the systematic associated with this imperfect modeling by augmenting
// the nominal model with some correction term (in this case a polynomial).
// The BernsteinCorrection utility will import into your workspace a corrected model
// given by nominal(x) * poly_N(x), where poly_N is an n-th order polynomial in
// the Bernstein basis.  The degree N of the polynomial is chosen by specifying the tolerance
// one has in adding an extra term to the polynomial.  
// The Bernstein basis is nice because it only has positive-definite terms
// and works well with PDFs.  
// Finally, the macro makes a plot of:
//  - the data (drawn from 'reality'), 
//  - the best fit of the nominal model (blue)
//  - and the best fit corrected model.
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooBernstein.h"
#include "TCanvas.h"
#include "RooAbsPdf.h"
#include "RooFit.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include <string>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <iostream>

#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooGaussian.h"
#include "RooNLLVar.h"
#include "RooMinuit.h"
#include "RooProfileLL.h"
#include "RooWorkspace.h"

#include "RooStats/BernsteinCorrection.h"

// use this order for safety on library loading
using namespace RooFit;
using namespace RooStats;


//____________________________________
void rs_bernsteinCorrection(){

  // set range of observable
  Double_t lowRange = -1, highRange =5;

  // make a RooRealVar for the observable
  RooRealVar x("x", "x", lowRange, highRange);

  // true model
  RooGaussian narrow("narrow","",x,RooConst(0.), RooConst(.8));
  RooGaussian wide("wide","",x,RooConst(0.), RooConst(2.));
  RooAddPdf reality("reality","",RooArgList(narrow, wide), RooConst(0.8));

  RooDataSet* data = reality.generate(x,1000);

  // nominal model
  RooRealVar sigma("sigma","",1.,0,10);
  RooGaussian nominal("nominal","",x,RooConst(0.), sigma);

  RooWorkspace* wks = new RooWorkspace("myWorksspace");

  wks->import(*data, Rename("data"));
  wks->import(nominal);

  // The tolerance sets the probability to add an unnecessary term.
  // lower tolerance will add fewer terms, while higher tolerance
  // will add more terms and provide a more flexible function.
  Double_t tolerance = 0.05; 
  BernsteinCorrection bernsteinCorrection(tolerance);
  Int_t degree = bernsteinCorrection.ImportCorrectedPdf(wks,"nominal","x","data");

  cout << " Correction based on Bernstein Poly of degree " << degree << endl;

  RooPlot* frame = x.frame();
  data->plotOn(frame);
  // plot the best fit nominal model in blue
  nominal.fitTo(*data,PrintLevel(-1));
  nominal.plotOn(frame);

  // plot the best fit corrected model in red
  RooAbsPdf* corrected = wks->pdf("corrected");  
  corrected->fitTo(*data,PrintLevel(-1));
  corrected->plotOn(frame,LineColor(kRed));

  // plot the correction term (* norm constant) in dashed green
  // should make norm constant just be 1, not depend on binning of data
  RooAbsPdf* poly = wks->pdf("poly");  
  poly->plotOn(frame,LineColor(kGreen), LineStyle(kDashed));
  
  // this is a switch to check the sampling distribution
  // of -2 log LR for two comparisons:
  // the first is for n-1 vs. n degree polynomial corrections
  // the second is for n vs. n+1 degree polynomial corrections
  // Here we choose n to be the one chosen by the tolerance
  // critereon above, eg. n = "degree" in the code.
  // Setting this to true is takes about 10 min.
  bool checkSamplingDist = false;

  TCanvas* c1 = new TCanvas();
  if(checkSamplingDist) {
    c1->Divide(1,2);
    c1->cd(1);
  }
  frame->Draw();

  if(checkSamplingDist) {
    // check sampling dist
    TH1F* samplingDist = new TH1F("samplingDist","",20,0,10);
    TH1F* samplingDistExtra = new TH1F("samplingDistExtra","",20,0,10);
    int numToyMC = 1000;
    bernsteinCorrection.CreateQSamplingDist(wks,"nominal","x","data",samplingDist, samplingDistExtra, degree,numToyMC);
    
    c1->cd(2);
    samplingDistExtra->SetLineColor(kRed);
    samplingDistExtra->Draw();
    samplingDist->Draw("same");
  }
}

