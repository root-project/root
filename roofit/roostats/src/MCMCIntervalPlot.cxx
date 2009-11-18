// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
This class provides simple and straightforward utilities to plot a MCMCInterval
object.  Basic use only requires a few lines once you have an MCMCInterval*:
</p>
<p>
MCMCIntervalPlot plot(*interval);
plot.Draw();
</p>
<p>
(The standard Draw() function will currently draw the confidence interval
range with bars if 1-D and a contour if 2-D.  The MCMC posterior will also be
plotted for the 1-D case.  Many other fun plotting options are available.
</p>
END_HTML
*/
//_________________________________________________

#ifndef ROOSTATS_MCMCIntervalPlot
#include "RooStats/MCMCIntervalPlot.h"
#endif
#include <iostream>
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TLine
#include "TLine.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TGraph
#include "TGraph.h"
#endif
#ifndef ROOT_TPad
#include "TPad.h"
#endif
#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif
#ifndef ROO_PLOT
#include "RooPlot.h"
#endif
#ifndef ROOT_TH2
#include "TH2.h"
#endif
#ifndef ROO_ARG_LIST
#include "RooArgList.h"
#endif
#ifndef ROOT_TAxis
#include "TAxis.h"
#endif

ClassImp(RooStats::MCMCIntervalPlot);

using namespace std;
using namespace RooStats;

MCMCIntervalPlot::MCMCIntervalPlot()
{
   fInterval = NULL;
   fParameters = NULL;
   fPosteriorHist = NULL;
   fPosteriorKeysPdf = NULL;
   fPosteriorKeysProduct = NULL;
   fDimension = 0;
   fLineColor = kBlack;
   fLineWidth = 1;
   //fContourColor = kBlack;
   fShowBurnIn = kTRUE;
   fWalk = NULL;
   fBurnIn = NULL;
   fFirst = NULL;
   fParamGraph = NULL;
   fNLLGraph = NULL;
   fNLLHist = NULL;
   fWeightHist = NULL;
}

MCMCIntervalPlot::MCMCIntervalPlot(MCMCInterval& interval)
{
   SetMCMCInterval(interval);
   fPosteriorHist = NULL;
   fPosteriorKeysPdf = NULL;
   fPosteriorKeysProduct = NULL;
   fLineColor = kBlack;
   fLineWidth = 1;
   //fContourColor = kBlack;
   fShowBurnIn = kTRUE;
   fWalk = NULL;
   fBurnIn = NULL;
   fFirst = NULL;
   fParamGraph = NULL;
   fNLLGraph = NULL;
   fNLLHist = NULL;
   fWeightHist = NULL;
}

MCMCIntervalPlot::~MCMCIntervalPlot()
{
   delete fParameters;
   // kbelasco: why does deleting fPosteriorHist remove the graphics
   // but deleting TGraphs doesn't?
   //delete fPosteriorHist;
   // can we delete fNLLHist and fWeightHist?
   //delete fNLLHist;
   //delete fWeightHist;

   // kbelasco: should we delete fPosteriorKeysPdf and fPosteriorKeysProduct?
   delete fPosteriorKeysPdf;
   delete fPosteriorKeysProduct;

   delete fWalk;
   delete fBurnIn;
   delete fFirst;
   delete fParamGraph;
   delete fNLLGraph;
}

void MCMCIntervalPlot::SetMCMCInterval(MCMCInterval& interval)
{
   fInterval = &interval;
   fDimension = fInterval->GetDimension();
   fParameters = fInterval->GetParameters();
}

void MCMCIntervalPlot::Draw(const Option_t* options)
{
   DrawInterval(options);
}

void MCMCIntervalPlot::DrawPosterior(const Option_t* options)
{
   if (fInterval->GetUseKeys())
      DrawPosteriorKeysPdf(options);
   else
      DrawPosteriorHist(options);
}

void MCMCIntervalPlot::DrawPosteriorHist(const Option_t* options, Bool_t scale)
{
   if (fPosteriorHist == NULL)
      fPosteriorHist = fInterval->GetPosteriorHist();

   // scale so highest bin has height 1
   if (scale)
      fPosteriorHist->Scale(1/fPosteriorHist->GetBinContent(fPosteriorHist->GetMaximumBin()));
   fPosteriorHist->Draw(options);
}

void MCMCIntervalPlot::DrawPosteriorKeysPdf(const Option_t* options)
{
   if (fPosteriorKeysPdf == NULL)
      fPosteriorKeysPdf = fInterval->GetPosteriorKeysPdf();

   if (fDimension == 1) {
      RooPlot* frame = ((RooRealVar*)fParameters->first())->frame();
      fPosteriorKeysPdf->plotOn(frame);
      frame->Draw(options);
   } else if (fDimension == 2) {
      RooArgList* axes = fInterval->GetAxes();
      RooRealVar* xVar = (RooRealVar*)axes->at(0);
      RooRealVar* yVar = (RooRealVar*)axes->at(1);
      TH2F* keysHist = (TH2F*)fPosteriorKeysPdf->createHistogram(
            "keysPlot2D", *xVar, RooFit::YVar(*yVar), RooFit::Scaling(kFALSE));
      keysHist->SetTitle(GetTitle());
      keysHist->Draw(options);
      delete axes;
   }
}

void MCMCIntervalPlot::DrawInterval(const Option_t* options)
{
   if (fInterval->GetUseKeys())
      DrawKeysPdfInterval(options);
   else
      DrawHistInterval(options);
}

void MCMCIntervalPlot::DrawKeysPdfInterval(const Option_t* options)
{
   if (fPosteriorKeysPdf == NULL)
      fPosteriorKeysPdf = fInterval->GetPosteriorKeysPdf();

   if (fDimension == 1) {
      RooRealVar* p = (RooRealVar*)fParameters->first();
      Double_t ul = fInterval->UpperLimitByKeys(*p);
      Double_t ll = fInterval->LowerLimitByKeys(*p);
      cout << "MCMC keys pdf lower limit on " << p->GetName() << " = "
           << ll << endl;
      cout << "MCMC keys pdf upper limit on " << p->GetName() << " = "
           << ul << endl;
      TLine* llLine = new TLine(ll, 0, ll, 1);
      TLine* ulLine = new TLine(ul, 0, ul, 1);
      llLine->SetLineColor(fLineColor);
      ulLine->SetLineColor(fLineColor);
      llLine->SetLineWidth(fLineWidth);
      ulLine->SetLineWidth(fLineWidth);
      llLine->Draw(options);
      ulLine->Draw(options);
   } else if (fDimension == 2) {
      RooArgList* axes = fInterval->GetAxes();
      RooRealVar* xVar = (RooRealVar*)axes->at(0);
      RooRealVar* yVar = (RooRealVar*)axes->at(1);
      TH2F* contHist = (TH2F*)fPosteriorKeysPdf->createHistogram(
            "keysContour2D", *xVar, RooFit::YVar(*yVar), RooFit::Scaling(kFALSE));
      contHist->SetTitle(GetTitle());
      contHist->SetStats(kFALSE);

      TString tmpOpt(options);
      if (!tmpOpt.Contains("CONT2")) tmpOpt.Append("CONT2");

      Double_t cutoff = fInterval->GetKeysPdfCutoff();
      contHist->SetContour(1, &cutoff);
      contHist->SetLineColor(fLineColor);
      contHist->SetLineWidth(fLineWidth);
      contHist->Draw(tmpOpt.Data());
      delete axes;
   } else {
      coutE(InputArguments) << "MCMCIntervalPlot::DrawKeysPdfInterval: "
         << " Sorry: " << fDimension << "-D plots not currently supported" << endl;
   }
}

void MCMCIntervalPlot::DrawHistInterval(const Option_t* options)
{
   if (fPosteriorHist == NULL)
      fPosteriorHist = fInterval->GetPosteriorHist();

   if (fDimension == 1) {
      // draw lower and upper limits
      RooRealVar* p = (RooRealVar*)fParameters->first();
      Double_t ul = fInterval->UpperLimitByHist(*p);
      Double_t ll = fInterval->LowerLimitByHist(*p);
      cout << "MCMC histogram lower limit on " << p->GetName() << " = "
           << ll << endl;
      cout << "MCMC histogram upper limit on " << p->GetName() << " = "
           << ul << endl;
      TLine* llLine = new TLine(ll, 0, ll, 1);
      TLine* ulLine = new TLine(ul, 0, ul, 1);
      llLine->SetLineColor(fLineColor);
      ulLine->SetLineColor(fLineColor);
      llLine->SetLineWidth(fLineWidth);
      ulLine->SetLineWidth(fLineWidth);
      llLine->Draw(options);
      ulLine->Draw(options);
   } else if (fDimension == 2) {
      fPosteriorHist->SetTitle(GetTitle());
      fPosteriorHist->SetStats(kFALSE);

      TString tmpOpt(options);
      if (!tmpOpt.Contains("CONT2")) tmpOpt.Append("CONT2");

      Double_t cutoff = fInterval->GetHistCutoff();
      fPosteriorHist->SetContour(1, &cutoff);
      fPosteriorHist->SetLineColor(fLineColor);
      fPosteriorHist->SetLineWidth(fLineWidth);
      fPosteriorHist->Draw(tmpOpt.Data());
   } else {
      coutE(InputArguments) << "MCMCIntervalPlot::DrawHistInterval: "
         << " Sorry: " << fDimension << "-D plots not currently supported" << endl;
   }
}

void MCMCIntervalPlot::DrawPosteriorKeysProduct(const Option_t* options)
{
   if (fPosteriorKeysProduct == NULL)
      fPosteriorKeysProduct = fInterval->GetPosteriorKeysProduct();

   if (fDimension == 1) {
      RooPlot* frame = ((RooRealVar*)fParameters->first())->frame();
      fPosteriorKeysProduct->plotOn(frame);
      frame->Draw(options);
   } else if (fDimension == 2) {
      RooArgList* axes = fInterval->GetAxes();
      RooRealVar* xVar = (RooRealVar*)axes->at(0);
      RooRealVar* yVar = (RooRealVar*)axes->at(1);
      TH2F* productHist = (TH2F*)fPosteriorKeysProduct->createHistogram(
            "prodPlot2D", *xVar, RooFit::YVar(*yVar), RooFit::Scaling(kFALSE));
      productHist->SetTitle(GetTitle());
      productHist->Draw(options);
      delete axes;
   }
}

void MCMCIntervalPlot::DrawChainScatter(RooRealVar& xVar, RooRealVar& yVar)
{
   const MarkovChain* markovChain = fInterval->GetChain();

   Int_t size = markovChain->Size();
   Int_t burnInSteps;
   if (fShowBurnIn)
      burnInSteps = fInterval->GetNumBurnInSteps();
   else
      burnInSteps = 0;

   Double_t* x = new Double_t[size - burnInSteps];
   Double_t* y = new Double_t[size - burnInSteps];
   Double_t* burnInX = NULL;
   Double_t* burnInY = NULL;
   if (burnInSteps > 0) {
      burnInX = new Double_t[burnInSteps];
      burnInY = new Double_t[burnInSteps];
   }
   Double_t firstX;
   Double_t firstY;

   for (Int_t i = burnInSteps; i < size; i++) {
      x[i - burnInSteps] = markovChain->Get(i)->getRealValue(xVar.GetName());
      y[i - burnInSteps] = markovChain->Get(i)->getRealValue(yVar.GetName());
   }

   for (Int_t i = 0; i < burnInSteps; i++) {
      burnInX[i] = markovChain->Get(i)->getRealValue(xVar.GetName());
      burnInY[i] = markovChain->Get(i)->getRealValue(yVar.GetName());
   }

   firstX = markovChain->Get(0)->getRealValue(xVar.GetName());
   firstY = markovChain->Get(0)->getRealValue(yVar.GetName());

   TGraph* walk = new TGraph(size - burnInSteps, x, y);
   // kbelasco: figure out how to set TGraph variable ranges
   walk->GetXaxis()->Set(xVar.numBins(), xVar.getMin(), xVar.getMax());
   walk->GetYaxis()->Set(yVar.numBins(), yVar.getMin(), yVar.getMax());
   walk->SetLineColor(kGray);
   walk->SetMarkerStyle(6);
   walk->SetMarkerColor(kViolet);
   walk->Draw("A,L,P,same");

   TGraph* burnIn = NULL;
   if (burnInX != NULL && burnInY != NULL) {
      burnIn = new TGraph(burnInSteps - 1, burnInX, burnInY);
      burnIn->SetLineColor(kPink);
      burnIn->SetMarkerStyle(6);
      burnIn->SetMarkerColor(kPink);
      burnIn->Draw("L,P,same");
   }

   TGraph* first = new TGraph(1, &firstX, &firstY);
   first->SetLineColor(kGreen);
   first->SetMarkerStyle(3);
   first->SetMarkerSize(2);
   first->SetMarkerColor(kGreen);
   first->Draw("L,P,same");

   //walkCanvas->Update();
   //delete x;
   //delete y;
   //delete burnInX;
   //delete burnInY;
   //delete walk;
   //delete burnIn;
   //delete first;
}

void MCMCIntervalPlot::DrawParameterVsTime(RooRealVar& param)
{
   const MarkovChain* markovChain = fInterval->GetChain();
   Int_t size = markovChain->Size();
   Int_t numEntries = 2 * size;
   Double_t* value = new Double_t[numEntries];
   Double_t* time = new Double_t[numEntries];
   Double_t val;
   Int_t weight;
   Int_t t = 0;
   for (Int_t i = 0; i < size; i++) {
      val = markovChain->Get(i)->getRealValue(param.GetName());
      weight = (Int_t)markovChain->Weight();
      value[2*i] = val;
      value[2*i + 1] = val;
      time[2*i] = t;
      t += weight;
      time[2*i + 1] = t;
   }

   TGraph* paramGraph = new TGraph(numEntries, time, value);
   //paramGraph->SetLineColor(fLineColor);
   paramGraph->Draw("A,L,same");
   //gPad->Update();
}

void MCMCIntervalPlot::DrawNLLVsTime()
{
   const MarkovChain* markovChain = fInterval->GetChain();
   Int_t size = markovChain->Size();
   Int_t numEntries = 2 * size;
   Double_t* nllValue = new Double_t[numEntries];
   Double_t* time = new Double_t[numEntries];
   Double_t nll;
   Int_t weight;
   Int_t t = 0;
   for (Int_t i = 0; i < size; i++) {
      nll = markovChain->NLL(i);
      weight = (Int_t)markovChain->Weight();
      nllValue[2*i] = nll;
      nllValue[2*i + 1] = nll;
      time[2*i] = t;
      t += weight;
      time[2*i + 1] = t;
   }

   TGraph* nllGraph = new TGraph(numEntries, time, nllValue);
   //nllGraph->SetLineColor(fLineColor);
   nllGraph->Draw("A,L,same");
   //gPad->Update();
}

void MCMCIntervalPlot::DrawNLLHist(const Option_t* options)
{
   if (fNLLHist == NULL) {
      const MarkovChain* markovChain = fInterval->GetChain();
      // find the max NLL value
      Double_t maxNLL = 0;
      Int_t size = markovChain->Size();
      for (Int_t i = 0; i < size; i++)
         if (markovChain->NLL(i) > maxNLL)
            maxNLL = markovChain->NLL(i);
      RooRealVar* nllVar = fInterval->GetNLLVar();
      fNLLHist = new TH1F("mcmc_nll_hist", "MCMC NLL Histogram", nllVar->getBins(),
            0, maxNLL);
      for (Int_t i = 0; i < size; i++)
         fNLLHist->Fill(markovChain->NLL(i), markovChain->Weight());
   }
   fNLLHist->Draw(options);
}

void MCMCIntervalPlot::DrawWeightHist(const Option_t* options)
{
   if (fWeightHist == NULL) {
      const MarkovChain* markovChain = fInterval->GetChain();
      // find the max weight value
      Double_t maxWeight = 0;
      Int_t size = markovChain->Size();
      for (Int_t i = 0; i < size; i++)
         if (markovChain->Weight(i) > maxWeight)
            maxWeight = markovChain->Weight(i);
      fWeightHist = new TH1F("mcmc_weight_hist", "MCMC Weight Histogram",
            (Int_t)(maxWeight + 1), 0, maxWeight * 1.02);
      for (Int_t i = 0; i < size; i++)
         fWeightHist->Fill(markovChain->Weight(i));
   }
   fWeightHist->Draw(options);
}

/*
/////////////////////////////////////////////////////////////////////
  // 3-d plot of the parameter points
  dataCanvas->cd(2);
  // also plot the points in the markov chain
  RooDataSet* markovChainData = ((MCMCInterval*)mcmcint)->GetChainAsDataSet();

  TTree& chain =  ((RooTreeDataStore*) markovChainData->store())->tree();
  chain.SetMarkerStyle(6);
  chain.SetMarkerColor(kRed);
  chain.Draw("s:ratioSigEff:ratioBkgEff","","box"); // 3-d box proporional to posterior

  // the points used in the profile construction
  TTree& parameterScan =  ((RooTreeDataStore*) fc.GetPointsToScan()->store())->tree();
  parameterScan.SetMarkerStyle(24);
  parameterScan.Draw("s:ratioSigEff:ratioBkgEff","","same");

  chain.SetMarkerStyle(6);
  chain.SetMarkerColor(kRed);
  //chain.Draw("s:ratioSigEff:ratioBkgEff", "_MarkovChain_local_nll","box");
  //chain.Draw("_MarkovChain_local_nll");
////////////////////////////////////////////////////////////////////
*/
