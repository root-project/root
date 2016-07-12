// @(#)root/hist:$Id$
// Author: Rene Brun   10/12/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "TROOT.h"
#include "TClassRef.h"
#include "TVirtualPad.h"
#include "TRatioPlot.h"
#include "TBrowser.h"
#include "TH1.h"
#include "TPad.h"
#include "TString.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TGAxis.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "../inc/TRatioPlot.h"

#define _(x) std::cout << #x;
#define __(x) std::cout << x << std::endl ;
#define var_dump(v) _(v); std::cout << "=" << (v) << std::endl;

ClassImp(TRatioPlot)

////////////////////////////////////////////////////////////////////////////////

/** \class TRatioPlot
    \ingroup Hist

*/


////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot default constructor

TRatioPlot::TRatioPlot()
   : TPad()//.
//     TNamed()
{
   std::cout << "hallo welt" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
///

// @TODO: This should work with stacks as well
TRatioPlot::TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, Option_t *divideOption)
   : TPad(name, title, 0, 0, 1, 1),
//     TNamed(name, title),
     fH1(h1),
     fH2(h2),
     fDivideOption(0),
     fUpperPad(0),
     fLowerPad(0),
     fTopPad(0),
     fRatioGraph(0),
     fSharedXAxis(0),
     fUpperGXaxis(0),
     fLowerGXaxis(0),
     fUpperGYaxis(0),
     fLowerGYaxis(0)
{
   gROOT->GetListOfCleanups()->Add(this);

   if (!fH1 || !fH2) {
      Warning("TRatioPlot", "Need two histograms.");
      return;
   }

   Bool_t h1IsTH1=fH1->IsA()->InheritsFrom(TH1::Class());
   Bool_t h2IsTH1=fH2->IsA()->InheritsFrom(TH1::Class());

   if (!h1IsTH1 && !h2IsTH1) {
      Warning("TRatioPlot", "Need two histograms deriving from TH2 or TH3.");
      return;
   }


   // taking x axis information from h1 by cloning it x axis
   fSharedXAxis = (TAxis*)(fH1->GetXaxis()->Clone());

   fParentPad = gPad;

   SetupPads();

   TString divideOptionString = TString(divideOption);

   if (divideOptionString.Contains("errprop")) {
      divideOptionString.ReplaceAll("errprop","");
      fDivideMode = DIVIDE_HIST;
   }
   else {
      fDivideMode = DIVIDE_GRAPH;
   }

   fDivideOption = divideOptionString;
}


void TRatioPlot::SetupPads() {

   if (fUpperPad != 0) {
      delete fUpperPad;
      fUpperPad = 0;
   }

   if (fLowerPad != 0) {
      delete fLowerPad;
      fLowerPad = 0;
   }

   fUpperPad = new TPad(TString::Format("%s_%s", fName.Data(), "upper_pad"), "", 0., fSplitFraction, 1., 1.);
   fLowerPad = new TPad(TString::Format("%s_%s", fName.Data(), "lower_pad"), "", 0., 0., 1., fSplitFraction);

   SetPadMargins();

   // connect to the pads signal
   fUpperPad->Connect("RangeChanged()", "TRatioPlot", this, "RangeAxisChanged()");
   fLowerPad->Connect("RangeChanged()", "TRatioPlot", this, "RangeAxisChanged()");

   fUpperPad->Connect("UnZoomed()", "TRatioPlot", this, "UnZoomed()");
   fLowerPad->Connect("UnZoomed()", "TRatioPlot", this, "UnZoomed()");



   if (fTopPad != 0) {
      delete fTopPad;
      fTopPad = 0;
   }

   Double_t margin = 0;//0.01;
   fTopPad = new TPad(TString::Format("%s_%s", fName.Data(), "top_pad"), "", margin, margin, 1-margin, 1-margin);

   fTopPad->SetBit(kCannotPick);

}


////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot destructor

//TRatioPlot::~TRatioPlot()
//{
//   gROOT->GetListOfCleanups()->Remove(this);
//}

////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot copy constructor

TRatioPlot::TRatioPlot(const TRatioPlot &hrp)
{
   Warning("TRatioPlot", "Copy constructor not yet implemented");
    return;
}


////////////////////////////////////////////////////////////////////////////////
/// Browse.

void TRatioPlot::Browse(TBrowser *b)
{
   Draw(b ? b->GetDrawOption() : "");
   gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Draw
void TRatioPlot::Draw(Option_t *option)
{

   __("TRatioPlot::Draw called");

   TVirtualPad *padsav = gPad;

   BuildRatio();
   TPad::Draw(option);

   // we are a TPad
   cd();

   fUpperPad->Draw();
   fLowerPad->Draw();

   fTopPad->SetFillStyle(0);
//   fTopPad->SetFillColor(kBlue);
   fTopPad->Draw();


   fUpperPad->cd();


   fH1->GetXaxis()->SetTickSize(0.);
   fH1->GetXaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);

   fH1->Draw("E");
   fH2->Draw("hist same");

   fLowerPad->cd();

   fRatioGraph->GetXaxis()->SetTickSize(0.);
   fRatioGraph->GetXaxis()->SetLabelSize(0.);
   fRatioGraph->GetYaxis()->SetTickSize(0.);
   fRatioGraph->GetYaxis()->SetLabelSize(0.);
   fRatioGraph->Draw("AP");


   fTopPad->cd();

//   CreateVisualAxes();

   SyncAxesRanges();

   padsav->cd();

}

void TRatioPlot::Paint(Option_t *opt) {

//   std::cout << "TRatioPlot::Paint called " << opt << std::endl;


//   CalculateSizes();

   fUpperPad->Paint();
   fLowerPad->Paint();

   CreateVisualAxes();


}

void TRatioPlot::SyncAxesRanges()
{
   // sync ranges
   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());


   fRatioGraph->GetXaxis()->SetLimits(first, last);
   fRatioGraph->GetXaxis()->SetRangeUser(first, last);
   fH1->GetXaxis()->SetRangeUser(first, last);
   fH2->GetXaxis()->SetRangeUser(first, last);

}

void TRatioPlot::BuildRatio()
{
//   __(__PRETTY_FUNCTION__ << " called");

   if (fRatioGraph != 0) {
      // have ratio graph, delete it
      fRatioGraph->IsA()->Destructor(fRatioGraph);
      fRatioGraph = 0;
   }

   if (fDivideMode == DIVIDE_GRAPH) {
//      std::cout << "DIVIDE_GRAPH " << fDivideOption << std::endl;
      TGraphAsymmErrors *ratioGraph = new TGraphAsymmErrors();
      ratioGraph->Divide(fH1, fH2, fDivideOption.Data());
      fRatioGraph = ratioGraph;
   }
   else {
//      std::cout << "DIVIDE_HIST " << fDivideOption << std::endl;
      TH1 *tmpHist = (TH1*)fH1->Clone();
      tmpHist->Reset();
      tmpHist->Divide(fH1, fH2, 1., 1., fDivideOption.Data());
      fRatioGraph = new TGraphErrors(tmpHist);
   }

   fRatioGraph->SetTitle("");
}



void TRatioPlot::CreateVisualAxes()
{

//   __(__PRETTY_FUNCTION__ << " called");

   if (fUpperGXaxis != 0) {
      delete fUpperGXaxis;
   }

   if (fLowerGXaxis != 0) {
      delete fLowerGXaxis;
   }

   if (fUpperGYaxis != 0) {
      delete fUpperGYaxis;
   }

   if (fLowerGYaxis != 0) {
      delete fLowerGYaxis;
   }


   Double_t upTM = fUpperPad->GetTopMargin();
   Double_t upBM = fUpperPad->GetBottomMargin();
   Double_t upLM = fUpperPad->GetLeftMargin();
   Double_t upRM = fUpperPad->GetRightMargin();

   Double_t lowTM = fLowerPad->GetTopMargin();
   Double_t lowBM = fLowerPad->GetBottomMargin();
   Double_t lowLM = fLowerPad->GetLeftMargin();
   Double_t lowRM = fLowerPad->GetRightMargin();

//   var_dump(upTM);
//   var_dump(upBM);
//   var_dump(upLM);
//   var_dump(upRM);
//   var_dump(lowTM);
//   var_dump(lowBM);
//   var_dump(lowLM);
//   var_dump(lowRM);

//   Double_t mf = 0.01 + 1;

   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());


   Double_t upYFirst = fUpperPad->GetUymin();
   Double_t upYLast = fUpperPad->GetUymax();
   Double_t lowYFirst = fLowerPad->GetUymin();
   Double_t lowYLast = fLowerPad->GetUymax();

   var_dump(upYFirst);
   var_dump(upYLast);
   var_dump(lowYFirst);
   var_dump(lowYLast);


   fUpperGXaxis = new TGaxis(upLM, upBM*(1-fSplitFraction)+fSplitFraction, (1-upRM), upBM*(1-fSplitFraction)+fSplitFraction, first, last);
   fUpperGXaxis->SetLabelSize(0.);

   fLowerGXaxis = new TGaxis(lowLM, lowBM*fSplitFraction, 1-lowRM, lowBM*fSplitFraction, first, last);

   fUpperGYaxis = new TGaxis(upLM, upBM*(1-fSplitFraction)+fSplitFraction, upLM, (1-upTM)*(1-fSplitFraction)+fSplitFraction, upYFirst, upYLast, 510, "-S");
   fLowerGYaxis = new TGaxis(lowLM, lowBM*fSplitFraction, lowLM, (1-lowTM)*fSplitFraction, lowYFirst, lowYLast, 510, "-S");

   Double_t ratio = ( (upBM-(1-upTM))*(1-fSplitFraction) ) / ( (lowBM-(1-lowTM))*fSplitFraction ) ;

   var_dump(ratio);

//   fUpperGYaxis->SetTickSize(0.02);
   fLowerGYaxis->SetTickSize(fUpperGYaxis->GetTickSize()*ratio);

   TVirtualPad *padsav = gPad;

   fTopPad->cd();

   fUpperGXaxis->Draw();
   fLowerGXaxis->Draw();
   fUpperGYaxis->Draw();
   fLowerGYaxis->Draw();

   padsav->cd();

}

void TRatioPlot::SetPadMargins()
{
   fUpperPad->SetTopMargin(fUpTopMargin);
   fUpperPad->SetBottomMargin(fUpBottomMargin);
   fUpperPad->SetLeftMargin(fLeftMargin);
   fUpperPad->SetRightMargin(fRightMargin);
   fLowerPad->SetTopMargin(fLowTopMargin);
   fLowerPad->SetBottomMargin(fLowBottomMargin);
   fLowerPad->SetLeftMargin(fLeftMargin);
   fLowerPad->SetRightMargin(fRightMargin);
}

Bool_t TRatioPlot::SyncPadMargins()
{

   Bool_t changed = kFALSE;

   if (fUpperPad->GetLeftMargin() != fLeftMargin) {
      fLeftMargin = fUpperPad->GetLeftMargin();
      changed = kTRUE;
   }
   else if (fLowerPad->GetLeftMargin() != fLeftMargin) {
      fLeftMargin = fLowerPad->GetLeftMargin();
      changed = kTRUE;
   }

   if (fUpperPad->GetRightMargin() != fRightMargin) {
      fRightMargin = fUpperPad->GetRightMargin();
      changed = kTRUE;
   }
   else if (fLowerPad->GetRightMargin() != fRightMargin) {
      fRightMargin = fLowerPad->GetRightMargin();
      changed = kTRUE;
   }

   if (changed) {

      SetPadMargins();
   }


   Bool_t verticalChanged = kFALSE;
   Double_t delta;

   if (fUpperPad->GetBottomMargin() != fUpBottomMargin) {

      verticalChanged = kTRUE;
      fUpBottomMargin = fUpperPad->GetBottomMargin();
      delta = (fUpBottomMargin - fUpBottomMarginNominal)*(1-fSplitFraction);

   } else if (fLowerPad->GetTopMargin() != fLowTopMargin) {

      verticalChanged = kTRUE;
      fLowTopMargin = fLowerPad->GetTopMargin();
      delta = -(fLowTopMargin - fLowTopMarginNominal)*fSplitFraction;

   } else if (fLowerPad->GetBottomMargin() != fLowBottomMargin) {

      fLowBottomMargin = fLowerPad->GetBottomMargin();

   } else if (fUpperPad->GetTopMargin() != fUpTopMargin) {

      fUpTopMargin = fUpperPad->GetTopMargin();

   }


   if (verticalChanged) {
      SetSplitFraction(fSplitFraction + delta);
      SetPadMargins();
   }

   return changed || verticalChanged;

}

void TRatioPlot::RangeAxisChanged()
{

//   __(__PRETTY_FUNCTION__);

   Double_t upFirst = fH1->GetXaxis()->GetBinLowEdge(fH1->GetXaxis()->GetFirst());
   Double_t upLast  = fH1->GetXaxis()->GetBinUpEdge(fH1->GetXaxis()->GetLast());

   Double_t lowFirst = fRatioGraph->GetXaxis()->GetBinLowEdge(fRatioGraph->GetXaxis()->GetFirst());
   Double_t lowLast  = fRatioGraph->GetXaxis()->GetBinUpEdge(fRatioGraph->GetXaxis()->GetLast());

   Double_t globFirst = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t globLast = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   Bool_t upChanged = kFALSE;
   Bool_t lowChanged = kFALSE;

   if (upFirst != globFirst || upLast != globLast) {
      fSharedXAxis->SetRangeUser(upFirst, upLast);
      upChanged = kTRUE;
   }
   else if (lowFirst != globFirst || lowLast != globLast) {
      fSharedXAxis->SetRangeUser(lowFirst, lowLast);
      lowChanged = kTRUE;
   }
   else {
   }

//   var_dump(upFirst);
//   var_dump(upLast);
//   var_dump(lowFirst);
//   var_dump(lowLast);
//   var_dump(globFirst);
//   var_dump(globLast);



   if (upChanged || lowChanged) {
      __("Axis range changed");
      SyncAxesRanges();
   }

   Bool_t marginsChanged = SyncPadMargins();


   if (marginsChanged) {
      SetSplitFraction(fSplitFraction);
      Paint();
   }

   if (upChanged || lowChanged) {


      Paint();


   }

}

void TRatioPlot::UnZoomed()
{
   __(__PRETTY_FUNCTION__ << " called");

   fSharedXAxis->SetRange(0, 0);

   SyncAxesRanges();


   Paint();
}

void TRatioPlot::SetSplitFraction(Float_t sf) {
   fSplitFraction = sf;
   fUpperPad->SetPad(0., fSplitFraction, 1., 1.);
   fLowerPad->SetPad(0., 0., 1., fSplitFraction);
}
