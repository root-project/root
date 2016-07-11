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
     fUpperGaxis(0),
     fLowerGaxis(0)
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

   if (fTopPad != 0) {
      delete fTopPad;
      fTopPad = 0;
   }

   fUpperPad = new TPad(TString::Format("%s_%s", fName.Data(), "upper_pad"), "", 0., fSplitFraction, 1., 1.);
   fLowerPad = new TPad(TString::Format("%s_%s", fName.Data(), "lower_pad"), "", 0., 0., 1., fSplitFraction);


//   fUpperPad->SetTopMargin(fUpTopMargin);
//   fUpperPad->SetBottomMargin(fUpBottomMargin);
//   fUpperPad->SetLeftMargin(fLeftMargin);
//   fUpperPad->SetRightMargin(fRightMargin);
//
//   fLowerPad->SetTopMargin(fLowTopMargin);
//   fLowerPad->SetBottomMargin(fLowBottomMargin);
//   fLowerPad->SetLeftMargin(fLeftMargin);
//   fLowerPad->SetRightMargin(fRightMargin);

   SetPadMargins();


   fTopPad = new TPad(TString::Format("%s_%s", fName.Data(), "top_pad"), "", 0., 0., 1., 1.);

   fTopPad->SetBit(kCannotPick);

   // connect to the pads events
   fUpperPad->Connect("RangeChanged()", "TRatioPlot", this, "RangeAxisChanged()");
   fLowerPad->Connect("RangeChanged()", "TRatioPlot", this, "RangeAxisChanged()");
}


////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot destructor

//TRatioPlot::~TRatioPlot()
//{
//   gROOT->GetListOfCleanups()->Remove(this);
//}

////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot copy constructor

TRatioPlot::TRatioPlot(const TRatioPlot &hrp) //:
//   TPad(hrp)//,
//   TNamed(hrp)
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

   // we are a TPad o.O
   cd();

   fUpperPad->Draw();
   fLowerPad->Draw();

   fTopPad->SetFillStyle(0);
   fTopPad->AppendPad();


   Double_t newGlobFirst = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t newGlobLast = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   fUpperPad->cd();
   fH1->Draw("E");
   fH2->Draw("hist same");

   fLowerPad->cd();
   fRatioGraph->Draw("AP");


   fTopPad->cd();
   padsav->cd();

   __(__FUNCTION__ << " gPad: " << gPad->GetName());
}

void TRatioPlot::Paint(Option_t *opt) {

   __(__FUNCTION__ << " gPad: " << gPad->GetName());

   std::cout << "TRatioPlot::Paint called " << opt << std::endl;

   // lets check if the axis range of one of our pads has changed

//   TList *upPrim = fUpperPad->GetListOfPrimitives();
//   upPrim->ls();
//
//   TFrame *upFrame = dynamic_cast<TFrame*>(upPrim->At(0));
//
//   var_dump(upFrame);

   CalculateSizes();
//   CreateVisualAxes();

//   fUpperGaxis->Paint();
//   fLowerGaxis->Paint();

//   fUpperPad->Paint();
//   fLowerPad->Paint();
   fH1->Paint();
   fH2->Paint();

   __(__FUNCTION__ << " gPad: " << gPad->GetName());
}

void TRatioPlot::CalculateSizes()
{

//   __(__PRETTY_FUNCTION__ << " called");

   fUpperPad->SetBottomMargin(0.05);
   fLowerPad->SetTopMargin(0.05);

}

void TRatioPlot::SyncAxesRanges()
{
   // sync ranges
   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

//   var_dump(first);
//   var_dump(last);

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
      std::cout << "DIVIDE_GRAPH " << fDivideOption << std::endl;
      TGraphAsymmErrors *ratioGraph = new TGraphAsymmErrors();
      ratioGraph->Divide(fH1, fH2, fDivideOption.Data());
      fRatioGraph = ratioGraph;
   }
   else {
      std::cout << "DIVIDE_HIST " << fDivideOption << std::endl;
      TH1 *tmpHist = (TH1*)fH1->Clone();
      tmpHist->Reset();
      tmpHist->Divide(fH1, fH2, 1., 1., fDivideOption.Data());
      fRatioGraph = new TGraphErrors(tmpHist);
   }

   fRatioGraph->SetTitle("");
}

void TRatioPlot::CreateVisualAxes()
{

   __(__PRETTY_FUNCTION__ << " called");

   if (fUpperGaxis != 0) {
      delete fUpperGaxis;
   }

   if (fLowerGaxis != 0) {
      delete fLowerGaxis;
   }


//   Double_t upTM = fUpperPad->GetTopMargin();
//   Double_t upBM = fUpperPad->GetBottomMargin();
//   Double_t upLM = fUpperPad->GetLeftMargin();
//   Double_t upRM = fUpperPad->GetRightMargin();
//
//   Double_t lowTM = fLowerPad->GetTopMargin();
//   Double_t lowBM = fLowerPad->GetBottomMargin();
//   Double_t lowLM = fLowerPad->GetLeftMargin();
//   Double_t lowRM = fLowerPad->GetRightMargin();
//
//   var_dump(upTM);
//   var_dump(upBM);
//   var_dump(upLM);
//   var_dump(upRM);
//   var_dump(lowTM);
//   var_dump(lowBM);
//   var_dump(lowLM);
//   var_dump(lowRM);

//
//   Double_t upWNDC = fUpperPad->GetWNDC();
//   Double_t upHNDC = fUpperPad->GetHNDC();
//   Double_t upLowXNDC = fUpperPad->GetXlowNDC();
//   Double_t upLowYNDC = fUpperPad->GetYlowNDC();
//   Double_t lowWNDC = fLowerPad->GetWNDC();
//   Double_t lowHNDC = fLowerPad->GetHNDC();
//   Double_t lowLowXNDC = fLowerPad->GetXlowNDC();
//   Double_t lowLowYNDC = fLowerPad->GetYlowNDC();
//
//
//
//   var_dump(upWNDC);
//   var_dump(upHNDC);
//   var_dump(upLowXNDC);
//   var_dump(upLowYNDC);
//   var_dump(lowWNDC);
//   var_dump(lowHNDC);
//   var_dump(lowLowXNDC);
//   var_dump(lowLowYNDC);


//   Double_t upUxmin = fUpperPad->GetUxmin();
//   Double_t upUxmax = fUpperPad->GetUxmax();
//   Double_t upUymin = fUpperPad->GetUymin();
//   Double_t upUymax = fUpperPad->GetUymax();
//   Double_t lowUxmin = fLowerPad->GetUxmin();
//   Double_t lowUxmax = fLowerPad->GetUxmax();
//   Double_t lowUymin = fLowerPad->GetUymin();
//   Double_t lowUymax = fLowerPad->GetUymax();
//
//
//   var_dump(upUxmin);
//   var_dump(upUxmax);
//   var_dump(upUymin);
//   var_dump(upUymax);
//   var_dump(lowUxmin);
//   var_dump(lowUxmax);
//   var_dump(lowUymin);
//   var_dump(lowUymax);
//
//
//
//
//
//   Double_t upX1 = fUpperPad->GetX1();
//   Double_t upX2 = fUpperPad->GetX2();
//   Double_t upY1 = fUpperPad->GetY1();
//   Double_t lowX1 = fUpperPad->GetX1();
//   Double_t lowX2 = fUpperPad->GetX2();
//   Double_t lowY1 = fUpperPad->GetY1();
//
//   var_dump(upX1);
//   var_dump(upX2);
//   var_dump(upY1);
//   var_dump(lowX1);
//   var_dump(lowX2);
//   var_dump(lowY1);


//   fUpperGaxis = new TGaxis(upLM, upBM+fSplitFraction, 1-upRM, upBM+fSplitFraction, 0., 1.);
//   fLowerGaxis = new TGaxis(lowLM, lowBM, 1-lowRM, lowBM, 0., 1.);

//   fUpperGaxis = new TGaxis(0, 0, 0, 0, 0., 1.);
//   fLowerGaxis = new TGaxis(0, 0, 0, 0, 0., 1.);

//   TVirtualPad *padsav = gPad;
//
//   fTopPad->cd();

//   fUpperGaxis->Draw();
//   fLowerGaxis->Draw();
//
//   padsav->cd();

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

void TRatioPlot::SyncPadMargins()
{

//   TQObject *sender = GetSender();
   var_dump(gTQSender == fUpperPad);


//   var_dump(fUpperPad->GetLeftMargin());
//   var_dump(fLowerPad->GetLeftMargin());

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
      __("margins changed");
      fLowerPad->SetLeftMargin(fLeftMargin);
      fUpperPad->SetLeftMargin(fLeftMargin);
      fLowerPad->SetRightMargin(fRightMargin);
      fUpperPad->SetRightMargin(fRightMargin);

      fCanvas->Update();
   }


   Bool_t verticalChanged = kFALSE;
   Double_t delta;

   if (fUpperPad->GetBottomMargin() != fUpBottomMargin) {
      verticalChanged = kTRUE;
      fUpBottomMargin = fUpperPad->GetBottomMargin();
      delta = (fUpBottomMargin - fUpBottomMarginNominal)*(1-fSplitFraction);
   }  else if (fLowerPad->GetTopMargin() != fLowTopMargin) {
      verticalChanged = kTRUE;
      fLowTopMargin = fLowerPad->GetTopMargin();
      delta = -(fLowTopMargin - fLowTopMarginNominal)*fSplitFraction;
   }

   if (verticalChanged) {
      var_dump(fLowTopMargin);
      var_dump(fUpBottomMargin);

      var_dump(delta);

      SetSplitFraction(fSplitFraction + delta);
      SetPadMargins();
//      Draw();

   }





}

void TRatioPlot::RangeAxisChanged()
{
   __(__PRETTY_FUNCTION__);
   __(__FUNCTION__ << " gPad: " << gPad->GetName());


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

   if (upChanged || lowChanged) {
      __("Axis range changed");
      SyncAxesRanges();
      fLowerPad->Paint();
      fUpperPad->Paint();
      fLowerPad->Update();
      fUpperPad->Update();
   }

   SyncPadMargins();

   TVirtualPad *padsav = gPad;
//   var_dump(gPad->GetName());

   if (upChanged) {
//      fLowerPad->cd();
//      fLowerPad->Update();
//      fLowerPad->Paint();
   }

   if (upChanged || lowChanged) {
//      fParentPad->Modified();
//      fParentPad->Update();
//      fParentPad->Paint();

      if (fCanvas) {
//         __("have canvas");
//         fCanvas->Modified();
//         fCanvas->Update();
//         fCanvas->Flush();
      }

   }

   padsav->cd();


}

void TRatioPlot::UnZoom(TAxis*)
{
//   __(__PRETTY_FUNCTION__ << " called");
//
//   fSharedXAxis->SetRange(0, 0);
//
//   SyncAxesRanges();
//   Paint();

}


void TRatioPlot::SetSplitFraction(Float_t sf) {
   fSplitFraction = sf;
//   SetupPads();
//   Draw();
   fUpperPad->SetPad(0., fSplitFraction, 1., 1.);
   fLowerPad->SetPad(0., 0., 1., fSplitFraction);

//   var_dump(fUpperPad->GetBBoxY1());

//   fUpperPad = new TPad(TString::Format("%s_%s", fName.Data(), "upper_pad"), "", );
//   fLowerPad = new TPad(TString::Format("%s_%s", fName.Data(), "lower_pad"), "", 0., 0., 1., fSplitFraction);

}
