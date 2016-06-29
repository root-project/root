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

ClassImp(TRatioPlot)

////////////////////////////////////////////////////////////////////////////////

/** \class TRatioPlot
    \ingroup Hist

*/


////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot default constructor

TRatioPlot::TRatioPlot(): TNamed()
{
   std::cout << "hallo welt" << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
///

// @TODO: This should work with stacks as well
TRatioPlot::TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, Option_t *divideOption)
   : TNamed(name, title),
     fH1(h1),
     fH2(h2),
     fDivideOption(0),
     fUpperPad(0),
     fLowerPad(0),
     fTopPad(0),
     fRatioGraph(0),
     fSharedXAxis(0)
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
//      fUpperPad->IsA()->
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
   fTopPad = new TPad(TString::Format("%s_%s", fName.Data(), "top_pad"), "", 0., 0., 1., 1.);

   fTopPad->SetBit(kCannotPick);
}


////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot destructor

TRatioPlot::~TRatioPlot()
{
   gROOT->GetListOfCleanups()->Remove(this);
}

////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot copy constructor

TRatioPlot::TRatioPlot(const TRatioPlot &hrp) :
   TNamed(hrp)
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
//   TString opt = option;
//   opt.ToLower();
//   if (gPad) {
//      if (!gPad->IsEditable()) gROOT->MakeDefCanvas();
//      if (!opt.Contains("same")) {
//         //the following statement is necessary in case one attempts to draw
//         //a temporary histogram already in the current pad
//         if (TestBit(kCanDelete)) gPad->GetListOfPrimitives()->Remove(this);
//         gPad->Clear();
//      }
//   }
//   AppendPad(opt.Data());

   BuildRatio();
   CalculateSizes();

   fUpperPad->Draw();
   fLowerPad->Draw();

   fTopPad->SetFillStyle(0);
   fTopPad->Draw();


   // set up graphical axis
   TGaxis *gaxis = new TGaxis(0., 0.2, 1., 0.2, 0., 10.);



   fUpperPad->cd();

   fH1->GetXaxis()->SetLabelOffset(999);
   fH1->GetXaxis()->SetLabelSize(0);
   fH1->GetXaxis()->SetTickLength(0);

   fH1->Draw("E");
   fH2->Draw("hist same");

   fTopPad->cd();
   gaxis->Draw();
//   fSharedXAxis->Draw();


   fLowerPad->cd();
   fRatioGraph->GetXaxis()->SetLabelOffset(999);
   fRatioGraph->GetXaxis()->SetLabelSize(0);
   fRatioGraph->GetXaxis()->SetTickLength(0);
   fRatioGraph->Draw("AP");
   gaxis->Draw();


//   fSharedXAxis->Draw();

   fParentPad->cd();

   AppendPad();
}

void TRatioPlot::Paint(Option_t *opt) {

   std::cout << "RP Paint called " << opt << std::endl;

   fH1->SetLineColor(kRed);

//   fUpperPad->Paint();
//   fLowerPad->Paint();
//   fTopPad->Paint();

//   fH1->Paint();
//   fH2->Paint();

//   fTopPad->SetFillStyle(0);
//   fTopPad->Draw();


}

void TRatioPlot::CalculateSizes() {

   fUpperPad->SetBottomMargin(0.05);
   fLowerPad->SetTopMargin(0.05);

   // sync ranges
   double first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   double last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   std::cout << first << "-" << last << std::endl;

   fRatioGraph->GetXaxis()->SetRangeUser(first, last);

}

void TRatioPlot::BuildRatio()
{
//   std::cout << __FUNCTION__ << std::endl;

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

void TRatioPlot::SetSplitFraction(Float_t sf) {
   fSplitFraction = sf;
   SetupPads();
}


