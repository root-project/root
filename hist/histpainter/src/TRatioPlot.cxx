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
#include "TGaxis.h"
#include "TCanvas.h"
#include "TFrame.h"

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
// @TODO: Needs options for axes
// @TODO: Needs options for drawing of h1, h2, ratio
// @TODO: Need getters for pads

TRatioPlot::TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, 
      Option_t *divideOption, Option_t *optH1, Option_t *optH2, Option_t *optGraph)
   : TPad(name, title, 0, 0, 1, 1),
     fUpperPad(0),
     fLowerPad(0),
     fTopPad(0),
     fH1(h1),
     fH2(h2),
     fDivideOption(0),
     fOptH1(0),
     fOptH2(0),
     fOptGraph(0),
     fRatioGraph(0),
     fSharedXAxis(0),
     fUpperGXaxis(0),
     fLowerGXaxis(0),
     fUpperGYaxis(0),
     fLowerGYaxis(0),
     fUpperGXaxisMirror(0),
     fLowerGXaxisMirror(0),
     fUpperGYaxisMirror(0),
     fLowerGYaxisMirror(0),
     fUpYaxis(0),
     fLowYaxis(0)
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

   TString optH1String = TString(optH1);
   TString optH2String = TString(optH2);
   TString optGraphString = TString(optGraph);
   
   optH2String.ReplaceAll("same", "");
   optH2String.ReplaceAll("SAME", "");

   fOptH1 = optH1String;
   fOptH2 = optH2String;
   fOptGraph = optGraphString;

   // build ratio, everything is ready
   BuildRatio();

   // taking x axis information from h1 by cloning it x axis
   fSharedXAxis = (TAxis*)(fH1->GetXaxis()->Clone());
   fUpYaxis = (TAxis*)(fH1->GetYaxis()->Clone());
   fLowYaxis = (TAxis*)(fRatioGraph->GetYaxis()->Clone());
}


void TRatioPlot::SetupPads() {

   // this method will delete all the pads before recreating them

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
   fUpperPad->Connect("RangeAxisChanged()", "TRatioPlot", this, "RangeAxisChanged()");
   fLowerPad->Connect("RangeAxisChanged()", "TRatioPlot", this, "RangeAxisChanged()");

   fUpperPad->Connect("UnZoomed()", "TRatioPlot", this, "UnZoomed()");
   fLowerPad->Connect("UnZoomed()", "TRatioPlot", this, "UnZoomed()");

   fUpperPad->Connect("Resized()", "TRatioPlot", this, "SubPadResized()");
   fLowerPad->Connect("Resized()", "TRatioPlot", this, "SubPadResized()");


   if (fTopPad != 0) {
      delete fTopPad;
      fTopPad = 0;
   }

   Double_t margin = 0;
   fTopPad = new TPad(TString::Format("%s_%s", fName.Data(), "top_pad"), "", margin, margin, 1-margin, 1-margin);

   fTopPad->SetBit(kCannotPick);

}



////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot copy constructor

TRatioPlot::TRatioPlot(const TRatioPlot &/*hrp*/)
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

void TRatioPlot::SetUpTopMargin(Float_t margin)
{
   fUpTopMargin = margin;
   SetPadMargins();
}

void TRatioPlot::SetUpBottomMargin(Float_t margin)
{
   fUpBottomMargin = margin;
   SetPadMargins();
}

void TRatioPlot::SetLowTopMargin(Float_t margin)
{
   fLowTopMargin = margin;
   SetPadMargins();
}

void TRatioPlot::SetLowBottomMargin(Float_t margin)
{
   fLowBottomMargin = margin;
   SetPadMargins();
}

void TRatioPlot::SetLeftMargin(Float_t margin)
{
   fLeftMargin = margin;
   SetPadMargins();
}

void TRatioPlot::SetRightMargin(Float_t margin)
{
   fRightMargin = margin;
   SetPadMargins();
}

void TRatioPlot::SetSeparationMargin(Float_t margin)
{
   Float_t sf = fSplitFraction;
   fUpBottomMargin = margin/2./(1-sf);
   fLowTopMargin = margin/2./sf;
   SetPadMargins();
}



////////////////////////////////////////////////////////////////////////////////
/// Draw
void TRatioPlot::Draw(Option_t *option)
{

   //__("TRatioPlot::Draw called");

   TVirtualPad *padsav = gPad;

   TPad::Draw(option);

   // we are a TPad
   cd();

   fUpperPad->Draw();
   fLowerPad->Draw();

   fTopPad->SetFillStyle(0);
   fTopPad->Draw();


   fUpperPad->cd();

   // we need to hide the original axes of the hist 
   fH1->GetXaxis()->SetTickSize(0.);
   fH1->GetXaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);

   fH1->Draw(fOptH1);
   fH2->Draw(fOptH2+"same");

   fLowerPad->cd();

   // hide visual axis of lower pad display
   fRatioGraph->GetXaxis()->SetTickSize(0.);
   fRatioGraph->GetXaxis()->SetLabelSize(0.);
   fRatioGraph->GetYaxis()->SetTickSize(0.);
   fRatioGraph->GetYaxis()->SetLabelSize(0.);
   fRatioGraph->Draw(fOptGraph);


   fTopPad->cd();

   // assign same axis ranges to lower pad as in upper pad
   // the visual axes will be created on paint
   SyncAxesRanges();

   padsav->cd();

}

// Does not really do anything right now, other than call super
// @TODO: Remove this if not needed
void TRatioPlot::Paint(Option_t *opt) {
   TPad::Paint(opt);
}

void TRatioPlot::PaintModified()
{
   TPad::PaintModified();
   
   // sync y axes
   fH1->GetYaxis()->ImportAttributes(fUpYaxis);
   fRatioGraph->GetYaxis()->ImportAttributes(fLowYaxis);

   fH1->GetXaxis()->SetTickSize(0.);
   fH1->GetXaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);
   
   fRatioGraph->GetXaxis()->SetTickSize(0.);
   fRatioGraph->GetXaxis()->SetLabelSize(0.);
   fRatioGraph->GetYaxis()->SetTickSize(0.);
   fRatioGraph->GetYaxis()->SetLabelSize(0.);
   
   // create the visual axes
   CreateVisualAxes();
}

void TRatioPlot::SyncAxesRanges()
{
   // get ranges from the shared axis clone
   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   // set range on computed graph, have to set it twice becaus 
   // TGraph's axis looks strange otherwise
   fRatioGraph->GetXaxis()->SetLimits(first, last);
   fRatioGraph->GetXaxis()->SetRangeUser(first, last);

   fH1->GetXaxis()->SetRangeUser(first, last);
   //fH2->GetXaxis()->SetRangeUser(first, last);

}

////////////////////////////////////////////////////////////////////////////////
/// Build the lower plot according to which constructor was called, and
/// which options were passed.
/// @TODO: Actually implement this, currently it can only do ratios
void TRatioPlot::BuildRatio()
{
//   __(__PRETTY_FUNCTION__ << " called");

   // Clear and delete the graph if not exists
   if (fRatioGraph != 0) {
      fRatioGraph->IsA()->Destructor(fRatioGraph);
      fRatioGraph = 0;
   }

   // Determine the divide mode and create the lower graph accordingly
   // Pass divide options given in constructor
   if (fDivideMode == DIVIDE_GRAPH) {
      // use TGraphAsymmErrors Divide method to create
      TGraphAsymmErrors *ratioGraph = new TGraphAsymmErrors();
      ratioGraph->Divide(fH1, fH2, fDivideOption.Data());
      fRatioGraph = ratioGraph;
   }
   else {
      // Use TH1's Divide method
      // @TODO: Add factors for both histograms: c1, c2, via constructor and setters
      TH1 *tmpHist = (TH1*)fH1->Clone();
      tmpHist->Reset();
      tmpHist->Divide(fH1, fH2, 1., 1., fDivideOption.Data());
      fRatioGraph = new TGraphErrors(tmpHist);
   }

   fRatioGraph->SetTitle("");
}


////////////////////////////////////////////////////////////////////////////////
/// (Re-)Creates the TGAxis objects that are used for consistent display of the 
/// axes.
void TRatioPlot::CreateVisualAxes()
{

//   __(__PRETTY_FUNCTION__ << " called");


   if (fUpperGXaxis != 0) delete fUpperGXaxis;
   if (fLowerGXaxis != 0) delete fLowerGXaxis;
   if (fUpperGYaxis != 0) delete fUpperGYaxis;
   if (fLowerGYaxis != 0) delete fLowerGYaxis;

   if (fUpperGXaxisMirror != 0) delete fUpperGXaxisMirror;
   if (fLowerGXaxisMirror != 0) delete fLowerGXaxisMirror;
   if (fUpperGYaxisMirror != 0) delete fUpperGYaxisMirror;
   if (fLowerGYaxisMirror != 0) delete fLowerGYaxisMirror;


   // figure out where the axis has to go.
   // Implicit assumption is, that the top pad spans the full other pads
   Double_t upTM = fUpperPad->GetTopMargin();
   Double_t upBM = fUpperPad->GetBottomMargin();
   Double_t upLM = fUpperPad->GetLeftMargin();
   Double_t upRM = fUpperPad->GetRightMargin();

   Double_t lowTM = fLowerPad->GetTopMargin();
   Double_t lowBM = fLowerPad->GetBottomMargin();
   Double_t lowLM = fLowerPad->GetLeftMargin();
   Double_t lowRM = fLowerPad->GetRightMargin();

   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   Double_t upYFirst = fUpperPad->GetUymin();
   Double_t upYLast = fUpperPad->GetUymax();
   Double_t lowYFirst = fLowerPad->GetUymin();
   Double_t lowYLast = fLowerPad->GetUymax();

   Float_t sf = fSplitFraction;

   fUpperGXaxis = new TGaxis(upLM, upBM*(1-sf)+sf, (1-upRM), upBM*(1-sf)+sf, first, last, 510, "+U");
   fLowerGXaxis = new TGaxis(lowLM, lowBM*sf, 1-lowRM, lowBM*sf, first, last);
   fUpperGYaxis = new TGaxis(upLM, upBM*(1-sf)+sf, upLM, (1-upTM)*(1-sf)+sf, upYFirst, upYLast, 510, "-S");
   fLowerGYaxis = new TGaxis(lowLM, lowBM*sf, lowLM, (1-lowTM)*sf, lowYFirst, lowYLast, 510, "-S");

   // U would disable labels but breaks tick size, so S and SetLabelSize(0.)
   fUpperGXaxisMirror = new TGaxis(upLM, (1-upTM)*(1-sf)+sf, (1-upRM), (1-upTM)*(1-sf)+sf, first, last, 510, "-S");
   fLowerGXaxisMirror = new TGaxis(lowLM, (1-lowTM)*sf, 1-lowRM, (1-lowTM)*sf, first, last, 510, "-S");
   fUpperGYaxisMirror = new TGaxis(1-upRM, upBM*(1-sf)+sf, 1-upRM, (1-upTM)*(1-sf)+sf, upYFirst, upYLast, 510, "+S");
   fLowerGYaxisMirror = new TGaxis(1-lowRM, lowBM*sf, 1-lowRM, (1-lowTM)*sf, lowYFirst, lowYLast, 510, "+S");
   

   // import infos from TAxes
   fUpperGXaxis->ImportAxisAttributes(fSharedXAxis);
   fUpperGYaxis->ImportAxisAttributes(fUpYaxis);
   fLowerGXaxis->ImportAxisAttributes(fSharedXAxis);
   fLowerGYaxis->ImportAxisAttributes(fLowYaxis);

   fUpperGXaxis->SetNdivisions(fSharedXAxis->GetNdivisions());
   fUpperGYaxis->SetNdivisions(fUpYaxis->GetNdivisions());
   fLowerGXaxis->SetNdivisions(fSharedXAxis->GetNdivisions());
   fLowerGYaxis->SetNdivisions(fLowYaxis->GetNdivisions());
   
   // check if gPad has the all sides axis set

   Bool_t mirroredAxes = gPad->GetFrameFillStyle() == 0; 

   if (mirroredAxes) {
   
      fUpperGXaxisMirror->ImportAxisAttributes(fSharedXAxis);
      fUpperGYaxisMirror->ImportAxisAttributes(fUpYaxis);
      fLowerGXaxisMirror->ImportAxisAttributes(fSharedXAxis);
      fLowerGYaxisMirror->ImportAxisAttributes(fLowYaxis);

      fUpperGXaxisMirror->SetNdivisions(fSharedXAxis->GetNdivisions());
      fUpperGYaxisMirror->SetNdivisions(fUpYaxis->GetNdivisions());
      fLowerGXaxisMirror->SetNdivisions(fSharedXAxis->GetNdivisions());
      fLowerGYaxisMirror->SetNdivisions(fLowYaxis->GetNdivisions());
      
      fUpperGXaxisMirror->SetLabelSize(0.);
      fLowerGXaxisMirror->SetLabelSize(0.);
      fUpperGYaxisMirror->SetLabelSize(0.);
      fLowerGYaxisMirror->SetLabelSize(0.);

   }


   // normalize the tick sizes. y axis ticks should be consistent
   // even if their length is different
   Double_t ratio = ( (upBM-(1-upTM))*(1-sf) ) / ( (lowBM-(1-lowTM))*sf ) ;
   //fUpperGXaxis->SetLabelSize(0.);
   Double_t ticksize = fUpperGYaxis->GetTickSize()*ratio;  
   var_dump(ticksize);
   fLowerGYaxis->SetTickSize(ticksize);
   fLowerGYaxisMirror->SetTickSize(ticksize);

   // draw TG axes to top pad
   TVirtualPad *padsav = gPad;
   fTopPad->cd();
   fUpperGXaxis->Draw();
   fLowerGXaxis->Draw();
   fUpperGYaxis->Draw();
   fLowerGYaxis->Draw();

   if (mirroredAxes) {

      fUpperGXaxisMirror->Draw();
      fLowerGXaxisMirror->Draw();
      fUpperGYaxisMirror->Draw();
      fLowerGYaxisMirror->Draw();

   }

   padsav->cd();

}

////////////////////////////////////////////////////////////////////////////////
/// Sets the margins of all the pads to the value specified in class members.
/// This one is called whenever those are changed, e.g. in setters
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

////////////////////////////////////////////////////////////////////////////////
/// Figures out which pad margin has deviated from the stored ones,
/// to figure out what the new nominal is and set the other pad to it
/// subsequently.
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

   if (fUpperPad->GetBottomMargin() != fUpBottomMargin) {

      verticalChanged = kTRUE;
      fUpBottomMargin = fUpperPad->GetBottomMargin();

   }
   
   if (fLowerPad->GetTopMargin() != fLowTopMargin) {

      verticalChanged = kTRUE;
      fLowTopMargin = fLowerPad->GetTopMargin();

   }
   
   if (fLowerPad->GetBottomMargin() != fLowBottomMargin) {

      fLowBottomMargin = fLowerPad->GetBottomMargin();

   }
   
   if (fUpperPad->GetTopMargin() != fUpTopMargin) {

      fUpTopMargin = fUpperPad->GetTopMargin();

   }


   if (verticalChanged) {
      SetPadMargins();
   }

   return changed || verticalChanged;

}

////////////////////////////////////////////////////////////////////////////////
/// Slot that receives the RangeAxisChanged signal from any of the pads and
/// reacts correspondingly.
void TRatioPlot::RangeAxisChanged()
{
   // Only run this concurrently once, in case it's called async
   if (fIsUpdating) {
      return;
   }

   //__("TRatioPlot::RangeAxisChanged" << " begin");

   fIsUpdating = kTRUE;

   // get axis ranges for upper and lower 
   Double_t upFirst = fH1->GetXaxis()->GetBinLowEdge(fH1->GetXaxis()->GetFirst());
   Double_t upLast  = fH1->GetXaxis()->GetBinUpEdge(fH1->GetXaxis()->GetLast());

   Double_t lowFirst = fRatioGraph->GetXaxis()->GetBinLowEdge(fRatioGraph->GetXaxis()->GetFirst());
   Double_t lowLast  = fRatioGraph->GetXaxis()->GetBinUpEdge(fRatioGraph->GetXaxis()->GetLast());

   Double_t globFirst = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t globLast = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   Bool_t upChanged = kFALSE;
   Bool_t lowChanged = kFALSE;

   // determine which one has changed
   if (upFirst != globFirst || upLast != globLast) {
      fSharedXAxis->SetRangeUser(upFirst, upLast);
      upChanged = kTRUE;
   }
   else if (lowFirst != globFirst || lowLast != globLast) {
      fSharedXAxis->SetRangeUser(lowFirst, lowLast);
      lowChanged = kTRUE;
   }

   if (upChanged || lowChanged) {
      SyncAxesRanges();
      CreateVisualAxes();

      // @TODO: Fix updating, it's not working if zooming on lower axis
      fUpperPad->Modified();
      fLowerPad->Modified();
      fTopPad->Modified();
      Modified();
      fCanvas->Modified();
      fCanvas->Update();
   }

   // sync the margins in case the user has dragged one of them
   Bool_t marginsChanged = SyncPadMargins();


   if (marginsChanged) {
      SetSplitFraction(fSplitFraction);
      CreateVisualAxes();
      
      // @TODO: Fix updating, it's not working if zooming on lower axis
      fUpperPad->Modified();
      fLowerPad->Modified();
      fTopPad->Modified();
      Modified();
      fCanvas->Modified();
      fCanvas->Update();

   }


   // figure out if y axis has changed 
   Double_t upYFirst = fUpperPad->GetUymin();
   Double_t upYLast = fUpperPad->GetUymax();
   Double_t lowYFirst = fLowerPad->GetUymin();
   Double_t lowYLast = fLowerPad->GetUymax();

   

   Bool_t ychanged = (
         upYFirst != fUpYFirst
      || upYLast != fUpYLast
      || lowYFirst != fLowYFirst
      || lowYLast != fLowYLast
   );
   
   fUpYFirst = upYFirst;
   fUpYLast = upYLast;
   fLowYFirst = lowYFirst;
   fLowYLast = lowYLast;

   // recreate axes if y changed
   if (ychanged) {
      CreateVisualAxes();
   }

   fIsUpdating = kFALSE;
   //__("TRatioPlot::RangeAxisChanged" << " end");
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for the UnZoomed signal that was introduced to TAxis. 
/// Unzooms both pads
void TRatioPlot::UnZoomed()
{
   //__(__PRETTY_FUNCTION__ << " called");

   // this is what resets the range
   fSharedXAxis->SetRange(0, 0);
   SyncAxesRanges();

   // Flushing
   // @TODO: Flushing is not working all the time
   fUpperPad->Modified();
   fLowerPad->Modified();
   fTopPad->Modified();
   Modified();
   fCanvas->Modified();
   fCanvas->Update();
}

void TRatioPlot::SubPadResized() 
{
   if (fIsPadUpdating) {
      return;
   }

   fIsPadUpdating = kTRUE;

   //Float_t upxlow = fUpperPad->GetAbsXlowNDC();
   Float_t upylow = fUpperPad->GetAbsYlowNDC();
   //Float_t uph = fUpperPad->GetAbsHNDC();
   //Float_t upw = fUpperPad->GetAbsWNDC();
   //Float_t upyup = upylow + uph; 
   //Float_t upxup = upxlow + upw;

   //Float_t lowxlow = fLowerPad->GetAbsXlowNDC();
   Float_t lowylow = fLowerPad->GetAbsYlowNDC();
   Float_t lowh = fLowerPad->GetAbsHNDC();
   //Float_t loww = fLowerPad->GetAbsWNDC();
   Float_t lowyup = lowylow + lowh; 
   //Float_t lowxup = lowxlow + loww;

   Bool_t changed = kFALSE;

   if (upylow != fSplitFraction) {
      // up changed
      SetSplitFraction(upylow);
      changed = kTRUE;
   }
   else if (lowyup != fSplitFraction) {
      // low changed
      SetSplitFraction(lowyup);
      changed = kTRUE;
   }
    
   if (changed) {
      CreateVisualAxes();
   }

   fIsPadUpdating = kFALSE;

}


////////////////////////////////////////////////////////////////////////////////
/// Set the fraction of the parent pad, at which the to sub pads should meet
void TRatioPlot::SetSplitFraction(Float_t sf) {
   fSplitFraction = sf;
   fUpperPad->SetPad(0., fSplitFraction, 1., 1.);
   fLowerPad->SetPad(0., 0., 1., fSplitFraction);
}
