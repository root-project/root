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
#include "TF1.h"
#include "TPad.h"
#include "TString.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TGaxis.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TMath.h"
#include "TLine.h"
#include "TVirtualFitter.h"

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
   : TPad()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor for two histograms 

// @TODO: Class should work with stacks as well
TRatioPlot::TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, 
      Option_t *displayOption, Option_t *optH1, Option_t *optH2, Option_t *optGraph,
      Double_t c1, Double_t c2)
   : TPad(name, title, 0, 0, 1, 1),
     fUpperPad(0),
     fLowerPad(0),
     fTopPad(0),
     fH1(h1),
     fH2(h2),
     fDisplayOption(0),
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

   TString displayOptionString = TString(displayOption);

   if (displayOptionString.Contains("errprop")) {
      displayOptionString.ReplaceAll("errprop", "");
      fDisplayMode = DIVIDE_HIST;
   } else if (displayOptionString.Contains("diff")) {
      displayOptionString.ReplaceAll("diff", "");
      fDisplayMode = DIFFERENCE;   
   } else {
      fDisplayMode = DIVIDE_GRAPH;
   }

   if (displayOptionString.Contains("grid")) {
      displayOptionString.ReplaceAll("grid", "");
      fShowGridline = kTRUE;
   }
   if (displayOptionString.Contains("nogrid")) {
      displayOptionString.ReplaceAll("nogrid", "");
      fShowGridline = kFALSE;
   }
   
   fDisplayOption = displayOptionString;

   TString optH1String = TString(optH1);
   TString optH2String = TString(optH2);
   TString optGraphString = TString(optGraph);
   
   optH2String.ReplaceAll("same", "");
   optH2String.ReplaceAll("SAME", "");

   fOptH1 = optH1String;
   fOptH2 = optH2String;
   fOptGraph = optGraphString;

   // build ratio, everything is ready
   BuildRatio(c1, c2);

   // taking x axis information from h1 by cloning it x axis
   fSharedXAxis = (TAxis*)(fH1->GetXaxis()->Clone());
   fUpYaxis = (TAxis*)(fH1->GetYaxis()->Clone());
   fLowYaxis = (TAxis*)(fRatioGraph->GetYaxis()->Clone());
}


TRatioPlot::TRatioPlot(TH1* h1, const char *name, const char *title, Option_t *displayOption, Option_t *optH1,
         /*Option_t *fitOpt, */Option_t *optGraph) 
   : TPad(name, title, 0, 0, 1, 1),
     fUpperPad(0),
     fLowerPad(0),
     fTopPad(0),
     fH1(h1),
     fH2(0),
     fDisplayOption(0),
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
     fUpYaxis(0)
{
   gROOT->GetListOfCleanups()->Add(this);

   if (!fH1) {
      Warning("TRatioPlot", "Need a histogram.");
      return;
   }

   Bool_t h1IsTH1=fH1->IsA()->InheritsFrom(TH1::Class());

   if (!h1IsTH1) {
      Warning("TRatioPlot", "Need a histogram deriving from TH2 or TH3.");
      return;
   }

   TList *h1Functions = fH1->GetListOfFunctions();

   if (h1Functions->GetSize() < 1) {
      Warning("TRatioPlot", "Histogram given needs to have a (fit) function associated with it");
      return;
   }

   fParentPad = gPad;


   fDisplayMode = FIT_RESIDUAL;
   
   TString displayOptionString = TString(displayOption);
   
   // determine which error style
   if (displayOptionString.Contains("errasym")) {
      fErrorMode = ERROR_ASYMMETRIC; 
      displayOptionString.ReplaceAll("errasym", "");
   }

   if (displayOptionString.Contains("grid")) {
      displayOptionString.ReplaceAll("grid", "");
      fShowGridline = kTRUE;
   }
   if (displayOptionString.Contains("nogrid")) {
      displayOptionString.ReplaceAll("nogrid", "");
      fShowGridline = kFALSE;
   }
   
   fDisplayOption = displayOptionString;

   BuildRatio();

   fOptH1 = optH1;
   fOptGraph = optGraph;

   fSharedXAxis = (TAxis*)(fH1->GetXaxis()->Clone());
   fUpYaxis = (TAxis*)(fH1->GetYaxis()->Clone());
   fLowYaxis = (TAxis*)(fRatioGraph->GetYaxis()->Clone());
 
   //SyncAxesRanges();
   
   SetupPads();

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

   SetLogx(fParentPad->GetLogx());
   SetLogy(fParentPad->GetLogy());

   fUpperPad->SetLogy(GetLogy());
   fUpperPad->SetLogx(GetLogx());
   fLowerPad->SetLogx(GetLogx());

   SetGridx(fParentPad->GetGridx());
   SetGridy(fParentPad->GetGridy());

   fUpperPad->SetGridx(GetGridx());
   fUpperPad->SetGridy(GetGridy());
   fLowerPad->SetGridx(GetGridx());
   fLowerPad->SetGridy(GetGridy());


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
   fParentPad = gPad;

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
   fH1->GetXaxis()->SetTitleSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTitleSize(0.);

   fH1->Draw(fOptH1);
   
   if (fH2 != 0) {
      fH2->Draw(fOptH2+"same");
   }

   fLowerPad->cd();

   // hide visual axis of lower pad display
   fRatioGraph->GetXaxis()->SetTickSize(0.);
   fRatioGraph->GetXaxis()->SetLabelSize(0.);
   fRatioGraph->GetXaxis()->SetTitleSize(0.);
   fRatioGraph->GetYaxis()->SetTickSize(0.);
   fRatioGraph->GetYaxis()->SetLabelSize(0.);
   fRatioGraph->GetYaxis()->SetTitleSize(0.);

   // @FIXME: This causes problems with the axes, since fconfint is not read out. Multigraph?
   fConfidenceIntervals->Draw("A3");
   fConfidenceIntervals->SetLineColor(kRed); 
   fConfidenceIntervals->SetFillColor(kYellow);
   fRatioGraph->Draw(fOptGraph+"SAME");
   //fConfidenceIntervals->Print();

   // assign same axis ranges to lower pad as in upper pad
   // the visual axes will be created on paint
   SyncAxesRanges();

   if (fShowGridline) {
      if (fGridline == 0) {
         fGridline = new TLine(0, 0, 0, 0);
         fGridline->SetLineStyle(2);
         fGridline->Draw();
      }

      Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
      Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());
     
      Double_t y = 1;

      if (fDisplayMode == DIFFERENCE || fDisplayMode == FIT_RESIDUAL) {
         y = 0;
      }

      fGridline->SetX1(first);
      fGridline->SetX2(last);
      fGridline->SetY1(y);
      fGridline->SetY2(y);

   }


   //fTopPad->cd();


   padsav->cd();

}

/// Does not really do anything right now, other than call super
void TRatioPlot::Paint(Option_t *opt) {
   TPad::Paint(opt);
}

void TRatioPlot::PaintModified()
{

   //if (!IsDrawn()) return;


   
   // sync y axes
   fH1->GetYaxis()->ImportAttributes(fUpYaxis);
   fRatioGraph->GetYaxis()->ImportAttributes(fLowYaxis);

   fH1->GetXaxis()->SetTickSize(0.);
   fH1->GetXaxis()->SetLabelSize(0.);
   fH1->GetXaxis()->SetTitleSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTitleSize(0.);
   
   fRatioGraph->GetXaxis()->SetTickSize(0.);
   fRatioGraph->GetXaxis()->SetLabelSize(0.);
   fRatioGraph->GetXaxis()->SetTitleSize(0.);
   fRatioGraph->GetYaxis()->SetTickSize(0.);
   fRatioGraph->GetYaxis()->SetLabelSize(0.);
   fRatioGraph->GetYaxis()->SetTitleSize(0.);
   
   // create the visual axes
   CreateVisualAxes();
      
   TPad::PaintModified();
   
   if (fIsUpdating) fIsUpdating = kFALSE;
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

}

////////////////////////////////////////////////////////////////////////////////
/// Build the lower plot according to which constructor was called, and
/// which options were passed.
void TRatioPlot::BuildRatio(Double_t c1, Double_t c2)
{
//   __(__PRETTY_FUNCTION__ << " called");

   // Clear and delete the graph if not exists
   if (fRatioGraph != 0) {
      fRatioGraph->IsA()->Destructor(fRatioGraph);
      fRatioGraph = 0;
   }

   if (fConfidenceIntervals == 0) {
      fConfidenceIntervals = new TGraphAsymmErrors();
   }

   // Determine the divide mode and create the lower graph accordingly
   // Pass divide options given in constructor
   if (fDisplayMode == DIVIDE_GRAPH) {
      // use TGraphAsymmErrors Divide method to create
      
      TH1 *tmpH1 = (TH1*)fH1->Clone();
      TH1 *tmpH2 = (TH1*)fH2->Clone();

      tmpH1->Scale(c1);
      tmpH2->Scale(c2);

      TGraphAsymmErrors *ratioGraph = new TGraphAsymmErrors();
      ratioGraph->Divide(tmpH1, tmpH2, fDisplayOption.Data());
      fRatioGraph = ratioGraph;

      delete tmpH1;
      delete tmpH2;

   } else if (fDisplayMode == DIFFERENCE) {

      TH1 *tmpHist = (TH1*)fH1->Clone();

      tmpHist->Reset();

      tmpHist->Add(fH1, fH2, c1, -1*c2);
      fRatioGraph = new TGraphErrors(tmpHist);

      delete tmpHist;
   } else if (fDisplayMode == FIT_RESIDUAL) {
      
      TF1 *func = dynamic_cast<TF1*>(fH1->GetListOfFunctions()->At(0));
      TH1D *tmpHist = dynamic_cast<TH1D*>(fH1->Clone());
      tmpHist->Reset();

      fRatioGraph = new TGraphAsymmErrors();
      Int_t ipoint = 1;
      Int_t ipointconf = 1;



      Double_t res;
      Double_t resConfUp;
      Double_t resConfLow;
      Double_t error;
      Double_t errorConfUp;
      Double_t errorConfLow;

      TGraphErrors *uncert = new TGraphErrors();
      
      for (Int_t i=1; i<=fH1->GetNbinsX();++i) {
            uncert->SetPoint(i, fH1->GetBinCenter(i), fH1->GetBinContent(i));
      }
      (TVirtualFitter::GetFitter())->GetConfidenceIntervals(uncert);

      //uncert->Print();

      for (Int_t i=1; i<=fH1->GetNbinsX();++i) {
         Double_t val = fH1->GetBinContent(i);
         Double_t uncertX;
         Double_t uncertY;

         uncert->GetPoint(i, uncertX, uncertY);
         Double_t uncertEy = uncert->GetErrorY(i);

         //var_dump(val);
         //var_dump(uncertX);
         //var_dump(uncertY);
         //var_dump(uncertEy);

 
         if (fErrorMode == ERROR_ASYMMETRIC) {
            
            Double_t errUp = fH1->GetBinErrorUp(i);
            Double_t errLow = fH1->GetBinErrorLow(i);

            if (val - func->Eval(fH1->GetBinCenter(i)) > 0) {
               // h1 > fit
               error = errLow;
            } else {
               // h1 < fit
               error = errUp;
            }

            errorConfUp = val - (uncertY+uncertEy) > 0 ? errLow : errUp;
            errorConfLow = val - (uncertY-uncertEy) > 0 ? errLow : errUp;


         } else if (fErrorMode == ERROR_SYMMETRIC) {
            error = fH1->GetBinError(i);
            errorConfUp = error;
            errorConfLow = error;
         } else {
            Warning("TRatioPlot", "error mode is invalid");
            error = 0;
         }

         //var_dump(errorConfUp);
         //var_dump(errorConfLow);

         if (error != 0) {
            res = (fH1->GetBinContent(i)- func->Eval(fH1->GetBinCenter(i) ) ) / error;
           

            ((TGraphAsymmErrors*)fRatioGraph)->SetPoint(ipoint, fH1->GetBinCenter(i), res);
            ((TGraphAsymmErrors*)fRatioGraph)->SetPointError(ipoint,  fH1->GetBinWidth(i)/2., fH1->GetBinWidth(i)/2., 0.5, 0.5);

            ++ipoint;
         }

         if (errorConfUp != 0 && errorConfLow != 0 && error != 0) {
            // @TODO: Error band calculation correct?
            resConfUp = (val - uncertY+uncertEy) / errorConfUp - res; 
            resConfLow = res - (val - uncertY-uncertEy) / errorConfLow;
            
            fConfidenceIntervals->SetPoint(ipoint, fH1->GetBinCenter(i), res);
            fConfidenceIntervals->SetPointEYhigh(ipoint, resConfUp);
            fConfidenceIntervals->SetPointEYlow(ipoint, resConfLow);

            ++ipointconf;
         }
      

      }

      //fConfidenceIntervals->Print();

      delete uncert;
      
   } else if (fDisplayMode == DIVIDE_HIST){
      // Use TH1's Divide method
      TH1 *tmpHist = (TH1*)fH1->Clone();
      tmpHist->Reset();
      tmpHist->Divide(fH1, fH2, c1, c2, fDisplayOption.Data());
      fRatioGraph = new TGraphErrors(tmpHist);
   
      delete tmpHist;
   }

   // need to set back to "" since recreation. we don't ever want
   // title on lower graph
   fRatioGraph->SetTitle("");
}


////////////////////////////////////////////////////////////////////////////////
/// (Re-)Creates the TGAxis objects that are used for consistent display of the 
/// axes.
void TRatioPlot::CreateVisualAxes()
{
   TVirtualPad *padsav = gPad;
   fTopPad->cd();

//   __(__PRETTY_FUNCTION__ << " called");

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

   // check if gPad has the all sides axis set
   Bool_t mirroredAxes = fParentPad->GetFrameFillStyle() == 0 || GetFrameFillStyle() == 0; 

   Bool_t logx = fUpperPad->GetLogx() || fLowerPad->GetLogx();
   Bool_t uplogy = fUpperPad->GetLogy();
   Bool_t lowlogy = fLowerPad->GetLogy();

   if (uplogy) {
      
      upYFirst = TMath::Power(10, upYFirst);
      upYLast = TMath::Power(10, upYLast);
         
      if (upYFirst <= 0 || upYLast <= 0) {
         Error(__FUNCTION__, "Cannot set upper Y axis to log scale");
      }
   }

   if (lowlogy) {
      lowYFirst = TMath::Power(10, lowYFirst);
      lowYLast = TMath::Power(10, lowYLast);

      if (lowYFirst <= 0 || lowYLast <= 0) {
         Error(__FUNCTION__, "Cannot set lower Y axis to log scale");
      }

   }


   // this is different than in y, y already has pad coords converted, x not...
   if (logx) {
      if (first <= 0 || last <= 0) {
         Error(__FUNCTION__, "Cannot set X axis to log scale");
      }
   }
   
   //var_dump(logx);
   //var_dump(uplogy);
   //var_dump(lowlogy);
   //var_dump(upYFirst);
   //var_dump(upYLast);
   //var_dump(lowYFirst);
   //var_dump(lowYLast);
   //var_dump(first);
   //var_dump(last);
   //__("");

   // determine axes options to create log axes if needed
   TString xopt = "";
   if (logx) xopt.Append("G");
   TString upyopt = "";
   if (uplogy) upyopt.Append("G");
   TString lowyopt = "";
   if (lowlogy) lowyopt.Append("G");

   // only actually create them once, reuse otherwise b/c memory
   if (fUpperGXaxis == 0) {
      fUpperGXaxis = new TGaxis(0, 0, 1, 1, 0, 1, 510, "+U"+xopt);
      fUpperGXaxis->Draw();
   }

   if (fUpperGYaxis == 0) { 
      fUpperGYaxis = new TGaxis(0, 0, 1, 1, upYFirst, upYLast, 510, "S"+upyopt);
      fUpperGYaxis->Draw();   
   }   
   
   if (fLowerGXaxis == 0) { 
      fLowerGXaxis = new TGaxis(0, 0, 1, 1, first, last, 510, "+S"+xopt);
      fLowerGXaxis->Draw();
   }   
   
   if (fLowerGYaxis == 0) {
      fLowerGYaxis = new TGaxis(0, 0, 1, 1, lowYFirst, lowYLast, 510, "-S"+lowyopt);
      fLowerGYaxis->Draw();
   }

   // import infos from TAxes
   fUpperGXaxis->ImportAxisAttributes(fSharedXAxis);
   fUpperGYaxis->ImportAxisAttributes(fUpYaxis);
   fLowerGXaxis->ImportAxisAttributes(fSharedXAxis);
   fLowerGYaxis->ImportAxisAttributes(fLowYaxis);
   
   // (re)set all the axes properties to what we want them
   fUpperGXaxis->SetTitle("");

   fUpperGXaxis->SetX1(upLM);
   fUpperGXaxis->SetX2(1-upRM);
   fUpperGXaxis->SetY1(upBM*(1-sf)+sf);
   fUpperGXaxis->SetY2(upBM*(1-sf)+sf);
   fUpperGXaxis->SetWmin(first);
   fUpperGXaxis->SetWmax(last);

   fUpperGYaxis->SetX1(upLM);
   fUpperGYaxis->SetX2(upLM);
   fUpperGYaxis->SetY1(upBM*(1-sf)+sf);
   fUpperGYaxis->SetY2( (1-upTM)*(1-sf)+sf );
   fUpperGYaxis->SetWmin(upYFirst);
   fUpperGYaxis->SetWmax(upYLast);

   fLowerGXaxis->SetX1(lowLM);
   fLowerGXaxis->SetX2(1-lowRM);
   fLowerGXaxis->SetY1(lowBM*sf);
   fLowerGXaxis->SetY2(lowBM*sf);
   fLowerGXaxis->SetWmin(first);
   fLowerGXaxis->SetWmax(last);

   fLowerGYaxis->SetX1(lowLM);
   fLowerGYaxis->SetX2(lowLM);
   fLowerGYaxis->SetY1(lowBM*sf);
   fLowerGYaxis->SetY2((1-lowTM)*sf);
   fLowerGYaxis->SetWmin(lowYFirst);
   fLowerGYaxis->SetWmax(lowYLast);

   fUpperGXaxis->SetNdivisions(fSharedXAxis->GetNdivisions());
   fUpperGYaxis->SetNdivisions(fUpYaxis->GetNdivisions());
   fLowerGXaxis->SetNdivisions(fSharedXAxis->GetNdivisions());
   fLowerGYaxis->SetNdivisions(fLowYaxis->GetNdivisions());
   
   fUpperGXaxis->SetOption("+U"+xopt);
   fUpperGYaxis->SetOption("S"+upyopt);
   fLowerGXaxis->SetOption("+S"+xopt);
   fLowerGYaxis->SetOption("-S"+lowyopt);
 

   // normalize the tick sizes. y axis ticks should be consistent
   // even if their length is different
   Double_t ratio = ( (upBM-(1-upTM))*(1-sf) ) / ( (lowBM-(1-lowTM))*sf ) ;
   fUpperGXaxis->SetLabelSize(0.);
   Double_t ticksize = fUpperGYaxis->GetTickSize()*ratio;  
   fLowerGYaxis->SetTickSize(ticksize);

   
   // Create the axes on the other sides of the graphs 
   // This is steered by an option on the containing pad or self
   if (mirroredAxes) {
      
      // only actually create them once, reuse otherwise b/c memory
      if (fUpperGXaxisMirror == 0) {
         fUpperGXaxisMirror = (TGaxis*)fUpperGXaxis->Clone(); 
         fUpperGXaxisMirror->Draw(); 
      } 

      if (fUpperGYaxisMirror == 0) {
         fUpperGYaxisMirror = (TGaxis*)fUpperGYaxis->Clone(); 
         fUpperGYaxisMirror->Draw();
      }
      
      if (fLowerGXaxisMirror == 0) { 
         fLowerGXaxisMirror = (TGaxis*)fLowerGXaxis->Clone(); 
         fLowerGXaxisMirror->Draw();
      }

      if (fLowerGYaxisMirror == 0) {
         fLowerGYaxisMirror = (TGaxis*)fLowerGYaxis->Clone();
         fLowerGYaxisMirror->Draw(); 
      }

      // import attributes from shared axes
      fUpperGXaxisMirror->ImportAxisAttributes(fSharedXAxis);
      fUpperGYaxisMirror->ImportAxisAttributes(fUpYaxis);
      fLowerGXaxisMirror->ImportAxisAttributes(fSharedXAxis);
      fLowerGYaxisMirror->ImportAxisAttributes(fLowYaxis);

      // remove titles
      fUpperGXaxisMirror->SetTitle("");
      fUpperGYaxisMirror->SetTitle("");
      fLowerGXaxisMirror->SetTitle("");
      fLowerGYaxisMirror->SetTitle("");


      // move them about and set required positions
      fUpperGXaxisMirror->SetX1(upLM);
      fUpperGXaxisMirror->SetX2(1-upRM);
      fUpperGXaxisMirror->SetY1((1-upTM)*(1-sf)+sf);
      fUpperGXaxisMirror->SetY2((1-upTM)*(1-sf)+sf);
      fUpperGXaxisMirror->SetWmin(first);
      fUpperGXaxisMirror->SetWmax(last);
      
      fUpperGYaxisMirror->SetX1(1-upRM); 
      fUpperGYaxisMirror->SetX2(1-upRM); 
      fUpperGYaxisMirror->SetY1(upBM*(1-sf)+sf);
      fUpperGYaxisMirror->SetY2( (1-upTM)*(1-sf)+sf );
      fUpperGYaxisMirror->SetWmin(upYFirst);
      fUpperGYaxisMirror->SetWmax(upYLast);

      fLowerGXaxisMirror->SetX1(lowLM);
      fLowerGXaxisMirror->SetX2(1-lowRM);
      fLowerGXaxisMirror->SetY1((1-lowTM)*sf);
      fLowerGXaxisMirror->SetY2((1-lowTM)*sf); 
      fLowerGXaxisMirror->SetWmin(first);
      fLowerGXaxisMirror->SetWmax(last);
         
      fLowerGYaxisMirror->SetX1(1-lowRM); 
      fLowerGYaxisMirror->SetX2(1-lowRM); 
      fLowerGYaxisMirror->SetY1(lowBM*sf);
      fLowerGYaxisMirror->SetY2((1-lowTM)*sf);
      fLowerGYaxisMirror->SetWmin(lowYFirst);
      fLowerGYaxisMirror->SetWmax(lowYLast);

      // also needs normalized ticksize
      fLowerGYaxisMirror->SetTickSize(ticksize);

      fUpperGXaxisMirror->SetOption("-S"+xopt);
      fUpperGYaxisMirror->SetOption("+S"+upyopt);
      fLowerGXaxisMirror->SetOption("-S"+xopt);
      fLowerGYaxisMirror->SetOption("+S"+lowyopt);

      fUpperGXaxisMirror->SetNdivisions(fSharedXAxis->GetNdivisions());
      fUpperGYaxisMirror->SetNdivisions(fUpYaxis->GetNdivisions());
      fLowerGXaxisMirror->SetNdivisions(fSharedXAxis->GetNdivisions());
      fLowerGYaxisMirror->SetNdivisions(fLowYaxis->GetNdivisions());
      
      fUpperGXaxisMirror->SetLabelSize(0.);
      fLowerGXaxisMirror->SetLabelSize(0.);
      fUpperGYaxisMirror->SetLabelSize(0.);
      fLowerGYaxisMirror->SetLabelSize(0.);
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

   // only reset margins, if any of the margins changed
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

   // only reset margins, if any of the margins changed
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
   // check if rp is already drawn.
   TList *siblings = fParentPad->GetListOfPrimitives();
   if (siblings->FindObject(this) == 0) {
      // not drawn yet
      return;
   }
 
   // Only run this concurrently once, in case it's called async
   if (fIsUpdating) {
      return;
   }

   fIsUpdating = kTRUE;

   // find out if logx has changed
   //var_dump(GetLogx());
   //var_dump(GetLogy());
   //var_dump(fUpperPad->GetLogx());
   //var_dump(fUpperPad->GetLogy());
   //var_dump(fLowerPad->GetLogx());
   //var_dump(fLowerPad->GetLogy());


   if (GetLogx()) {
      if (!fUpperPad->GetLogx() || !fLowerPad->GetLogx()) {
         SetLogx(kFALSE);
      }
   } else {
      if (fUpperPad->GetLogx() || fLowerPad->GetLogx()) {
         SetLogx(kTRUE);
      }
   }

   // set log to pad
   fUpperPad->SetLogx(GetLogx());
   fLowerPad->SetLogx(GetLogx());


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
      fUpperPad->Modified();
      fLowerPad->Modified();
      fTopPad->Modified();
      Modified();
      fCanvas->Modified();
      fCanvas->Update();
   }

	// maybe we don't need this anymore
   // figure out if y axis has changed 
   //Double_t upYFirst = fUpperPad->GetUymin();
   //Double_t upYLast = fUpperPad->GetUymax();
   //Double_t lowYFirst = fLowerPad->GetUymin();
   //Double_t lowYLast = fLowerPad->GetUymax();

   //Bool_t ychanged = (
         //upYFirst != fUpYFirst
      //|| upYLast != fUpYLast
      //|| lowYFirst != fLowYFirst
      //|| lowYLast != fLowYLast
   //);
   
   //fUpYFirst = upYFirst;
   //fUpYLast = upYLast;
   //fLowYFirst = lowYFirst;
   //fLowYLast = lowYLast;

   // recreate axes if y changed
   //if (ychanged) {
      //CreateVisualAxes();
   //}

   CreateVisualAxes();
   fIsUpdating = kFALSE;
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for the UnZoomed signal that was introduced to TAxis. 
/// Unzooms both pads
void TRatioPlot::UnZoomed()
{
   // this is what resets the range
   fSharedXAxis->SetRange(0, 0);
   SyncAxesRanges();

   // Flushing
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

   Float_t upylow = fUpperPad->GetYlowNDC();
   Float_t lowylow = fLowerPad->GetYlowNDC();
   Float_t lowh = fLowerPad->GetHNDC();
   Float_t lowyup = lowylow + lowh; 

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

Bool_t TRatioPlot::IsDrawn()
{
   TList *siblings = fParentPad->GetListOfPrimitives();
   return siblings->FindObject(this) != 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Set the fraction of the parent pad, at which the to sub pads should meet
void TRatioPlot::SetSplitFraction(Float_t sf) {
   fSplitFraction = sf;
   fUpperPad->SetPad(0., fSplitFraction, 1., 1.);
   fLowerPad->SetPad(0., 0., 1., fSplitFraction);
}
