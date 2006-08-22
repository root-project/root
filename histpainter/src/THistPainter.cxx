// @(#)root/histpainter:$Name:  $:$Id: THistPainter.cxx,v 1.262 2006/07/19 15:15:41 couet Exp $
// Author: Rene Brun   26/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "Riostream.h"
#include "TROOT.h"
#include "THistPainter.h"
#include "TH3.h"
#include "TH2.h"
#include "TF2.h"
#include "TF3.h"
#include "TPad.h"
#include "TPaveStats.h"
#include "TFrame.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TPoints.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TPainter3dAlgorithms.h"
#include "TGraphPainter.h"
#include "TGraphDelaunay.h"
#include "TView.h"
#include "TMath.h"
#include "TRandom.h"
#include "TObjArray.h"
#include "TVectorD.h"
#include "Hoption.h"
#include "Hparam.h"
#include "TPluginManager.h"
#include "TPaletteAxis.h"
#include "TCrown.h"
#include "TVirtualUtilPad.h"
#include "TEnv.h"
#include "TPoint.h"


//______________________________________________________________________________
//   The histogram painter class
//   ===========================
//
//  Histograms are drawn via the THistPainter class. Each histogram has
//  a pointer to its own painter (to be usable in a multithreaded program).
//  When the canvas has to be redrawn, the Paint function of the objects
//  in the pad is called. In case of histograms, TH1::Paint invokes directly
//  THistPainter::Paint.
//
//    See THistPainter::Paint for the list of drawing options
//    and examples.

TH1 *gCurrentHist = 0;

Hoption_t Hoption;
Hparam_t  Hparam;

const Int_t kNMAX = 2000;

const Int_t kMAXCONTOUR  = 104;
const UInt_t kCannotRotate = BIT(11);

static TString gStringEntries;
static TString gStringMean;
static TString gStringMeanX;
static TString gStringMeanY;
static TString gStringMeanZ;
static TString gStringRMS;
static TString gStringRMSX;
static TString gStringRMSY;
static TString gStringRMSZ;
static TString gStringUnderflow;
static TString gStringOverflow;
static TString gStringIntegral;
static TString gStringSkewness;
static TString gStringSkewnessX;
static TString gStringSkewnessY;
static TString gStringSkewnessZ;
static TString gStringKurtosis;
static TString gStringKurtosisX;
static TString gStringKurtosisY;
static TString gStringKurtosisZ;

ClassImp(THistPainter)

//______________________________________________________________________________
THistPainter::THistPainter()
{
   // Histogram default constructor.

   fH = 0;
   fXaxis = 0;
   fYaxis = 0;
   fZaxis = 0;
   fFunctions = 0;
   fXbuf  = 0;
   fYbuf  = 0;
   fNcuts = 0;
   fStack = 0;
   fLego  = 0;
   fGraphPainter = 0;
   fShowProjection = 0;
   fShowOption = "";
  
   gStringEntries   = gEnv->GetValue("Hist.Stats.Entries",   "Entries");
   gStringMean      = gEnv->GetValue("Hist.Stats.Mean",      "Mean");
   gStringMeanX     = gEnv->GetValue("Hist.Stats.MeanX",     "Mean x");
   gStringMeanY     = gEnv->GetValue("Hist.Stats.MeanY",     "Mean y");
   gStringMeanZ     = gEnv->GetValue("Hist.Stats.MeanZ",     "Mean z");
   gStringRMS       = gEnv->GetValue("Hist.Stats.RMS",       "RMS");
   gStringRMSX      = gEnv->GetValue("Hist.Stats.RMSX",      "RMS x");
   gStringRMSY      = gEnv->GetValue("Hist.Stats.RMSY",      "RMS y");
   gStringRMSZ      = gEnv->GetValue("Hist.Stats.RMSZ",      "RMS z");
   gStringUnderflow = gEnv->GetValue("Hist.Stats.Underflow", "Underflow");
   gStringOverflow  = gEnv->GetValue("Hist.Stats.Overflow",  "Overflow");
   gStringIntegral  = gEnv->GetValue("Hist.Stats.Integral",  "Integral");
   gStringSkewness  = gEnv->GetValue("Hist.Stats.Skewness",  "Skewness");
   gStringSkewnessX = gEnv->GetValue("Hist.Stats.SkewnessX", "Skewness x");
   gStringSkewnessY = gEnv->GetValue("Hist.Stats.SkewnessY", "Skewness y");
   gStringSkewnessZ = gEnv->GetValue("Hist.Stats.SkewnessZ", "Skewness z");
   gStringKurtosis  = gEnv->GetValue("Hist.Stats.Kurtosis",  "Kurtosis");
   gStringKurtosisX = gEnv->GetValue("Hist.Stats.KurtosisX", "Kurtosis x");
   gStringKurtosisY = gEnv->GetValue("Hist.Stats.KurtosisY", "Kurtosis y");
   gStringKurtosisZ = gEnv->GetValue("Hist.Stats.KurtosisZ", "Kurtosis z");
}

//______________________________________________________________________________
THistPainter::~THistPainter()
{
   // Histogram default destructor.
}


//______________________________________________________________________________
Int_t THistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a line.
   //
   //     Compute the closest distance of approach from point px,py to elements
   //     of an histogram.
   //     The distance is computed in pixels units.
   //
   //     Algorithm:
   //     Currently, this simple model computes the distance from the mouse
   //     to the histogram contour only.

   const Int_t big = 9999;
   const Int_t kMaxDiff = 7;

   Double_t x  = gPad->AbsPixeltoX(px);
   Double_t x1 = gPad->AbsPixeltoX(px+1);

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   Int_t curdist = big;
   Int_t yxaxis, dyaxis,xyaxis, dxaxis;
   Bool_t dsame;
   TString doption = gPad->GetPadPointer()->GetDrawOption();
   Double_t factor = 1;
   if (fH->GetNormFactor() != 0) {
      factor = fH->GetNormFactor()/fH->GetSumOfWeights();
   }
   //     return if point is not in the histogram area

   //     If a 3-D view exists, check distance to axis
   TView *view = gPad->GetView();
   Int_t d1,d2,d3;
   if (view) {
      Double_t ratio;
      d3 = view->GetDistancetoAxis(3, px, py, ratio);
      if (d3 <= kMaxDiff) {gPad->SetSelected(fZaxis); return 0;}
      d1 = view->GetDistancetoAxis(1, px, py, ratio);
      if (d1 <= kMaxDiff) {gPad->SetSelected(fXaxis); return 0;}
      d2 = view->GetDistancetoAxis(2, px, py, ratio);
      if (d2 <= kMaxDiff) {gPad->SetSelected(fYaxis); return 0;}
      if ( px > puxmin && px < puxmax && py > puymax && py < puymin) curdist = 1;
      goto FUNCTIONS;
   }
   //     check if point is close to an axis
   doption.ToLower();
   dsame = kFALSE;
   if (doption.Contains("same")) dsame = kTRUE;

   dyaxis = Int_t(2*(puymin-puymax)*fYaxis->GetLabelSize());
   if (doption.Contains("y+")) {
      xyaxis = puxmax + Int_t((puxmax-puxmin)*fYaxis->GetLabelOffset());
      if (px <= xyaxis+dyaxis && px >= xyaxis && py >puymax && py < puymin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fYaxis);
            else                    gPad->SetSelected(fXaxis);
            return 0;
         }
      }
   } else {
      xyaxis = puxmin - Int_t((puxmax-puxmin)*fYaxis->GetLabelOffset());
      if (px >= xyaxis-dyaxis && px <= xyaxis && py >puymax && py < puymin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fYaxis);
            else                    gPad->SetSelected(fXaxis);
            return 0;
         }
      }
   }

   dxaxis = Int_t((puymin-puymax)*fXaxis->GetLabelSize());
   if (doption.Contains("x+")) {
      yxaxis = puymax - Int_t((puymin-puymax)*fXaxis->GetLabelOffset());
      if (py >= yxaxis-dxaxis && py <= yxaxis && px <puxmax && px > puxmin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fXaxis);
            else                    gPad->SetSelected(fYaxis);
            return 0;
         }
      }
   } else {
      yxaxis = puymin + Int_t((puymin-puymax)*fXaxis->GetLabelOffset());
      if (yxaxis < puymin) yxaxis = puymin;
      if (py <= yxaxis+dxaxis && py >= yxaxis && px <puxmax && px > puxmin) {
         if (!dsame) {
            if (gPad->IsVertical()) gPad->SetSelected(fXaxis);
            else                    gPad->SetSelected(fYaxis);
            return 0;
         }
      }
   }

   //     if object is 2-D or 3-D return this object
   if (fH->GetDimension() == 2) {
      Int_t delta2 = 5; //Give a margin of delta2 pixels to be in the 2-d area
      if ( px > puxmin + delta2
        && px < puxmax - delta2
        && py > puymax + delta2
        && py < puymin - delta2) {curdist =1; goto FUNCTIONS;}
   }

   //     point is inside histogram area. Find channel number
   if (gPad->IsVertical()) {
      Int_t bin      = fXaxis->FindFixBin(gPad->PadtoX(x));
      Int_t binsup   = fXaxis->FindFixBin(gPad->PadtoX(x1));
      Double_t binval = factor*fH->GetBinContent(bin);
      Int_t pybin    = gPad->YtoAbsPixel(gPad->YtoPad(binval));
      if (binval == 0 && pybin < puymin) pybin = 10000;
      // special case if more than one bin for the pixel
      if (binsup-bin>1) {
         Double_t binvalmin, binvalmax;
         binvalmin=binval;
         binvalmax=binval;
         for (Int_t ibin=bin+1; ibin<binsup; ibin++) {
            Double_t binvaltmp = factor*fH->GetBinContent(ibin);
            if (binvalmin>binvaltmp) binvalmin=binvaltmp;
            if (binvalmax<binvaltmp) binvalmax=binvaltmp;
         }
         Int_t pybinmin = gPad->YtoAbsPixel(gPad->YtoPad(binvalmax));
         Int_t pybinmax = gPad->YtoAbsPixel(gPad->YtoPad(binvalmin));
         if (py<pybinmax+kMaxDiff/2 && py>pybinmin-kMaxDiff/2) pybin = py;
      }
      if (TMath::Abs(py - pybin) <= kMaxDiff) return TMath::Abs(py - pybin);
   } else {
      Double_t y  = gPad->AbsPixeltoY(py);
      Double_t y1 = gPad->AbsPixeltoY(py+1);
      Int_t bin      = fXaxis->FindFixBin(gPad->PadtoY(y));
      Int_t binsup   = fXaxis->FindFixBin(gPad->PadtoY(y1));
      Double_t binval = factor*fH->GetBinContent(bin);
      Int_t pxbin    = gPad->XtoAbsPixel(gPad->XtoPad(binval));
      if (binval == 0 && pxbin > puxmin) pxbin = 10000;
      // special case if more than one bin for the pixel
      if (binsup-bin>1) {
         Double_t binvalmin, binvalmax;
         binvalmin=binval;
         binvalmax=binval;
         for (Int_t ibin=bin+1; ibin<binsup; ibin++) {
            Double_t binvaltmp = factor*fH->GetBinContent(ibin);
            if (binvalmin>binvaltmp) binvalmin=binvaltmp;
            if (binvalmax<binvaltmp) binvalmax=binvaltmp;
         }
         Int_t pxbinmin = gPad->XtoAbsPixel(gPad->XtoPad(binvalmax));
         Int_t pxbinmax = gPad->XtoAbsPixel(gPad->XtoPad(binvalmin));
         if (px<pxbinmax+kMaxDiff/2 && px>pxbinmin-kMaxDiff/2) pxbin = px;
      }
      if (TMath::Abs(px - pxbin) <= kMaxDiff) return TMath::Abs(px - pxbin);
   }
   //     Loop on the list of associated functions and user objects
FUNCTIONS:
   TObject *f;
   TIter   next(fFunctions);
   while ((f = (TObject*) next())) {
      Int_t dist;
      if (f->InheritsFrom(TF1::Class())) dist = f->DistancetoPrimitive(-px,py);
      else                               dist = f->DistancetoPrimitive(px,py);
      if (dist < kMaxDiff) {gPad->SetSelected(f); return dist;}
   }
   return curdist;
}

//______________________________________________________________________________
void THistPainter::DrawPanel()
{
   // Display a panel with all histogram drawing options.
   //
   //      See class TDrawPanelHist for example

   gCurrentHist = fH;
   if (!gPad) {
      Error("DrawPanel", "need to draw histogram first");
      return;
   }

   //The pad utility manager is required (a plugin)
   TVirtualUtilPad *util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
   if (!util) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilPad"))) {
         if (h->LoadPlugin() == -1) return;
         h->ExecPlugin(0);
         util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
      }
   }
   util->DrawPanel(gPad,fH);
}

//______________________________________________________________________________
void THistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //     This member function is called when a histogram is clicked with the locator
   //
   //     If Left button clicked on the bin top value, then the content of this bin
   //     is modified according to the new position of the mouse when it is released.

   static Int_t bin, px1, py1, px2, py2, pyold;
   Double_t xlow, xup, ylow, binval, x, baroffset, barwidth, binwidth;

   if (!gPad->IsEditable()) return;

   //     come here if we have a lego/surface in the pad
   TView *view = gPad->GetView();
   if (!fShowProjection && view && view->TestBit(kCannotRotate) == 0) {
      view->ExecuteRotateView(event, px, py);
      return;
   }

   Double_t factor = 1;
   if (fH->GetNormFactor() != 0) {
      factor = fH->GetNormFactor()/fH->GetSumOfWeights();
   }

   switch (event) {

   case kButton1Down:

      gVirtualX->SetLineColor(-1);
      fH->TAttLine::Modify();

      // No break !!!

   case kMouseMotion:

      if (fShowProjection) {ShowProjection3(px,py); break;}
     
      if (Hoption.Bar) {
         baroffset = fH->GetBarOffset();
         barwidth  = fH->GetBarWidth();
      } else {
         baroffset = 0;
         barwidth  = 1;
      }
      x        = gPad->AbsPixeltoX(px);
      bin      = fXaxis->FindFixBin(gPad->PadtoX(x));
      binwidth = fH->GetBinWidth(bin);
      xlow     = gPad->XtoPad(fH->GetBinLowEdge(bin) + baroffset*binwidth);
      xup      = gPad->XtoPad(xlow + barwidth*binwidth);
      ylow     = gPad->GetUymin();
      px1      = gPad->XtoAbsPixel(xlow);
      px2      = gPad->XtoAbsPixel(xup);
      py1      = gPad->YtoAbsPixel(ylow);
      py2      = py;
      pyold    = py;
      if (gROOT->GetEditHistograms()) gPad->SetCursor(kArrowVer);
      else                            gPad->SetCursor(kPointer);

      break;

   case kButton1Motion:

      if (gROOT->GetEditHistograms()) {
         gVirtualX->DrawBox(px1, py1, px2, py2,TVirtualX::kHollow);  //    Draw the old box
         py2 += py - pyold;
         gVirtualX->DrawBox(px1, py1, px2, py2,TVirtualX::kHollow);  //    Draw the new box
         pyold = py;
      }

      break;

   case kButton1Up:

      if (gROOT->GetEditHistograms()) {
         binval = gPad->PadtoY(gPad->AbsPixeltoY(py2))/factor;
         fH->SetBinContent(bin,binval);
         PaintInit();   // recalculate Hparam structure and recalculate range
      }

      // might resize pad pixmap so should be called before any paint routine
      RecalculateRange();

      gPad->Modified(kTRUE);
      gVirtualX->SetLineColor(-1);

      break;

   case kButton1Locate:

      ExecuteEvent(kButton1Down, px, py);

      while (1) {
         px = py = 0;
         event = gVirtualX->RequestLocator(1, 1, px, py);

         ExecuteEvent(kButton1Motion, px, py);

         if (event != -1) {                     // button is released
            ExecuteEvent(kButton1Up, px, py);
            return;
         }
      }
   }
}

//______________________________________________________________________________
void THistPainter::FitPanel()
{
   // Display a panel with all histogram fit options
   //
   //      See class TFitPanel for example

   gCurrentHist = fH;
   if (!gPad) {
      Error("FitPanel", "need to draw histogram first");
      return;
   }

   //The pad utility manager is required (a plugin)
   TVirtualUtilPad *util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
   if (!util) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilPad"))) {
         if (h->LoadPlugin() == -1) return;
         h->ExecPlugin(0);
         util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
      }
   }
   util->FitPanel(gPad,fH);
}

//______________________________________________________________________________
TList *THistPainter::GetContourList(Double_t contour) const
{
   // Get a contour (as a list of TGraphs) using the Delaunay triangulation

   TGraphDelaunay *dt;

   // Check if fH contains a TGraphDelaunay
   TList *hl = fH->GetListOfFunctions();
   dt = (TGraphDelaunay*)hl->FindObject("TGraphDelaunay");
   if (!dt) return 0;

   gCurrentHist = fH;

   if (!fGraphPainter) ((THistPainter*)this)->fGraphPainter = new TGraphPainter(dt);

   return fGraphPainter->GetContourList(contour);
}

//______________________________________________________________________________
char *THistPainter::GetObjectInfo(Int_t px, Int_t py) const
{
   //   Redefines TObject::GetObjectInfo.
   //   Displays the histogram info (bin number, contents, integral up to bin
   //   corresponding to cursor position px,py

   if (!gPad) return (char*)"";
   static char info[64];
   Double_t x  = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y  = gPad->PadtoY(gPad->AbsPixeltoY(py));
   Double_t x1 = gPad->PadtoX(gPad->AbsPixeltoX(px+1));
   const char *drawOption = fH->GetDrawOption();
   Double_t xmin, xmax, uxmin,uxmax;
   Double_t ymin, ymax, uymin,uymax;
   if (fH->GetDimension() == 2) {
      if (gPad->GetView() || strncmp(drawOption,"cont",4) == 0
                          || strncmp(drawOption,"CONT",4) == 0) {
         uxmin=gPad->GetUxmin();
         uxmax=gPad->GetUxmax();
         xmin = fXaxis->GetBinLowEdge(fXaxis->GetFirst());
         xmax = fXaxis->GetBinUpEdge(fXaxis->GetLast());
         x = xmin +(xmax-xmin)*(x-uxmin)/(uxmax-uxmin);
         uymin=gPad->GetUymin();
         uymax=gPad->GetUymax();
         ymin = fYaxis->GetBinLowEdge(fYaxis->GetFirst());
         ymax = fYaxis->GetBinUpEdge(fYaxis->GetLast());
         y = ymin +(ymax-ymin)*(y-uymin)/(uymax-uymin);
      }
   }
   Int_t binx,biny,binmin,binx1;
   if (gPad->IsVertical()) {
      binx   = fXaxis->FindFixBin(x);
      binmin = fXaxis->GetFirst();
      binx1  = fXaxis->FindFixBin(x1);
      // special case if more than 1 bin in x per pixel
      if (binx1-binx>1 && fH->GetDimension() == 1) {
         Double_t binval=fH->GetBinContent(binx);
         Int_t binnear=binx;
         for (Int_t ibin=binx+1; ibin<binx1; ibin++) {
            Double_t binvaltmp = fH->GetBinContent(ibin);
            if (TMath::Abs(y-binvaltmp) < TMath::Abs(y-binval)) {
               binval=binvaltmp;
               binnear=ibin;
            }
         }
         binx = binnear;
      }
   } else {
      x1 = gPad->PadtoY(gPad->AbsPixeltoY(py+1));
      binx   = fXaxis->FindFixBin(y);
      binmin = fXaxis->GetFirst();
      binx1  = fXaxis->FindFixBin(x1);
      // special case if more than 1 bin in x per pixel
      if (binx1-binx>1 && fH->GetDimension() == 1) {
         Double_t binval=fH->GetBinContent(binx);
         Int_t binnear=binx;
         for (Int_t ibin=binx+1; ibin<binx1; ibin++) {
            Double_t binvaltmp = fH->GetBinContent(ibin);
            if (TMath::Abs(x-binvaltmp) < TMath::Abs(x-binval)) {
               binval=binvaltmp;
               binnear=ibin;
            }
         }
         binx = binnear;
      }
   }
   if (fH->GetDimension() == 1) {
      Double_t integ = 0;
      for (Int_t bin=binmin;bin<=binx;bin++) {integ += fH->GetBinContent(bin);}
      sprintf(info,"(x=%g, y=%g, binx=%d, binc=%g, Sum=%g)",x,y,binx,fH->GetBinContent(binx),integ);
   } else {
      biny = fYaxis->FindFixBin(y);
      sprintf(info,"(x=%g, y=%g, binx=%d, biny=%d, binc=%g)",x,y,binx,biny,fH->GetCellContent(binx,biny));
   }
   return info;
}

//______________________________________________________________________________
Bool_t THistPainter::IsInside(Int_t ix, Int_t iy)
{
   // return kTRUE if the cell ix, iy is inside one of the graphical cuts

   for (Int_t i=0;i<fNcuts;i++) {
      Double_t x = fXaxis->GetBinCenter(ix);
      Double_t y = fYaxis->GetBinCenter(iy);
      if (fCutsOpt[i] > 0) {
         if (!fCuts[i]->IsInside(x,y)) return kFALSE;
      } else {
         if (fCuts[i]->IsInside(x,y))  return kFALSE;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t THistPainter::IsInside(Double_t x, Double_t y)
{
   // return kTRUE if the point x,y is inside one of the graphical cuts

   for (Int_t i=0;i<fNcuts;i++) {
      if (fCutsOpt[i] > 0) {
         if (!fCuts[i]->IsInside(x,y)) return kFALSE;
      } else {
         if (fCuts[i]->IsInside(x,y))  return kFALSE;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Int_t THistPainter::MakeChopt(Option_t *choptin)
{
   // Decode string chopt and fill Hoption structure.

   char *l;
   char chopt[128];
   Int_t nch = strlen(choptin);
   strcpy(chopt,choptin);

   Hoption.Axis = Hoption.Bar    = Hoption.Curve   = Hoption.Error = 0;
   Hoption.Hist = Hoption.Line   = Hoption.Mark    = Hoption.Fill  = 0;
   Hoption.Same = Hoption.Func   = Hoption.Plus    = Hoption.Scat  = 0;
   Hoption.Star = Hoption.Arrow  = Hoption.Box     = Hoption.Text  = 0;
   Hoption.Char = Hoption.Color  = Hoption.Contour = Hoption.Logx  = 0;
   Hoption.Logy = Hoption.Logz   = Hoption.Lego    = Hoption.Surf  = 0;
   Hoption.Off  = Hoption.Tri    = Hoption.Proj    = Hoption.AxisPos = 0;

   //    special 2-D options
   Hoption.List     = 0;
   Hoption.Zscale   = 0;
   Hoption.FrontBox = 1;
   Hoption.BackBox  = 1;
   Hoption.System   = kCARTESIAN;

   Hoption.HighRes  = 0;

   //check for graphical cuts
   MakeCuts(chopt);

   for (Int_t i=0;i<nch;i++) chopt[i] = toupper(chopt[i]);
   if (fH->GetDimension() > 1) Hoption.Scat = 1;
   if (!nch) Hoption.Hist = 1;
   if (fFunctions->First()) Hoption.Func = 2;
   if (fH->GetSumw2N() && fH->GetDimension() == 1) Hoption.Error = 2;

   l = strstr(chopt,"GL");
   if (l) {
      strncpy(l,"  ",2);
   }
   l = strstr(chopt,"X+");
   if (l) {
      Hoption.AxisPos = 10;
      strncpy(l,"  ",2);
   }
   l = strstr(chopt,"Y+");
   if (l) {
      Hoption.AxisPos += 1;
      strncpy(l,"  ",2);
   }
   if((Hoption.AxisPos == 10 || Hoption.AxisPos == 1) && (nch == 2)) Hoption.Hist = 1;
   if(Hoption.AxisPos == 11 && nch == 4) Hoption.Hist = 1;

   l = strstr(chopt,"SAMES");
   if (l) {
      if (nch == 5) Hoption.Hist = 1;
      Hoption.Same = 2;
      strncpy(l,"     ",5);
   }
   l = strstr(chopt,"SAME");
   if (l) {
      if (nch == 4) Hoption.Hist = 1;
      Hoption.Same = 1;
      strncpy(l,"    ",4);
   }

   l = strstr(chopt,"LEGO");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Lego = 1; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Lego = 11; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Lego = 12; l[4] = ' '; }
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"SURF");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Surf = 1; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Surf = 11; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Surf = 12; l[4] = ' '; }
      if (l[4] == '3') { Hoption.Surf = 13; l[4] = ' '; }
      if (l[4] == '4') { Hoption.Surf = 14; l[4] = ' '; }
      if (l[4] == '5') { Hoption.Surf = 15; l[4] = ' '; }
      if (l[4] == '6') { Hoption.Surf = 16; l[4] = ' '; }
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"TF3");
   if (l) {
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"ISO");
   if (l) {
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"TRI");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Tri = 1; strncpy(l,"   ",3);
      l = strstr(chopt,"FB");   if (l) { Hoption.FrontBox = 0; strncpy(l,"  ",2); }
      l = strstr(chopt,"BB");   if (l) { Hoption.BackBox = 0;  strncpy(l,"  ",2); }
   }

   l = strstr(chopt,"LIST");    if (l) { Hoption.List = 1;  strncpy(l,"    ",4);}

   l = strstr(chopt,"CONT");
   if (l) {
      Hoption.Scat = 0;
      Hoption.Contour = 1; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Contour = 11; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Contour = 12; l[4] = ' '; }
      if (l[4] == '3') { Hoption.Contour = 13; l[4] = ' '; }
      if (l[4] == '4') { Hoption.Contour = 14; l[4] = ' '; }
      if (l[4] == '5') { Hoption.Contour = 15; l[4] = ' '; }
   }
   l = strstr(chopt,"HBAR");
   if (l) {
      Hoption.Hist = 0;
      Hoption.Bar = 20; strncpy(l,"    ",4);
      if (l[4] == '1') { Hoption.Bar = 21; l[4] = ' '; }
      if (l[4] == '2') { Hoption.Bar = 22; l[4] = ' '; }
      if (l[4] == '3') { Hoption.Bar = 23; l[4] = ' '; }
      if (l[4] == '4') { Hoption.Bar = 24; l[4] = ' '; }
   }
   l = strstr(chopt,"BAR");
   if (l) {
      Hoption.Hist = 0;
      Hoption.Bar = 10; strncpy(l,"   ",3);
      if (l[3] == '1') { Hoption.Bar = 11; l[3] = ' '; }
      if (l[3] == '2') { Hoption.Bar = 12; l[3] = ' '; }
      if (l[3] == '3') { Hoption.Bar = 13; l[3] = ' '; }
      if (l[3] == '4') { Hoption.Bar = 14; l[3] = ' '; }
   }
   l = strstr(chopt,"+-");   if (l) { Hoption.Plus = 2; strncpy(l,"  ",2); }
   l = strstr(chopt,"-+");   if (l) { Hoption.Plus = 2; strncpy(l,"  ",2); }

   l = strstr(chopt,"ARR" ); if (l) { Hoption.Arrow  = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"BOX" );
   if (l) {
      Hoption.Scat = 0;
      Hoption.Box  = 1; strncpy(l,"   ", 3);
      if (l[3] == '1') { Hoption.Box = 11; l[3] = ' '; }
   }
   l = strstr(chopt,"COLZ"); if (l) { Hoption.Color  = 2; strncpy(l,"    ",4); Hoption.Scat = 0; Hoption.Zscale = 1;}
   l = strstr(chopt,"COL" ); if (l) { Hoption.Color  = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"CHAR"); if (l) { Hoption.Char   = 1; strncpy(l,"    ",4); Hoption.Scat = 0; }
   l = strstr(chopt,"FUNC"); if (l) { Hoption.Func   = 2; strncpy(l,"    ",4); Hoption.Hist = 0; }
   l = strstr(chopt,"HIST"); if (l) { Hoption.Hist   = 2; strncpy(l,"    ",4); Hoption.Func = 0; Hoption.Error = 0;}
   l = strstr(chopt,"AXIS"); if (l) { Hoption.Axis   = 1; strncpy(l,"    ",4); }
   l = strstr(chopt,"AXIG"); if (l) { Hoption.Axis   = 2; strncpy(l,"    ",4); }
   l = strstr(chopt,"SCAT"); if (l) { Hoption.Scat   = 1; strncpy(l,"    ",4); }
   l = strstr(chopt,"TEXT");
   if (l) {
      Int_t angle;
      if (sscanf(&l[4],"%d",&angle) > 0) {
         if (angle < 0)  angle=0;
         if (angle > 90) angle=90;
         Hoption.Text = 1000+angle;
      } else {
         Hoption.Text = 1;
      }
      strncpy(l,"    ",4);
      Hoption.Scat = 0;
   }
   l = strstr(chopt,"POL");  if (l) { Hoption.System = kPOLAR;       strncpy(l,"   ",3); }
   l = strstr(chopt,"CYL");  if (l) { Hoption.System = kCYLINDRICAL; strncpy(l,"   ",3); }
   l = strstr(chopt,"SPH");  if (l) { Hoption.System = kSPHERICAL;   strncpy(l,"   ",3); }
   l = strstr(chopt,"PSR");  if (l) { Hoption.System = kRAPIDITY;    strncpy(l,"   ",3); }

   l = strstr(chopt,"AITOFF");
   if (l) {
      Hoption.Proj = 1; strncpy(l,"     ",6);       //Aitoff projection
   }
   l = strstr(chopt,"MERCATOR");
   if (l) {
      Hoption.Proj = 2; strncpy(l,"       ",8);     //Mercator projection
   }
   l = strstr(chopt,"SINUSOIDAL");
   if (l) {
      Hoption.Proj = 3; strncpy(l,"         ",10);  //Sinusoidal projection
   }
   l = strstr(chopt,"PARABOLIC");
   if (l) {
      Hoption.Proj = 4; strncpy(l,"        ",9);    //Parabolic projection
   }
   if (Hoption.Proj > 0) {
      Hoption.Scat = 0;
      Hoption.Contour = 14;
   }

   if (strstr(chopt,"A"))   Hoption.Axis = -1;
   if (strstr(chopt,"B"))   Hoption.Bar  = 1;
   if (strstr(chopt,"C")) { Hoption.Curve =1; Hoption.Hist = -1;}
   if (strstr(chopt,"F"))   Hoption.Fill =1;
   if (strstr(chopt,"][")) {Hoption.Off  =1; Hoption.Hist =1;}
   if (strstr(chopt,"F2"))  Hoption.Fill =2;
   if (strstr(chopt,"L")) { Hoption.Line =1; Hoption.Hist = -1;}
   if (strstr(chopt,"P")) { Hoption.Mark =1; Hoption.Hist = -1;}
   if (strstr(chopt,"Z"))   Hoption.Zscale =1;
   if (strstr(chopt,"*"))   Hoption.Star =1;
   if (strstr(chopt,"+"))   Hoption.Plus =1;
   if (strstr(chopt,"-"))   Hoption.Plus =-1;
   if (strstr(chopt,"H"))   Hoption.Hist =2;
   if (strstr(chopt,"P0"))  Hoption.Mark =10;
   if (strstr(chopt,"E")) {
      if (fH->GetDimension() == 1) {
         Hoption.Error = 1;
         if (strstr(chopt,"E0"))  Hoption.Error = 10;
         if (strstr(chopt,"E1"))  Hoption.Error = 11;
         if (strstr(chopt,"E2"))  Hoption.Error = 12;
         if (strstr(chopt,"E3"))  Hoption.Error = 13;
         if (strstr(chopt,"E4"))  Hoption.Error = 14;
         if (strstr(chopt,"X0") && Hoption.Error == 1)  Hoption.Error += 20; 
      } else {
         if (Hoption.Error == 0) {
            Hoption.Error = 100;
            Hoption.Scat  = 0;
         }
      }
      if (strstr(chopt,"X0"))  Hoption.Error += 10; 
   }

   if (strstr(chopt,"9"))  Hoption.HighRes = 1;

   if (Hoption.Surf == 15) {
      if (Hoption.System == kPOLAR || Hoption.System == kCARTESIAN) {
         Hoption.Surf = 13;
         Warning("MakeChopt","option SURF5 is not supported in Cartesian and Polar modes");
      }
   }

   //      Copy options from current style
   Hoption.Logx = gPad->GetLogx();
   Hoption.Logy = gPad->GetLogy();
   Hoption.Logz = gPad->GetLogz();

   //       Check options incompatibilities
   if (Hoption.Bar  == 1) Hoption.Hist = -1;
   if (Hoption.Same && Hoption.Plus) {
      Error("MakeChopt", "select only one of the options S,+");
      return 0;
   }
   if (Hoption.Plus) {
      if (Hoption.Line || Hoption.Curve || Hoption.Text || Hoption.Mark) {
         Error("MakeChopt", "options L,C,T,P are incompatible with options U and K");
         if (Hoption.Hist && Hoption.Bar) return 0;
      }
   }
   if (Hoption.Error || Hoption.Func || Hoption.Star) {
      if (Hoption.Plus) {
         Error("MakeChopt", "U, + options incompatible with errors/function");
         return 0;
      }
   }
   return 1;
}

//______________________________________________________________________________
Int_t THistPainter::MakeCuts(char *choptin)
{
   // Decode string chopt and fill Graphical cuts structure.

   fNcuts = 0;
   char *left = (char*)strchr(choptin,'[');
   if (!left) return 0;
   char *right = (char*)strchr(choptin,']');
   if (!right) return 0;
   Int_t nch = right-left;
   if (nch < 2) return 0;
   char *cuts = left+1;
   *right = 0;
   char *comma, *minus;
   Int_t i;
   while(1) {
      comma = strchr(cuts,',');
      if (comma) *comma = 0;
      minus = strchr(cuts,'-');
      if (minus) cuts = minus+1;
      while (*cuts == ' ') cuts++;
      Int_t nc = strlen(cuts);
      while (cuts[nc-1] == ' ') {cuts[nc-1] = 0; nc--;}
      TIter next(gROOT->GetListOfSpecials());
      TCutG *cut=0;
      TObject *obj;
      while ((obj = next())) {
         if (!obj->InheritsFrom(TCutG::Class())) continue;
         if (strcmp(obj->GetName(),cuts)) continue;
         cut = (TCutG*)obj;
         break;
      }
      if (cut) {
         fCuts[fNcuts] = cut;
         fCutsOpt[fNcuts] = 1;
         if (minus) fCutsOpt[fNcuts] = -1;
         fNcuts++;
      }
      if (!comma) break;
      cuts = comma+1;
   }
   for (i=0;i<=nch;i++) left[i] = ' ';
   return fNcuts;
}

//______________________________________________________________________________
void THistPainter::Paint(Option_t *option)
{
   // Control routine to paint any kind of histograms.
   //
   // When you call the Draw method of a histogram for the first time (TH1::Draw),
   // it creates a THistPainter object and saves a pointer to painter as a
   // data member of the histogram.
   // The THistPainter class specializes in the drawing of histograms. It is
   // separate from the histogram so that one can have histograms without
   // the graphics overhead, for example in a batch program. The choice
   // to give each histogram have its own painter rather than a central
   // singleton painter, allows two histograms to be drawn in two threads
   // without overwriting the painter's values.
   //
   // When a displayed histogram is filled again you do not have to call the Draw
   // method again. The image is refreshed the next time the pad is updated.
   // A pad is updated after one of these three actions:::Paint
   //   - a carriage control on the ROOT command line
   //   - a click inside the pad
   //   - a call to TPad::Update
   //
   // By default a call to TH1::Draw clears the pad of all objects before drawing the
   // new image of the histogram. You can use the "SAME" option to leave the previous
   // display intact and superimpose the new histogram. The same histogram can be
   // drawn with different graphics options in different pads.
   //
   // When a displayed histogram is deleted, its image is automatically removed from the pad.
   //
   // To create a copy of the histogram when drawing it, you can use TH1::DrawClone. This
   // will clone the histogram and allow you to change and delete the original one
   // without affecting the clone.
   //
   // Setting the Style
   // =================
   // Histograms use the current style (gStyle). When you change the current style and
   // would like to propagate the change to the histogram you can call TH1::UseCurrentStyle.
   // You will need to call UseCurrentStyle on each histogram.
   // When reading many histograms from a file and you wish to update them to the current
   // style you can use gROOT::ForceStyle and all histograms read after this call
   // will be updated to use the current style.
   //
   //  The following options are supported on all types:
   //  =================================================
   //    "AXIS"   : Draw only axis
   //    "AXIG"   : Draw only grid (if the grid is requested)
   //    "HIST"   : When an histogram has errors it is visualized by default with
   //               error bars. To visualize it without errors use the option HIST
   //               together with the required option (eg "hist same c").
   //               The "HIST" option can also be used to plot only the histogram
   //               and not the associated function(s).
   //    "SAME"   : Superimpose on previous picture in the same pad
   //    "CYL"    : Use Cylindrical coordinates. The X coordinate is mapped on
   //               the angle and the Y coordinate on the cylinder length. 
   //    "POL"    : Use Polar coordinates. The X coordinate is mapped on the
   //               angle and the Y coordinate on the radius.
   //    "SPH"    : Use Spherical coordinates. The X coordinate is mapped on the
   //               latitude and the Y coordinate on the longitude.
   //    "PSR"    : Use PseudoRapidity/Phi coordinates. The X coordinate is 
   //               mapped on Phi.
   //    "LEGO"   : Draw a lego plot with hidden line removal
   //    "LEGO1"  : Draw a lego plot with hidden surface removal
   //    "LEGO2"  : Draw a lego plot using colors to show the cell contents
   //    "SURF"   : Draw a surface plot with hidden line removal
   //    "SURF1"  : Draw a surface plot with hidden surface removal
   //    "SURF2"  : Draw a surface plot using colors to show the cell contents
   //    "SURF3"  : same as SURF with in addition a contour view drawn on the top
   //    "SURF4"  : Draw a surface using Gouraud shading
   //    "SURF5"  : Same as SURF3 but only the colored contour is drawn. Used with
   //               option CYL, SPH or PSR it allows to draw colored contours on a
   //               sphere, a cylinder or a in pseudo rapidy space. In cartesian
   //               or polar coordinates, option SURF3 is used.
   //    "X+"     : The X-axis is drawn on the top side of the plot.
   //    "Y+"     : The Y-axis is drawn on the right side of the plot.
   //
   //  The following options are supported for 1-D types:
   //    "AH"     : Draw histogram without axis. "A" can be combined with any drawing option.
   //             : For instance, "AC" draws the histogram as a smooth Curve without axis.
   //    "]["     : When this option is selected the first and last vertical lines
   //             : of the histogram are not drawn.
   //    "B"      : Bar chart option
   //    "C"      : Draw a smooth Curve througth the histogram bins
   //    "E"      : Draw error bars
   //    "E0"     : Draw error bars including bins with o contents
   //    "E1"     : Draw error bars with perpendicular lines at the edges
   //    "E2"     : Draw error bars with rectangles
   //    "E3"     : Draw a fill area througth the end points of the vertical error bars
   //    "E4"     : Draw a smoothed filled area through the end points of the error bars
   //    "X0"     : When used with one of the "E" option, it suppress the error
   //               bar along X as gStyle->SetErrorX(0) would do.
   //    "L"      : Draw a line througth the bin contents
   //    "P"      : Draw current marker at each bin except empty bins
   //    "P0"     : Draw current marker at each bin including empty bins
   //    "*H"     : Draw histogram with a * at each bin
   //    "LF2"    : Draw histogram like with option "L" but with a fill area.
   //             : Note that "L" draws also a fill area if the hist fillcolor is set
   //             : but the fill area corresponds to the histogram contour.
   //    "9"      : Force histogram to be drawn in high resolution mode.
   //             : By default, the histogram is drawn in low resolution
   //             : in case the number of bins is greater than the number of pixels
   //             : in the current pad.
   //
   //
   //  The following options are supported for 2-D types:
   //    "ARR"    : arrow mode. Shows gradient between adjacent cells
   //    "BOX"    : a box is drawn for each cell with surface proportional to the
   //               content's absolute value. A negative content is marked with a X.
   //    "BOX1"   : a button is drawn for each cell with surface proportional to 
   //               content's absolute value. A sunken button is drawn for negative values
   //               a raised one for positive.
   //    "COL"    : a box is drawn for each cell with a color scale varying with contents
   //    "COLZ"   : same as "COL". In addition the color palette is also drawn
   //    "CONT"   : Draw a contour plot (same as CONT0)
   //    "CONT0"  : Draw a contour plot using surface colors to distinguish contours
   //    "CONT1"  : Draw a contour plot using line styles to distinguish contours
   //    "CONT2"  : Draw a contour plot using the same line style for all contours
   //    "CONT3"  : Draw a contour plot using fill area colors
   //    "CONT4"  : Draw a contour plot using surface colors (SURF option at theta = 0)
   //    "CONT5"  : (TGraph2D only) Draw a contour plot using Delaunay triangles
   //    "LIST"   : Generate a list of TGraph objects for each contour
   //    "FB"     : With LEGO or SURFACE, suppress the Front-Box
   //    "BB"     : With LEGO or SURFACE, suppress the Back-Box
   //    "SCAT"   : Draw a scatter-plot (default)
   //    "TEXT"   : Draw bin contents as text (format set via gStyle->SetPaintTextFormat)
   //    "TEXTnn" : Draw bin contents as text at angle nn (0 < nn < 90)
   //    "[cutg]" : Draw only the sub-range selected by the TCutG named "cutg"
   //
   // Most options can be concatenated without spaces or commas, for example:
   //        h->Draw("E1 SAME");
   //
   // The options are not case sensitive:
   //        h->Draw("e1 same");
   //
   // The options "BOX", "COL" or "COLZ", use the color palette
   // defined in the current style (see TStyle::SeTPaletteAxis)
   //
   // The options "CONT" or "SURF" or "LEGO" have by default 20 equidistant contour
   // levels, you can change the number of levels with TH1::SetContour or
   // TStyle::SetNumberContours.
   //
   // You can also set the default drawing option with TH1::SetOption. To see the current
   // option use TH1::GetOption.
   //
   // Setting line, fill, marker, and text attributes
   // =====================================
   // The histogram classes inherit from the attribute classes:
   //    TAttLine, TAttFill, TAttMarker and TAttText.
   // See the description of these classes for the list of options.
   //
   //
   //  Setting Tick marks on the histogram axis
   //  ========================================
   // The TPad::SetTicks method specifies the type of tick marks on the axis.
   //
   // Assume tx = gPad->GetTickx() and ty = gPad->GetTicky().
   //
   //    tx = 1 ;  tick marks on top side are drawn (inside)
   //    tx = 2;   tick marks and labels on top side are drawn
   //    ty = 1;   tick marks on right side are drawn (inside)
   //    ty = 2;   tick marks and labels on right side are drawn
   // By default only the left Y axis and X bottom axis are drawn (tx = ty = 0)
   //
   // Use TPad::SetTicks(tx,ty) to set these options
   // See also The TAxis functions to set specific axis attributes.
   //
   //  In case multiple collor filled histograms are drawn on the same pad, the fill
   //  area may hide the axis tick marks. One can force a redraw of the axis
   //  over all the histograms by calling:
   //    gPad->RedrawAxis();
   //
   //
   //  Giving titles to the X, Y and Z axis
   //  =================================
   //    h->GetXaxis()->SetTitle("X axis title");
   //    h->GetYaxis()->SetTitle("Y axis title");
   //  The histogram title and the axis titles can be any TLatex string.
   //  The titles are part of the persistent histogram.
   //
   //
   //  Superimposing two histograms with different scales in the same pad
   //  ==================================================================
   //  The following script creates two histograms, the second histogram is
   //  the bins integral of the first one. It shows a procedure to
   //  draw the two histograms in the same pad and it draws the scale of
   //  the second histogram using a new vertical axis on the right side.
   //   (see also tutorial transpad.C for a variant of this example)
   //
   //   void twoscales() {
   //    TCanvas *c1 = new TCanvas("c1","hists with different scales",600,400);
   //
   //     //create/fill draw h1
   //     gStyle->SetOptStat(kFALSE);
   //     TH1F *h1 = new TH1F("h1","my histogram",100,-3,3);
   //     Int_t i;
   //     for (i=0;i<10000;i++) h1->Fill(gRandom->Gaus(0,1));
   //     h1->Draw();
   //     c1->Update();
   //
   //     //create hint1 filled with the bins integral of h1
   //     TH1F *hint1 = new TH1F("hint1","h1 bins integral",100,-3,3);
   //     Float_t sum = 0;
   //     for (i=1;i<=100;i++) {
   //        sum += h1->GetBinContent(i);
   //        hint1->SetBinContent(i,sum);
   //     }
   //
   //     //scale hint1 to the pad coordinates
   //     Float_t rightmax = 1.1*hint1->GetMaximum();
   //     Float_t scale = gPad->GetUymax()/rightmax;
   //     hint1->SetLineColor(kRed);
   //     hint1->Scale(scale);
   //     hint1->Draw("same");
   //
   //     //draw an axis on the right side
   //     TGaxis *axis = new TGaxis(gPad->GetUxmax(),gPad->GetUymin(),
   //           gPad->GetUxmax(), gPad->GetUymax(),0,rightmax,510,"+L");
   //     axis->SetLineColor(kRed);
   //     axis->SetTextColor(kRed);
   //     axis->Draw();
   // }
   //Begin_Html
   /*
   <img src="gif/twoscales.gif">
   */
   //End_Html
   //
   //
   // Statistics Display
   // ======================================
   // The type of information shown in the histogram statistics box
   //  can be selected with gStyle->SetOptStat(mode).
   //
   //  The mode has up to seven digits that can be set to on(1) or off(0).
   //
   //  mode = iourmen  (default = 0001111)
   //    n = 1;  name of histogram is printed
   //    e = 1;  number of entries printed
   //    m = 1;  mean value printed
   //    r = 1;  rms printed
   //    u = 1;  number of underflows printed
   //    o = 1;  number of overflows printed
   //    i = 1;  integral of bins printed
   //
   // For example: gStyle->SetOptStat(11);
   // displays only the name of histogram and the number of entries.
   // For example: gStyle->SetOptStat(1101);
   // displays the name of histogram, mean value and RMS.
   // WARNING: never call SetOptStat(000111); but SetOptStat(1111), 0001111 will
   //          be taken as an octal number !!
   // SetOptStat can also take any combination of letters IOURMEN as its argument.
   // For example gStyle->SetOptStat("NE"), gStyle->SetOptStat("NMR") and
   // gStyle->SetOptStat("RMEN") are equivalent to the examples above.
   //
   // When the histogram is drawn, a TPaveStats object is created and added
   // to the list of functions of the histogram. If a TPaveStats object already
   // exists in the histogram list of functions, the existing object is just
   // updated with the current histogram parameters.
   // With the option "same", the statistic box is not redrawn.
   // With the option "sames", the statistic box is drawn. If it hiddes
   // the previous statistics box, you can change its position
   // with these lines (if h is the pointer to the histogram):
   //
   //  Root > TPaveStats *st = (TPaveStats*)h->FindObject("stats")
   //  Root > st->SetX1NDC(newx1); //new x start position
   //  Root > st->SetX2NDC(newx2); //new x end position
   //
   // To change the type of information for an histogram with an existing TPaveStats
   // you should do: st->SetOptStat(mode) where mode has the same meaning than
   // when calling gStyle->SetOptStat(mode) (see above).
   //
   // You can delete the stats box for a histogram TH1* h with h->SetStats(0)
   // and activate it again with h->SetStats(1).
   //
   // Fit Statistics
   // ==============
   // You can change the statistics box to display the fit parameters with
   // the TStyle::SetOptFit(mode) method. This mode has four digits.
   // mode = pcev  (default = 0111)
   //    v = 1;  print name/values of parameters
   //    e = 1;  print errors (if e=1, v must be 1)
   //    c = 1;  print Chisquare/Number of degress of freedom
   //    p = 1;  print Probability
   //
   // For example: gStyle->SetOptFit(1011);
   // prints the fit probability, parameter names/values, and errors.
   //
   //
   //  The ERROR bars options
   //  ======================
   //   'E' default. Shows only the error bars, not a marker
   //   'E1' Small lines are drawn at the end of the error bars.
   //   'E2' Error rectangles are drawn.
   //   'E3' A filled area is drawn through the end points of the vertical error bars.
   //   '4' A smoothed filled area is drawn through the end points of the
   //       vertical error bars.
   //   'E0' Draw also bins with null contents.
   //Begin_Html
   /*
   <img src="gif/PaintErrors.gif">
   */
   //End_Html
   //
   //  The BAR options
   //  ===============
   //  When the option "bar" or "hbar" is specified, a bar chart is drawn.
   //   ----Vertical BAR chart: Options "bar","bar0","bar1","bar2","bar3","bar4"
   //  The bar is filled with the histogram fill color
   //  The left side of the bar is drawn with a light fill color
   //  The right side of the bar is drawn with a dark fill color
   //  The percentage of the bar drawn with either the light or dark color
   //    is  0 per cent for option "bar" or "bar0"
   //    is 10 per cent for option "bar1"
   //    is 20 per cent for option "bar2"
   //    is 30 per cent for option "bar3"
   //    is 40 per cent for option "bar4"
   //
   //  Use TH1::SetBarWidth to control the bar width (default is the bin width)
   //  Use TH1::SetBarOffset to control the bar offset (default is 0)
   //    See example in $ROOTSYS/tutorials/hbars.C
   //Begin_Html
   /*
   <img src="gif/PaintBar.gif">
   */
   //End_Html
   //
   //   ----Horizontal BAR chart: Options "hbar","hbar0","hbar1","hbar2","hbar3","hbar4"
   //  An horizontal bar is drawn for each bin.
   //  The bar is filled with the histogram fill color
   //  The bottom side of the bar is drawn with a light fill color
   //  The top side of the bar is drawn with a dark fill color
   //  The percentage of the bar drawn with either the light or dark color
   //    is  0 per cent for option "hbar" or "hbar0"
   //    is 10 per cent for option "hbar1"
   //    is 20 per cent for option "hbar2"
   //    is 30 per cent for option "hbar3"
   //    is 40 per cent for option "hbar4"
   //
   //  Use TH1::SetBarWidth to control the bar width (default is the bin width)
   //  Use TH1::SetBarOffset to control the bar offset (default is 0)
   //    See example in $ROOTSYS/tutorials/hbars.C
   //Begin_Html
   /*
   <img src="gif/PaintBarH.gif">
   */
   //End_Html
   //
   //  The SCATter plot option (default for 2-D histograms)
   // =======================
   //  For each cell (i,j) a number of points proportional to the cell content is drawn.
   //  A maximum of 500 points per cell are drawn. If the maximum is above 500
   //  contents are normalized to 500.
   //
   //
   //  The ARRow option.  Shows gradient between adjacent cells
   //  ================
   //    For each cell (i,j) an arrow is drawn
   //    The orientation of the arrow follows the cell gradient
   //
   //
   //
   //  The BOX option
   //  ==============
   //    For each cell (i,j) a box is drawn with surface proportional to contents
   //
   //
   //  The COLor option
   //  ==============
   //  For each cell (i,j) a box is drawn with a color proportional
   //    to the cell content.
   //    The color table used is defined in the current style (gStyle).
   //    The color palette in TStyle can be modified via TStyle::SeTPaletteAxis.
   //Begin_Html
   /*
   <img src="gif/h2_c2h.gif">
   */
   //End_Html
   //
   //
   //  The TEXT and TEXTnn Option
   //  ==========================
   //    For each bin the content is printed.
   //    The text attributes are:
   //      - text font = current TStyle font
   //      - text size = 0.02*padheight*markersize
   //      - text color= marker color
   //    "nn" is the angle used to draw the text (0 < nn < 90)
   //Begin_Html
   /*
   <img src="gif/h2_text.gif">
   */
   //End_Html
   //
   //
   //  The CONTour options
   //  ===================
   //  The following contour options are supported:
   //    "CONT"   : Draw a contour plot (same as CONT0)
   //    "CONT0"  : Draw a contour plot using surface colors to distinguish contours
   //    "CONT1"  : Draw a contour plot using line styles to distinguish contours
   //    "CONT2"  : Draw a contour plot using the same line style for all contours
   //    "CONT3"  : Draw a contour plot using fill area colors
   //    "CONT4"  : Draw a contour plot using surface colors (SURF option at theta = 0)
   //    "CONT5"  : Draw a contour plot using Delaunay triangles
   //
   // The following options select the "CONT4" option and are usefull for
   // skymaps or exposure maps. (see example in tutorial earth.C)
   //    "AITOFF"     : Draw a contour via an AITOFF projection
   //    "MERCATOR"   : Draw a contour via an Mercator projection
   //    "SINUSOIDAL" : Draw a contour via an Sinusoidal projection
   //    "PARABOLIC"  : Draw a contour via an Parabolic projection
   //
   //  The default number of contour levels is 20 equidistant levels and can
   //  be changed with TH1::SetContour or TStyle::SetNumberContours.
   //
   //  When option "LIST" is specified together with option "CONT",
   //  the points used to draw the contours are saved in the TGraph object
   //  and are accessible in the following way:
   //
   //     TObjArray *contours =
   //           gROOT->GetListOfSpecials()->FindObject("contours")
   //     Int_t ncontours = contours->GetSize();
   //     TList *list = (TList*)contours->At(i);
   //
   // Where i is a contour number, and list contains a list of TGraph objects.
   // For one given contour, more than one disjoint polyline may be generated.
   // The number of TGraphs per countour is given by list->GetSize().
   // Here we show only the case to access the first graph in the list.
   //    TGraph *gr1 = (TGraph*)list->First();
   //
   //Begin_Html
   /*
   <img src="gif/h2_cont.gif">
   */
   //End_Html
   //
   //
   //
   //  The LEGO options
   //  ================
   //    In a lego plot the cell contents are drawn as 3-d boxes, with
   //    the height of the box proportional to the cell content.
   //    A lego plot can be represented in several coordinate systems, the
   //    default system is Cartesian coordinates.
   //    Other possible coordinate systems are CYL,POL,SPH,PSR.
   //
   //    "LEGO"   : Draw a lego plot with hidden line removal
   //    "LEGO1"  : Draw a lego plot with hidden surface removal
   //    "LEGO2"  : Draw a lego plot using colors to show the cell contents
   //
   //      See TStyle::SeTPaletteAxis to change the color palette.
   //      We suggest you use palette 1 with the call
   //        gStyle->SetColorPalette(1)
   //
   //Begin_Html
   /*
   <img src="gif/h2_lego.gif">
   */
   //End_Html
   //
   //
   //   The "SURFace" options
   //   =====================
   //  In a surface plot, cell contents are represented as a mesh.
   //     The height of the mesh is proportional to the cell content.
   //
   //  A surface plot can be represented in several coordinate systems.
   //  The default is cartesian coordinates, and the other possible systems
   //  are CYL,POL,SPH,PSR.
   //
   //    "SURF"   : Draw a surface plot with hidden line removal
   //    "SURF1"  : Draw a surface plot with hidden surface removal
   //    "SURF2"  : Draw a surface plot using colors to show the cell contents
   //    "SURF3"  : same as SURF with in addition a contour view drawn on the top
   //    "SURF4"  : Draw a surface using Gouraud shading
   //
   //  The following picture uses SURF1.
   //
   //      See TStyle::SeTPaletteAxis to change the color palette.
   //      We suggest you use palette 1 with the call
   //      gStyle->SetColorPalette(1)
   //
   //Begin_Html
   /*
   <img src="gif/h2_surf.gif">
   */
   //End_Html
   //
   //
   //  Option "Z" : Adding the color palette on the right side of the pad
   //  ==================================================================
   // When this option is specified, a color palette with an axis indicating
   // the value of the corresponding color is drawn on the right side of
   // the picture. In case, not enough space is left, you can increase the size
   // of the right margin by calling TPad::SetRightMargin.
   // The attributes used to display the palette axis values are taken from
   // the Z axis of the object. For example, you can set the labels size
   // on the palette axis via hist->GetZaxis()->SetLabelSize().
   //
   //  Setting the color palette
   //  =========================
   // You can set the color palette with TStyle::SeTPaletteAxis, eg
   //
   //      gStyle->SeTPaletteAxis(ncolors,colors);
   //
   // For example the option "COL" draws a 2-D histogram with cells
   // represented by a box filled with a color index which is a function
   // of the cell content.
   // If the cell content is N, the color index used will be the color number
   // in colors[N],etc. If the maximum cell content is > ncolors, all
   // cell contents are scaled to ncolors.
   //
   // if ncolors <= 0, a default palette (see below) of 50 colors is defined.
   // This palette is recommended for pads, labels
   //
   // if ncolors == 1 && colors == 0, a pretty palette with a violet to red
   // spectrum is created. We recommend you use this palette when drawing legos,
   // surfaces or contours.
   //
   // if ncolors > 0 and colors == 0, the default palette is used
   // with a maximum of ncolors.
   //
   // The default palette defines:
   //   index  0  to  9 : shades of grey
   //   index 10  to 19 : shades of brown
   //   index 20  to 29 : shades of blue
   //   index 30  to 39 : shades of red
   //   index 40  to 49 : basic colors
   //
   //  The color numbers specified in the palette can be viewed by selecting
   //  the item "colors" in the "VIEW" menu of the canvas toolbar.
   //  The color'a red, green, and blue values can be changed via TColor::SetRGB.
   //
   //   Drawing a sub-range of a 2-D histogram; the [cutg] option
   //   =========================================================
   //   Using a TCutG object, it is possible to draw a sub-range of a 2-D histogram.
   //   One must create a graphical cut (mouse or C++) and specify the name
   //   of the cut between [] in the Draw option.
   //   For example, with a TCutG named "cutg", one can call:
   //      myhist->Draw("surf1 [cutg]");
   //   To invert the cut, it is enough to put a "-" in front of its name:
   //      myhist->Draw("surf1 [-cutg]");
   //   It is possible to apply several cuts ("," means logical AND):
   //      myhist->Draw("surf1 [cutg1,cutg2]");
   //   See a complete example in the tutorial fit2a.C. This example produces
   //   the following picture:
   //
   //Begin_Html
   /*
   <img src="gif/h2_surf_cutg.gif">
   */
   //End_Html
   //
   //   Drawing options for 3-D histograms
   //   ==================================
   //   By default a 3-d scatter plot is drawn
   //   If option "BOX" is specified, a 3-D box with a volume proportional
   //   to the cell content is drawn.

   if (fH->GetBuffer()) fH->BufferEmpty(-1);

   gPad->SetVertical(kTRUE);

   TH1 *oldhist = gCurrentHist;
   gCurrentHist = fH;
   TH1 *hsave   = fH;
   Double_t minsav = fH->GetMinimumStored();

   if (!MakeChopt(option)) return; //check options and fill Hoption structure

   fXbuf  = new Double_t[kNMAX];
   fYbuf  = new Double_t[kNMAX];
   if (fH->GetDimension() > 2) {
      PaintH3(option);
      fH->SetMinimum(minsav);
      if (Hoption.Func) {
         Hoption_t hoptsave = Hoption;
         Hparam_t  hparsave = Hparam;
         PaintFunction(option);
         SetHistogram(hsave);
         Hoption = hoptsave;
         Hparam  = hparsave;
      }
      gCurrentHist = oldhist;
      delete [] fXbuf; delete [] fYbuf;
      return;
   }
   TView *view = gPad->GetView();
   if (view) {
      if (!Hoption.Lego && !Hoption.Surf && !Hoption.Tri) {
         delete view;
         gPad->SetView(0);
      }
   }
   if (fH->GetDimension() > 1 || Hoption.Lego || Hoption.Surf) {
      // In case of 1D histogram, Z axis becomes Y axis.
      Int_t logysav=0, logzsav=0;
      if (fH->GetDimension() == 1) {
         logysav = Hoption.Logy;
         logzsav = Hoption.Logz;
         Hoption.Logz = 0;
         if (Hoption.Logy) {
            Hoption.Logz = 1;
            Hoption.Logy = 0;
         }
      }
      PaintTable(option);
      fH->SetMinimum(minsav);
      if (Hoption.Func) {
         Hoption_t hoptsave = Hoption;
         Hparam_t  hparsave = Hparam;
         PaintFunction(option);
         SetHistogram(hsave);
         Hoption = hoptsave;
         Hparam  = hparsave;
      }
      gCurrentHist = oldhist;
      delete [] fXbuf; delete [] fYbuf;
      if (fH->GetDimension() == 1) {
         Hoption.Logy = logysav;
         Hoption.Logz = logzsav;
      }
      return;
   }

   if (Hoption.Bar >= 20) {PaintBarH(option);
      delete [] fXbuf; delete [] fYbuf;
      return;
   }

   // fill Hparam structure with histo parameters
   if (!PaintInit()) {
      delete [] fXbuf; delete [] fYbuf;
      return;
   }

   //          Picture surround (if new page) and page number (if requested).
   //          Histogram surround (if not option "Same").
   PaintFrame();

   //          Paint histogram axis only
   Bool_t gridx = gPad->GetGridx();
   Bool_t gridy = gPad->GetGridy();
   if (Hoption.Axis > 0) {
      if (Hoption.Axis > 1) PaintAxis(kTRUE);  //axis with grid
      else {
         if (gridx) gPad->SetGridx(0);
         if (gridy) gPad->SetGridy(0);
         PaintAxis(kFALSE);
         if (gridx) gPad->SetGridx(1);
         if (gridy) gPad->SetGridy(1);
      }
      if (Hoption.Same ==1) Hoption.Same = 2;
      goto paintstat;
   }
   if (gridx || gridy) PaintAxis(kTRUE); //    Draw the grid only

   //          test for options BAR or HBAR
   if (Hoption.Bar >= 10) {
      PaintBar(option);
   }

   //          do not draw histogram if error bars required
   if (!Hoption.Error) {
      if (Hoption.Hist) PaintHist(option);
   }

   //         test for error bars or option E
   if (Hoption.Error) {
      PaintErrors(option);
      if (Hoption.Hist == 2) PaintHist(option);
   }

   if (Hoption.Text) PaintText(option);

   //         test for associated function
   if (Hoption.Func) {
      Hoption_t hoptsave = Hoption;
      Hparam_t  hparsave = Hparam;
      PaintFunction(option);
      SetHistogram(hsave);
      Hoption = hoptsave;
      Hparam  = hparsave;
   }

   if (gridx) gPad->SetGridx(0);
   if (gridy) gPad->SetGridy(0);
   PaintAxis(kFALSE);
   if (gridx) gPad->SetGridx(1);
   if (gridy) gPad->SetGridy(1);

   PaintTitle();  // Draw histogram title

   // Draw box with histogram statistics and/or fit parameters
paintstat:
   if (Hoption.Same != 1 && !fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
      TIter next(fFunctions);
      TObject *obj = 0;
      while ((obj = next())) {
         if (obj->InheritsFrom(TF1::Class())) break;
         obj = 0;
      }
      PaintStat(gStyle->GetOptStat(),(TF1*)obj);
   }
   fH->SetMinimum(minsav);
   gCurrentHist = oldhist;
   delete [] fXbuf; fXbuf = 0;
   delete [] fYbuf; fYbuf = 0;

}

//______________________________________________________________________________
void THistPainter::PaintArrows(Option_t *)
{
   // Control function to draw a table as an arrow plot.
   //
   //       For each cell (i,j) an arrow is drawn
   //       The orientation of the arrow follows the cell gradient
   //Begin_Html
   /*
   <img src="gif/PaintArrows.gif">
   */
   //End_Html

   Style_t linesav   = fH->GetLineStyle();
   Width_t widthsav  = fH->GetLineWidth();
   fH->SetLineStyle(1);
   fH->SetLineWidth(1);
   fH->TAttLine::Modify();

   Double_t dx, dy, si, co, anr, x1, x2, y1, y2, xc, yc, dxn, dyn;
   Int_t   ncx  = Hparam.xlast - Hparam.xfirst + 1;
   Int_t   ncy  = Hparam.ylast - Hparam.yfirst + 1;
   Double_t xrg = gPad->GetUxmin();
   Double_t yrg = gPad->GetUymin();
   Double_t xln = gPad->GetUxmax() - xrg;
   Double_t yln = gPad->GetUymax() - yrg;
   Double_t cx  = (xln/Double_t(ncx) -0.03)/2;
   Double_t cy  = (yln/Double_t(ncy) -0.03)/2;
   Double_t dn  = 1.E-30;

   for (Int_t id=1;id<=2;id++) {
      for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
         for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
            if (i == Hparam.xfirst) {
               dx = fH->GetCellContent(i+1, j) - fH->GetCellContent(i, j);
            } else if (i == Hparam.xlast) {
               dx = fH->GetCellContent(i, j) - fH->GetCellContent(i-1, j);
            } else {
               dx = 0.5*(fH->GetCellContent(i+1, j) - fH->GetCellContent(i-1, j));
            }
            if (j == Hparam.yfirst) {
               dy = fH->GetCellContent(i, j+1) - fH->GetCellContent(i, j);
            } else if (j == Hparam.ylast) {
               dy = fH->GetCellContent(i, j) - fH->GetCellContent(i, j-1);
            } else {
               dy = 0.5*(fH->GetCellContent(i, j+1) - fH->GetCellContent(i, j-1));
            }
            if (id == 1) {
               dn = TMath::Max(dn, TMath::Abs(dx));
               dn = TMath::Max(dn, TMath::Abs(dy));
            } else if (id == 2) {
               xc  = xrg + xln*(Double_t(i - Hparam.xfirst+1)-0.5)/Double_t(ncx);
               dxn = cx*dx/dn;
               x1  = xc - dxn;
               x2  = xc + dxn;
               yc  = yrg + yln*(Double_t(j - Hparam.yfirst+1)-0.5)/Double_t(ncy);
               dyn = cy*dy/dn;
               y1  = yc - dyn;
               y2  = yc + dyn;
               fXbuf[0] = x1;
               fXbuf[1] = x2;
               fYbuf[0] = y1;
               fYbuf[1] = y2;
               if (TMath::Abs(x2-x1) > 0.01 || TMath::Abs(y2-y1) > 0.01) {
                  anr = 0.005*.5*TMath::Sqrt(2/(dxn*dxn + dyn*dyn));
                  si  = anr*(dxn + dyn);
                  co  = anr*(dxn - dyn);
                  fXbuf[2] = x2 - si;
                  fYbuf[2] = y2 + co;
                  gPad->PaintPolyLine(3, fXbuf, fYbuf);
                  fXbuf[0] = x2;
                  fXbuf[1] = x2 - co;
                  fYbuf[0] = y2;
                  fYbuf[1] = y2 - si;
                  gPad->PaintPolyLine(2, fXbuf, fYbuf);
               }
               else {
                  gPad->PaintPolyLine(2, fXbuf, fYbuf);
               }
            }
         }
      }
   }

   if (Hoption.Zscale) PaintPalette();
   fH->SetLineStyle(linesav);
   fH->SetLineWidth(widthsav);
   fH->TAttLine::Modify();
}

//______________________________________________________________________________
void THistPainter::PaintAxis(Bool_t drawGridOnly)
{
   //  Draw axis (2D case) of an histogram
   //
   //   Assume tx = gPad->GetTickx() and ty = gPad->GetTicky()
   //   by default only the left Y axis and X bottom axis are drawn (tx = ty = 0)
   //    tx = 1;   tick marks on top side are drawn (inside)
   //    tx = 2;   tick marks and labels on top side are drawn
   //    ty = 1;   tick marks on right side are drawn (inside)
   //    ty = 2;   tick marks and labels on right side are drawn
   //   Use TPad::SetTicks(tx,ty) to set these options
   //
   //   If drawGridOnly is TRUE, only the grid is painted (if needed). This 
   //   allows to draw the grid and the axis separately. In THistPainter::Paint
   //   this feature is used to make sure that the grid is drawn in the background
   //   and the axis tick marks in the foreground of the pad.

   if (Hoption.Axis == -1) return;
   if (Hoption.Same && Hoption.Axis <= 0) return;

   static char chopt[10] = "";
   Double_t gridl = 0;
   Int_t ndiv, ndivx, ndivy, nx1, nx2, ndivsave;
   Int_t useHparam = 0;
   Double_t umin, umax, uminsave, umaxsave;
   Short_t xAxisPos = Hoption.AxisPos/10;
   Short_t yAxisPos = Hoption.AxisPos - 10*xAxisPos;

   Double_t axmin = gPad->GetUxmin();
   Double_t axmax = gPad->GetUxmax();
   Double_t aymin = gPad->GetUymin();
   Double_t aymax = gPad->GetUymax();
   char *cw = 0;
   TGaxis axis;

   // In case of option 'cont4' or in case of option 'same' over a 'cont4 plot' 
   // Hparam must be use for the axis limits.
   if (Hoption.Contour == 14) useHparam = 1;
   if (Hoption.Same) {
      TObject *obj;
      TIter next(gPad->GetListOfPrimitives());
      while ((obj=next())) {
         if (strstr(obj->GetDrawOption(),"cont4")) {
            useHparam = 1;
            break;
         }
      }
   }

   // Paint X axis
   ndivx = fXaxis->GetNdivisions();
   if (ndivx > 1000) {
      nx2   = ndivx/100;
      nx1   = TMath::Max(1, ndivx%100);
      ndivx = 100*nx2 + Int_t(Float_t(nx1)*gPad->GetAbsWNDC());
   }
   axis.SetTextAngle(0);
   axis.ImportAxisAttributes(fXaxis);

   chopt[0] = 0;
   strcat(chopt, "SDH");
   if (ndivx < 0) strcat(chopt, "N");
   if (gPad->GetGridx()) {
      gridl = (aymax-aymin)/(gPad->GetY2() - gPad->GetY1());
      strcat(chopt, "W");
   }

   // Define X-Axis limits
   if (Hoption.Logx) {
      strcat(chopt, "G");
      ndiv = TMath::Abs(ndivx);
      if (useHparam) {
         umin = TMath::Power(10,Hparam.xmin);
         umax = TMath::Power(10,Hparam.xmax);
      } else {
         umin = TMath::Power(10,axmin);
         umax = TMath::Power(10,axmax);
      }
   } else {
      ndiv = TMath::Abs(ndivx);
      if (useHparam) {
         umin = Hparam.xmin;
         umax = Hparam.xmax;
      } else {
         umin = axmin;
         umax = axmax;
      }
   }

   // Display axis as time
   if (fXaxis->GetTimeDisplay()) {
      strcat(chopt,"t");
      if (strlen(fXaxis->GetTimeFormatOnly()) == 0) {
         axis.SetTimeFormat(fXaxis->ChooseTimeFormat(Hparam.xmax-Hparam.xmin));
      }
   }

   // The main X axis can be on the bottom or on the top of the pad 
   Double_t xAxisYPos1, xAxisYPos2;
   if (xAxisPos == 1) {
      // Main X axis top
      xAxisYPos1 = aymax;
      xAxisYPos2 = aymin;
   } else {
      // Main X axis bottom
      xAxisYPos1 = aymin;
      xAxisYPos2 = aymax;
   }

   // Paint the main X axis (always)
   uminsave = umin;
   umaxsave = umax;
   ndivsave = ndiv;
   axis.SetOption(chopt);
   if (xAxisPos) {
      strcat(chopt, "-");
      gridl = -gridl;
   }
   axis.PaintAxis(axmin, xAxisYPos1,
                  axmax, xAxisYPos1,
                  umin, umax,  ndiv, chopt, gridl, drawGridOnly);

   // Paint additional X axis (if needed)
   if (gPad->GetTickx()) {
      if (xAxisPos) {
         cw=strstr(chopt,"-");
         *cw='z';
      } else {
         strcat(chopt, "-");
      }
      if (gPad->GetTickx() < 2) strcat(chopt, "U");
      if ((cw=strstr(chopt,"W"))) *cw='z';
      axis.SetTitle("");
      axis.PaintAxis(axmin, xAxisYPos2,
                     axmax, xAxisYPos2,
                     uminsave, umaxsave,  ndivsave, chopt, gridl, drawGridOnly);
   }

   // Paint Y axis
   ndivy = fYaxis->GetNdivisions();
   axis.ImportAxisAttributes(fYaxis);

   chopt[0] = 0;
   strcat(chopt, "SDH");
   if (ndivy < 0) strcat(chopt, "N");
   if (gPad->GetGridy()) {
      gridl = (axmax-axmin)/(gPad->GetX2() - gPad->GetX1());
      strcat(chopt, "W");
   }

   // Define Y-Axis limits
   if (Hoption.Logy) {
      strcat(chopt, "G");
      ndiv = TMath::Abs(ndivy);
      if (useHparam) {
         umin = TMath::Power(10,Hparam.ymin);
         umax = TMath::Power(10,Hparam.ymax);
      } else {
         umin = TMath::Power(10,aymin);
         umax = TMath::Power(10,aymax);
      }
   } else {
      ndiv = TMath::Abs(ndivy);
      if (useHparam) {
         umin = Hparam.ymin;
         umax = Hparam.ymax;
      } else {
         umin = aymin;
         umax = aymax;
      }
   }

   // Display axis as time
   if (fYaxis->GetTimeDisplay()) {
      strcat(chopt,"t");
      if (strlen(fYaxis->GetTimeFormatOnly()) == 0) {
         axis.SetTimeFormat(fYaxis->ChooseTimeFormat(Hparam.ymax-Hparam.ymin));
      }
   }

   // The main Y axis can be on the left or on the right of the pad 
   Double_t yAxisXPos1, yAxisXPos2;
   if (yAxisPos == 1) {
      // Main Y axis left
      yAxisXPos1 = axmax;
      yAxisXPos2 = axmin;
   } else {
      // Main Y axis right
      yAxisXPos1 = axmin;
      yAxisXPos2 = axmax;
   }

   // Paint the main Y axis (always)
   uminsave = umin;
   umaxsave = umax;
   ndivsave = ndiv;
   axis.SetOption(chopt);
   if (yAxisPos) {
      strcat(chopt, "+L");
      gridl = -gridl;
   }
   axis.PaintAxis(yAxisXPos1, aymin,
                  yAxisXPos1, aymax,
                  umin, umax,  ndiv, chopt, gridl, drawGridOnly);

   // Paint the additional Y axis (if needed)
   if (gPad->GetTicky()) {
      if (gPad->GetTicky() < 2) {
         strcat(chopt, "U");
         axis.SetTickSize(-fYaxis->GetTickLength());
      } else {
         strcat(chopt, "+L");
      }
      if ((cw=strstr(chopt,"W"))) *cw='z';
      axis.SetTitle("");
      axis.PaintAxis(yAxisXPos2, aymin,
                     yAxisXPos2, aymax,
                     uminsave, umaxsave,  ndivsave, chopt, gridl, drawGridOnly);
   }
}


//______________________________________________________________________________
void THistPainter::PaintBar(Option_t *)
{
   //  Draw a bar chart in a normal pad.
   //     (see PaintBarH to draw a bar chart in a rotated pad)
   //
   // This function is called by THistPainter::Paint when the option "bar"
   // has been specified.
   //
   // A vertical bar is drawn for each bin.
   // The bar is filled with the histogram fill color
   // The left side of the bar is drawn with a light fill color
   // The right side of the bar is drawn with a dark fill color
   // The percentage of the bar drawn with either the light or dark color
   //   is  0 per cent for option "bar" or "bar0"
   //   is 10 per cent for option "bar1"
   //   is 20 per cent for option "bar2"
   //   is 30 per cent for option "bar3"
   //   is 40 per cent for option "bar4"
   //
   // Use TH1::SetBarWidth to control the bar width (default is the bin width)
   // Use TH1::SetBarOffset to control the bar offset (default is 0)
   //   See example in $ROOTSYS/tutorials/hbars.C
   //Begin_Html
   /*
   <img src="gif/PaintBar.gif">
   */
   //End_Html

   Int_t bar = Hoption.Bar - 10;
   Double_t xmin,xmax,ymin,ymax,umin,umax,w,y;
   Double_t offset = fH->GetBarOffset();
   Double_t width  = fH->GetBarWidth();
   TBox box;
   Int_t hcolor = fH->GetFillColor();
   Int_t hstyle = fH->GetFillStyle();
   box.SetFillColor(hcolor);
   box.SetFillStyle(hstyle);
   for (Int_t bin=fXaxis->GetFirst();bin<=fXaxis->GetLast();bin++) {
      y    = fH->GetBinContent(bin);
      xmin = gPad->XtoPad(fXaxis->GetBinLowEdge(bin));
      xmax = gPad->XtoPad(fXaxis->GetBinUpEdge(bin));
      ymin = gPad->GetUymin();
      ymax = gPad->YtoPad(y);
      if (ymax < gPad->GetUymin()) continue;
      if (ymax > gPad->GetUymax()) ymax = gPad->GetUymax();
      if (ymin < gPad->GetUymin()) ymin = gPad->GetUymin();
      w    = (xmax-xmin)*width;
      xmin += offset*(xmax-xmin);
      xmax = xmin + w;
      if (bar < 1) {
         box.PaintBox(xmin,ymin,xmax,ymax);
      } else {
         umin = xmin + bar*(xmax-xmin)/10.;
         umax = xmax - bar*(xmax-xmin)/10.;
         box.SetFillColor(hcolor+150); //bright
         box.PaintBox(xmin,ymin,umin,ymax);
         box.SetFillColor(hcolor);
         box.PaintBox(umin,ymin,umax,ymax);
         box.SetFillColor(hcolor+100); //dark
         box.PaintBox(umax,ymin,xmax,ymax);
      }
   }
}

//______________________________________________________________________________
void THistPainter::PaintBarH(Option_t *)
{
   // Draw a bar char in a rotated pad (X vertical, Y horizontal)
   //     (see PaintBar to draw a bar chart in a normal pad)
   //
   // This function is called by THistPainter::Paint when the option "hbar"
   // has been specified.
   //
   // An horizontal bar is drawn for each bin.
   // The bar is filled with the histogram fill color
   // The bottom side of the bar is drawn with a light fill color
   // The top side of the bar is drawn with a dark fill color
   // The percentage of the bar drawn with either the light or dark color
   //   is  0 per cent for option "hbar" or "hbar0"
   //   is 10 per cent for option "hbar1"
   //   is 20 per cent for option "hbar2"
   //   is 30 per cent for option "hbar3"
   //   is 40 per cent for option "hbar4"
   //
   // Use TH1::SetBarWidth to control the bar width (default is the bin width)
   // Use TH1::SetBarOffset to control the bar offset (default is 0)
   //   See example in $ROOTSYS/tutorials/hbars.C
   //Begin_Html
   /*
   <img src="gif/PaintBarH.gif">
   */
   //End_Html

   gPad->SetVertical(kFALSE);

   PaintInitH();

   TAxis *xaxis = fXaxis;
   TAxis *yaxis = fYaxis;
   if (!strcmp(xaxis->GetName(),"xaxis")) {
      fXaxis = yaxis;
      fYaxis = xaxis;
   }

   PaintFrame();
   PaintAxis(kFALSE);

   Int_t bar = Hoption.Bar - 20;
   Double_t xmin,xmax,ymin,ymax,umin,umax,w;
   Double_t offset = fH->GetBarOffset();
   Double_t width  = fH->GetBarWidth();
   TBox box;
   Int_t hcolor = fH->GetFillColor();
   Int_t hstyle = fH->GetFillStyle();
   box.SetFillColor(hcolor);
   box.SetFillStyle(hstyle);
   for (Int_t bin=fYaxis->GetFirst();bin<=fYaxis->GetLast();bin++) {
      ymin = gPad->YtoPad(fYaxis->GetBinLowEdge(bin));
      ymax = gPad->YtoPad(fYaxis->GetBinUpEdge(bin));
      xmin = gPad->GetUxmin();
      xmax = gPad->XtoPad(fH->GetBinContent(bin));
      if (xmax < gPad->GetUxmin()) continue;
      if (xmax > gPad->GetUxmax()) xmax = gPad->GetUxmax();
      if (xmin < gPad->GetUxmin()) xmin = gPad->GetUxmin();
      w    = (ymax-ymin)*width;
      ymin += offset*(ymax-ymin);
      ymax = ymin + w;
      if (bar < 1) {
         box.PaintBox(xmin,ymin,xmax,ymax);
      } else {
         umin = ymin + bar*(ymax-ymin)/10.;
         umax = ymax - bar*(ymax-ymin)/10.;
         box.SetFillColor(hcolor+100); //dark
         box.PaintBox(xmin,ymin,xmax,umin);
         box.SetFillColor(hcolor);
         box.PaintBox(xmin,umin,xmax,umax);
         box.SetFillColor(hcolor+150); //bright
         box.PaintBox(xmin,umax,xmax,ymax);
      }
   }

   PaintTitle();
   fXaxis = xaxis;
   fYaxis = yaxis;
   //    Draw box with histogram statistics and/or fit parameters
   if (Hoption.Same != 1 && !fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
      TIter next(fFunctions);
      TObject *obj = 0;
      while ((obj = next())) {
         if (obj->InheritsFrom(TF1::Class())) break;
         obj = 0;
      }
      PaintStat(gStyle->GetOptStat(),(TF1*)obj);
   }
}

//______________________________________________________________________________
void THistPainter::PaintBoxes(Option_t *)
{
   // Control function to draw a table as a box plot.
   //
   // For each cell (i,j) a box is drawn.
   // The size of the box is proportional to the absolute value of the cell content.
   // The cells with a negative content draw with a X on top of the boxes.
   //Begin_Html
   /*
   <img src="gif/PaintBox.gif">
   */
   //End_Html
   // With option BOX1 a button is drawn for each cell with surface proportional to
   // content's absolute value. A sunken button is drawn for negative values
   // a raised one for positive.
   //Begin_Html
   /*
   <img src="gif/PaintBox1.gif">
   */
   //End_Html

   Style_t fillsav   = fH->GetFillStyle();
   Style_t colsav    = fH->GetFillColor();
   if (fH->GetFillColor() == 0)  fH->SetFillStyle(0);
   if (Hoption.Box == 11) fH->SetFillStyle(1001);
   fH->TAttLine::Modify();
   fH->TAttFill::Modify();

   Double_t z, xk,xstep, yk, ystep, xcent, ycent, xlow, xup, ylow, yup;
   Double_t ux1 = gPad->PixeltoX(1);
   Double_t ux0 = gPad->PixeltoX(0);
   Double_t uy1 = gPad->PixeltoY(1);
   Double_t uy0 = gPad->PixeltoY(0);
   Double_t dxmin = 0.51*(gPad->PadtoX(ux1)-gPad->PadtoX(ux0));
   Double_t dymin = 0.51*(gPad->PadtoY(uy0)-gPad->PadtoY(uy1));

   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();

   if (Hoption.Logz) {
      if (zmin > 0) {
         zmin = TMath::Log10(zmin*0.1);
         zmax = TMath::Log10(zmax);
      } else {
         return;
      }
   } else {
      zmin = 0;
      zmax = TMath::Max(TMath::Abs(zmin),TMath::Abs(zmax));
   }

   // In case of option SAME, zmin and zmax values are taken from the
   // first plotted 2D histogram.
   if (Hoption.Same) {
      TH2 *h2;
      TIter next(gPad->GetListOfPrimitives());
      while ((h2 = (TH2 *)next())) {
         if (!h2->InheritsFrom(TH2::Class())) continue;
         zmin = h2->GetMinimum();
         zmax = h2->GetMaximum();
         if (Hoption.Logz) {
            zmax = TMath::Log10(zmax);
            if (zmin <= 0) {
               zmin = TMath::Log10(zmax*0.001);
            } else {
               zmin = TMath::Log10(zmin);
            }
         }
         break;
      }
   }

   Double_t zratio, dz = zmax - zmin;
   Bool_t kZNeg        = kFALSE;

   // Define the dark and light colors the "button style" boxes.
   Color_t color = fH->GetFillColor();
   Color_t light=0, dark=0;
   if (Hoption.Box == 11) {
      if (color == 0) {
         light = 0;
         dark  = 0;
      } else if (color <= 50 && color != 0) {
         light = color + 150;
         dark  = color + 100;
      } else {
         Float_t r, g, b, h, l, s;
         TColor *c = gROOT->GetColor(color);
         if (c) c->GetRGB(r, g, b);
         else {r = 0.5; g=0.5; b=0.5;}
         TColor::RGBtoHLS(r, g, b, h, l, s);
         TColor::HLStoRGB(h, 0.7*l, s, r, g, b);
         dark = TColor::GetColor(r, g, b);
         TColor::HLStoRGB(h, 1.2*l, s, r, g, b);
         light = TColor::GetColor(r, g, b);
      }
   }

   // Loop over all the bins and draw the boxes
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      ycent = 0.5*ystep;
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         xcent = 0.5*xstep;
         z     = Hparam.factor*fH->GetBinContent(bin);
         kZNeg = kFALSE;
         if (z < 0) {
            if (Hoption.Logz) continue;
            z = -z;
            kZNeg = kTRUE;
         }
         if (Hoption.Logz) {
            if (z != 0) z = TMath::Log10(z);
            else        z = zmin;
         }
         
         if (z <  zmin) continue; //   Can be the case with 
         if (z >  zmax) z = zmax; // option Same

         if (dz == 0) continue;
         zratio = TMath::Sqrt((z-zmin)/dz);
         if (zratio == 0) continue;

         xup  = xcent*zratio + xk + xcent;
         xlow = 2*(xk + xcent) - xup;
         if (xup-xlow < dxmin) xup = xlow+dxmin;
         if (Hoption.Logx) {
            if (xup > 0)  xup  = TMath::Log10(xup);
            else continue;
            if (xlow > 0) xlow = TMath::Log10(xlow);
            else continue;
         }

         yup  = ycent*zratio + yk + ycent;
         ylow = 2*(yk + ycent) - yup;
         if (yup-ylow < dymin) yup = ylow+dymin;
         if (Hoption.Logy) {
            if (yup > 0)  yup  = TMath::Log10(yup);
            else continue;
            if (ylow > 0) ylow = TMath::Log10(ylow);
            else continue;
         }

         if (xlow < gPad->GetUxmin()) continue;
         if (ylow < gPad->GetUymin()) continue;
         if (xup  > gPad->GetUxmax()) continue;
         if (yup  > gPad->GetUymax()) continue;

         if (Hoption.Box == 1) {
            fH->SetFillColor(color);
            fH->TAttFill::Modify();
            gPad->PaintBox(xlow, ylow, xup, yup);
            if (kZNeg) {
               gPad->PaintLine(xlow, ylow, xup, yup);
               gPad->PaintLine(xlow, yup, xup, ylow);
            }
         } else if (Hoption.Box == 11) {
            // Draw the center of the box
            fH->SetFillColor(color);
            fH->TAttFill::Modify();
            gPad->PaintBox(xlow, ylow, xup, yup);

            // Draw top&left part of the box
            Double_t x[7], y[7];
            Double_t bwidth = 0.1;
            x[0] = xlow;                     y[0] = ylow;
            x[1] = xlow + bwidth*(xup-xlow); y[1] = ylow + bwidth*(yup-ylow);
            x[2] = x[1];                     y[2] = yup - bwidth*(yup-ylow);
            x[3] = xup - bwidth*(xup-xlow);  y[3] = y[2];
            x[4] = xup;                      y[4] = yup;
            x[5] = xlow;                     y[5] = yup;
            x[6] = xlow;                     y[6] = ylow;
            if (kZNeg) fH->SetFillColor(dark);
            else       fH->SetFillColor(light);
            fH->TAttFill::Modify();
            gPad->PaintFillArea(7, x, y);

            // Draw bottom&right part of the box
            x[0] = xlow;                     y[0] = ylow;
            x[1] = xlow + bwidth*(xup-xlow); y[1] = ylow + bwidth*(yup-ylow);
            x[2] = xup - bwidth*(xup-xlow);  y[2] = y[1];
            x[3] = x[2];                     y[3] = yup - bwidth*(yup-ylow);
            x[4] = xup;                      y[4] = yup;
            x[5] = xup;                      y[5] = ylow;
            x[6] = xlow;                     y[6] = ylow;
            if (kZNeg) fH->SetFillColor(light);
            else       fH->SetFillColor(dark);
            fH->TAttFill::Modify();
            gPad->PaintFillArea(7, x, y);
         }
      }
   }

   if (Hoption.Zscale) PaintPalette();
   fH->SetFillStyle(fillsav);
   fH->SetFillColor(colsav);
   fH->TAttFill::Modify();
}

//______________________________________________________________________________
void THistPainter::PaintColorLevels(Option_t *)
{
   // Control function to draw a table as a color plot.
   //
   //       For each cell (i,j) a box is drawn with a color proportional
   //       to the cell content.
   //       The color table used is defined in the current style (gStyle).
   //       The color palette in TStyle can be modified via TStyle::SeTPaletteAxis.
   //Begin_Html
   /*
   <img src="gif/PaintCol.gif">
   */
   //End_Html

   Double_t z, zc, xk, xstep, yk, ystep, xlow, xup, ylow, yup;

   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();
   if (Hoption.Logz) {
      if (zmin > 0) {
         zmin = TMath::Log10(zmin);
         zmax = TMath::Log10(zmax);
      } else {
         return;
      }
   }

   Double_t dz = zmax - zmin;

   if (dz <= 0) return;

   Style_t fillsav   = fH->GetFillStyle();
   Style_t colsav    = fH->GetFillColor();
   fH->SetFillStyle(1001);
   fH->TAttFill::Modify();

   // Initialize the levels on the Z axis
   Int_t ncolors  = gStyle->GetNumberOfColors();
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);
   Double_t scale = ndivz/dz;

   Int_t color;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      if (Hoption.System == kPOLAR && yk<0) yk= 2*TMath::Pi()+yk;
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         z     = fH->GetBinContent(bin);
         if (z == 0 && (zmin >= 0 || Hoption.Logz)) continue; // don't draw the empty bins for histograms with positive content
         if (Hoption.Logz) {
            if (z > 0) z = TMath::Log10(z);
            else       z = zmin;
         }
         if (z < zmin) continue;
         xup  = xk + xstep;
         xlow = xk;
         if (Hoption.Logx) {
            if (xup > 0)  xup  = TMath::Log10(xup);
            else continue;
            if (xlow > 0) xlow = TMath::Log10(xlow);
            else continue;
         }
         yup  = yk + ystep;
         ylow = yk;
         if (Hoption.System != kPOLAR) {
            if (Hoption.Logy) {
               if (yup > 0)  yup  = TMath::Log10(yup);
               else continue;
               if (ylow > 0) ylow = TMath::Log10(ylow);
               else continue;
            }
            if (xup  < gPad->GetUxmin()) continue;
            if (yup  < gPad->GetUymin()) continue;
            if (xlow > gPad->GetUxmax()) continue;
            if (ylow > gPad->GetUymax()) continue;
            if (xlow < gPad->GetUxmin()) xlow = gPad->GetUxmin();
            if (ylow < gPad->GetUymin()) ylow = gPad->GetUymin();
            if (xup  > gPad->GetUxmax()) xup  = gPad->GetUxmax();
            if (yup  > gPad->GetUymax()) yup  = gPad->GetUymax();
         }

         if (fH->TestBit(TH1::kUserContour)) {
            zc = fH->GetContourLevelPad(0);
            if (z < zc) continue;
            color = -1;
            for (Int_t k=0; k<ndiv; k++) {
               zc = fH->GetContourLevelPad(k);
               if (z < zc) {
                  continue;
               } else {
                  color++;
               }
            }
         } else {
            color = Int_t(0.01+(z-zmin)*scale);
         }

         Int_t theColor = Int_t((color+0.99)*Float_t(ncolors)/Float_t(ndivz));
         if (theColor > ncolors-1) theColor = ncolors-1;
         fH->SetFillColor(gStyle->GetColorPalette(theColor));
         fH->TAttFill::Modify();
         if (Hoption.System != kPOLAR) {
            gPad->PaintBox(xlow, ylow, xup, yup);
         } else  {
            TCrown crown(0,0,xlow,xup,ylow*TMath::RadToDeg(),yup*TMath::RadToDeg());
            crown.SetFillColor(gStyle->GetColorPalette(theColor));
            crown.Paint();
         }
      }
   }

   if (Hoption.Zscale) PaintPalette();

   fH->SetFillStyle(fillsav);
   fH->SetFillColor(colsav);
   fH->TAttFill::Modify();

}

//______________________________________________________________________________
void THistPainter::PaintContour(Option_t *option)
{
   // Control function to draw a table as a contour plot.
   //
   //     Hoption.Contour may have the following values:
   //        1  The contour is drawn with filled colour levels. ("cont")
   //       11  Use colour to distinguish contours. ("cont1")
   //       12  Use line style to distinguish contours. ("cont2")
   //       13  Line style and colour are the same for all contours. ("cont3")
   //       14  Same as 1 but uses the "SURF" algorithm ("cont4")
   //           see also options "AITOFF","MERCATOR",etc below
   //       15  Use Delaunay triangles to compute the contours
   //
   //     When option "List" is specified together with option "cont",
   //     the points used to draw the contours are saved in the TGraph format
   //     and are accessible in the following way:
   //    TObjArray *contours =
   //            (TObjArray*)gROOT->GetListOfSpecials()->FindObject("contours")
   //    Int_t ncontours = contours->GetSize();
   //    TList *list = (TList*)contours->At(i); //where i is a contour number
   //    list contains a list of TGraph objects. For one given contour, more than
   //    one disjoint polyline may be generated. The number of TGraphs per
   //    countour is given by list->GetSize().
   //    Here we show only the case to access the first graph in the list.
   //       TGraph *gr1 = (TGraph*)list->First();
   //
   //
   //Begin_Html
   /*
   <img src="gif/PaintContour1.gif">
   */
   //End_Html
   //
   // This function is also called when one of the options below is called;
   // skymaps or exposure maps. (see example in tutorial earth.C)
   //    "AITOFF"     : Draw a contour via an AITOFF projection
   //    "MERCATOR"   : Draw a contour via an Mercator projection
   //    "SINUSOIDAL" : Draw a contour via an Sinusoidal projection
   //    "PARABOLIC"  : Draw a contour via an Parabolic projection
   //
   // The tutorial earth.C uses these 4 options and produce the following picture:
   //
   //Begin_Html
   /*
   <img src="gif/PaintContour4Earth.gif">
   */
   //End_Html

   Int_t i, j, count, ncontour, icol, n, lj, m, ix, jx, ljfill;
   Int_t itars, mode, ir[4];
   Double_t xsave, ysave, thesave,phisave,x[4], y[4], zc[4];

   if (Hoption.Contour == 14) {
      Hoption.Surf = 12;
      Hoption.Axis = 1;
      thesave = gPad->GetTheta();
      phisave = gPad->GetPhi();
      gPad->SetPhi(0.);
      gPad->SetTheta(90.);
      PaintSurface(option);
      gPad->SetPhi(phisave);
      gPad->SetTheta(thesave);
      gPad->GetView()->SetBit(kCannotRotate); //tested in ExecuteEvent
      PaintAxis();
      return;
   }

   if (Hoption.Same) {
      // If the contour is painted on a 3d plot, the contour lines are
      // paint in 3d too.
      TObject *obj;
      TIter next(gPad->GetListOfPrimitives());
      while ((obj=next())) {
         if (strstr(obj->GetDrawOption(),"surf") ||
             strstr(obj->GetDrawOption(),"lego") ||
             strstr(obj->GetDrawOption(),"tri")) {
               Hoption.Surf = 16;
               PaintSurface(option);
               return;
         }
      }
   }

   if (Hoption.Contour == 15) {
      TGraphDelaunay *dt;
      TList *hl = fH->GetListOfFunctions();
      dt = (TGraphDelaunay*)hl->FindObject("TGraphDelaunay");
      if (!dt) return;
      if (!fGraphPainter) fGraphPainter = new TGraphPainter(dt);
      fGraphPainter->Paint(option);
      return;
   }

   gPad->SetBit(TGraph::kClipFrame);

   Double_t *levels  = new Double_t[2*kMAXCONTOUR];
   Double_t *xarr    = new Double_t[2*kMAXCONTOUR];
   Double_t *yarr    = new Double_t[2*kMAXCONTOUR];
   Int_t  *itarr     = new Int_t[2*kMAXCONTOUR];

   Int_t npmax = 0;
   for (i=0;i<2*kMAXCONTOUR;i++) itarr[i] = 0;

   ncontour = fH->GetContour();
   if (ncontour == 0) {
      ncontour = gStyle->GetNumberContours();
      fH->SetContour(ncontour);
   }
   if (ncontour > kMAXCONTOUR) {
      Warning("PaintContour", "maximum number of contours is %d, asked for %d",
              kMAXCONTOUR, ncontour);
      ncontour = kMAXCONTOUR-1;
   }
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ncontour);

   for (i=0;i<ncontour;i++) levels[i] = fH->GetContourLevelPad(i);
   //for (i=0;i<ncontour;i++)
   //   levels[i] = Hparam.zmin+(Hparam.zmax-Hparam.zmin)/ncontour*i;
   Int_t linesav   = fH->GetLineStyle();
   Int_t colorsav  = fH->GetLineColor();
   Int_t fillsav  = fH->GetFillColor();
   if (Hoption.Contour == 13) {
      fH->TAttLine::Modify();
   }

   TPolyLine **polys = 0;
   TPolyLine *poly=0;
   TObjArray *contours = 0;
   TList *list = 0;
   TGraph *graph = 0;
   Int_t *np = 0;
   if (Hoption.Contour == 1) {
      np = new Int_t[ncontour];
      for (i=0;i<ncontour;i++) np[i] = 0;
      polys = new TPolyLine*[ncontour];
      for (i=0;i<ncontour;i++) {
         polys[i] = new TPolyLine(100);
      }
      if (Hoption.List == 1) {
         contours = (TObjArray*)gROOT->GetListOfSpecials()->FindObject("contours");
         if (contours) {
            gROOT->GetListOfSpecials()->Remove(contours);
            count = contours->GetSize();
            for (i=0;i<count;i++) {
               list = (TList*)contours->At(i);
               if (list) list->Delete();
            }
         }
         contours = new TObjArray(ncontour);
         contours->SetName("contours");
         gROOT->GetListOfSpecials()->Add(contours);
         for (i=0;i<ncontour;i++) {
            list = new TList();
            contours->Add(list);
         }
      }
   }
   Int_t theColor;
   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz   = TMath::Abs(ncontour);

   Int_t k,ipoly;
   for (j=Hparam.yfirst; j<Hparam.ylast; j++) {
      y[0] = fYaxis->GetBinCenter(j);
      y[1] = y[0];
      y[2] = fYaxis->GetBinCenter(j+1);
      y[3] = y[2];
      for (i=Hparam.xfirst; i<Hparam.xlast; i++) {
         zc[0] = fH->GetBinContent(i,   j);
         zc[1] = fH->GetBinContent(i+1, j);
         zc[2] = fH->GetBinContent(i+1, j+1);
         zc[3] = fH->GetBinContent(i,   j+1);
         if (!IsInside(fXaxis->GetBinCenter(i),fYaxis->GetBinCenter(j))) continue;
         if (Hoption.Logz) {
            if (zc[0] > 0)   zc[0] = TMath::Log10(zc[0]);
            else             zc[0] = Hparam.zmin;
            if (zc[1] > 0)   zc[1] = TMath::Log10(zc[1]);
            else             zc[1] = Hparam.zmin;
            if (zc[2] > 0)   zc[2] = TMath::Log10(zc[2]);
            else             zc[2] = Hparam.zmin;
            if (zc[3] > 0)   zc[3] = TMath::Log10(zc[3]);
            else             zc[3] = Hparam.zmin;
         }
         for (k=0;k<4;k++) {
            ir[k] = TMath::BinarySearch(ncontour,levels,zc[k]);
         }
         if (ir[0] != ir[1] || ir[1] != ir[2] || ir[2] != ir[3] || ir[3] != ir[0]) {
            x[0] = fXaxis->GetBinCenter(i);
            x[3] = x[0];
            x[1] = fXaxis->GetBinCenter(i+1);
            x[2] = x[1];
            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=1;
            for (ix=1;ix<=4;ix++) {
               m = n%4 + 1;
               ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],zc[m-1],
                     ir[m-1],x[m-1],y[m-1],&xarr[lj-1],&yarr[lj-1],&itarr[lj-1], levels);
               lj += 2*ljfill;
               n = m;
            }

            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=2;
            for (ix=1;ix<=4;ix++) {
               if (n == 1) m = 4;
               else        m = n-1;
               ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],zc[m-1],
                     ir[m-1],x[m-1],y[m-1],&xarr[lj-1],&yarr[lj-1],&itarr[lj-1], levels);
               lj += 2*ljfill;
               n = m;
            }

   //     Re-order endpoints

            count = 0;
            for (ix=1; ix<=lj-5; ix +=2) {
               //count = 0;
               while (itarr[ix-1] != itarr[ix]) {
                  xsave = xarr[ix];
                  ysave = yarr[ix];
                  itars = itarr[ix];
                  for (jx=ix; jx<=lj-5; jx +=2) {
                     xarr[jx]  = xarr[jx+2];
                     yarr[jx]  = yarr[jx+2];
                     itarr[jx] = itarr[jx+2];
                  }
                  xarr[lj-3]  = xsave;
                  yarr[lj-3]  = ysave;
                  itarr[lj-3] = itars;
                  if (count > 100) break;
                  count++;
               }
            }

            if (count > 100) continue;
            for (ix=1; ix<=lj-2; ix +=2) {
               theColor = Int_t((itarr[ix-1]+0.99)*Float_t(ncolors)/Float_t(ndivz));
               icol = gStyle->GetColorPalette(theColor);
               if (Hoption.Contour == 11) {
                  fH->SetLineColor(icol);
               }
               if (Hoption.Contour == 12) {
                  mode = icol%5;
                  if (mode == 0) mode = 5;
                  fH->SetLineStyle(mode);
               }
               if (Hoption.Contour != 1) {
                  fH->TAttLine::Modify();
                  gPad->PaintPolyLine(2,&xarr[ix-1],&yarr[ix-1]);
                  continue;
               }

               ipoly = itarr[ix-1];
               if (ipoly >=0 && ipoly <ncontour) {
                  poly = polys[ipoly];
                  poly->SetPoint(np[ipoly]  ,xarr[ix-1],yarr[ix-1]);
                  poly->SetPoint(np[ipoly]+1,xarr[ix],  yarr[ix]);
                  np[ipoly] += 2;
                  if (npmax < np[ipoly]) npmax = np[ipoly];
               }
            }
         } // end of if (ir[0]
      } //end of for (i
   } //end of for (j

   Double_t xmin,ymin;
   Double_t *xp, *yp;
   Int_t nadd,iminus,iplus;
   Double_t *xx, *yy;
   Int_t istart;
   Int_t first = ncontour;
   Int_t *polysort = 0;
   Int_t contListNb;
   if (Hoption.Contour != 1) goto theEND;

   //The 2 points line generated above are now sorted/merged to generate
   //a list of consecutive points.
   // If the option "List" has been specified, the list of points is saved
   // in the form of TGraph objects in the ROOT list of special objects.
   xmin = gPad->GetUxmin();
   ymin = gPad->GetUymin();
   xp = new Double_t[2*npmax];
   yp = new Double_t[2*npmax];
   polysort = new Int_t[ncontour];
   //find first positive contour
   for (ipoly=0;ipoly<ncontour;ipoly++) {
      if (levels[ipoly] >= 0) {first = ipoly; break;}
   }
   //store negative contours from 0 to minimum, then all positive contours
   k = 0;
   for (ipoly=first-1;ipoly>=0;ipoly--) {polysort[k] = ipoly; k++;}
   for (ipoly=first;ipoly<ncontour;ipoly++) {polysort[k] = ipoly; k++;}
   // we can now draw sorted contours
   contListNb = 0;
   fH->SetFillStyle(1001);
   for (k=0;k<ncontour;k++) {
      ipoly = polysort[k];
      if (np[ipoly] == 0) continue;
      if (Hoption.List) list = (TList*)contours->At(contListNb);
      contListNb++;
      poly = polys[ipoly];
      xx = poly->GetX();
      yy = poly->GetY();
      istart = 0;
      while (1) {
         iminus = npmax;
         iplus  = iminus+1;
         xp[iminus]= xx[istart];   yp[iminus] = yy[istart];
         xp[iplus] = xx[istart+1]; yp[iplus]  = yy[istart+1];
         xx[istart]   = xmin; yy[istart]   = ymin;
         xx[istart+1] = xmin; yy[istart+1] = ymin;
         while (1) {
            nadd = 0;
            for (i=2;i<np[ipoly];i+=2) {
               if (xx[i] == xp[iplus] && yy[i] == yp[iplus]) {
                  iplus++;
                  xp[iplus] = xx[i+1]; yp[iplus]  = yy[i+1];
                  xx[i]   = xmin; yy[i]   = ymin;
                  xx[i+1] = xmin; yy[i+1] = ymin;
                  nadd++;
               }
               if (xx[i+1] == xp[iminus] && yy[i+1] == yp[iminus]) {
                  iminus--;
                  xp[iminus] = xx[i];   yp[iminus]  = yy[i];
                  xx[i]   = xmin; yy[i]   = ymin;
                  xx[i+1] = xmin; yy[i+1] = ymin;
                  nadd++;
               }
            }
            if (nadd == 0) break;
         }
         theColor = Int_t((ipoly+0.99)*Float_t(ncolors)/Float_t(ndivz));
         icol = gStyle->GetColorPalette(theColor);
         if (ndivz > 1) fH->SetFillColor(icol);
         fH->TAttFill::Modify();
         gPad->PaintFillArea(iplus-iminus+1,&xp[iminus],&yp[iminus]);
         if (Hoption.List) {
            graph = new TGraph(iplus-iminus+1,&xp[iminus],&yp[iminus]);
            graph->SetFillColor(icol);
            graph->SetLineWidth(fH->GetLineWidth());
            list->Add(graph);
         }
         //check if more points are left
         istart = 0;
         for (i=2;i<np[ipoly];i+=2) {
            if (xx[i] != xmin && yy[i] != ymin) {
               istart = i;
               break;
            }
         }
         if (istart == 0) break;
      }
   }

   for (i=0;i<ncontour;i++) delete polys[i];
   delete [] polys;
   delete [] xp;
   delete [] yp;
   delete [] polysort;

theEND:
   gPad->ResetBit(TGraph::kClipFrame);
   if (Hoption.Zscale) PaintPalette();
   fH->SetLineStyle(linesav);
   fH->SetLineColor(colorsav);
   fH->SetFillColor(fillsav);
   delete [] xarr;
   delete [] yarr;
   delete [] itarr;
   delete [] levels;
}

//______________________________________________________________________________
Int_t THistPainter::PaintContourLine(Double_t elev1, Int_t icont1, Double_t x1, Double_t y1,
                            Double_t elev2, Int_t icont2, Double_t x2, Double_t y2,
                            Double_t *xarr, Double_t *yarr, Int_t *itarr, Double_t *levels)
{
   // Fill the matrix XARR YARR for Contour Plot.

   Bool_t vert;
   Double_t tlen, tdif, elev, diff, pdif, xlen;
   Int_t n, i, icount;

   if (x1 == x2) {
      vert = kTRUE;
      tlen = y2 - y1;
   } else {
      vert = kFALSE;
      tlen = x2 - x1;
   }

   n = icont1 +1;
   tdif = elev2 - elev1;
   i = 0;
   icount = 0;
   while (n <= icont2 && i <= kMAXCONTOUR/2 -3) {
      //elev = fH->GetContourLevel(n);
      elev = levels[n];
      diff = elev - elev1;
      pdif = diff/tdif;
      xlen = tlen*pdif;
      if (vert) {
         if (Hoption.Logx)
            xarr[i] = TMath::Log10(x1);
         else
            xarr[i] = x1;
         if (Hoption.Logy)
            yarr[i] = TMath::Log10(y1 + xlen);
         else
            yarr[i] = y1 + xlen;
      } else {
         if (Hoption.Logx)
            xarr[i] = TMath::Log10(x1 + xlen);
         else
            xarr[i] = x1 + xlen;
         if (Hoption.Logy)
            yarr[i] = TMath::Log10(y1);
         else
            yarr[i] = y1;
      }
      itarr[i] = n;
      icount++;
      i +=2;
      n++;
   }
   return icount;
}

//______________________________________________________________________________
void THistPainter::PaintErrors(Option_t *)
{
   // Draw histogram error bars.
   //
   //       Draws error bars for the current histogram. The current polymarker
   //       is drawn at the centre of the errors according to CHOPT:
   //
   //       ' ' Coordinates are expressed in histogram coordinates
   //           (of the last drawn histogram). Error bars are drawn.
   //       '1' Small lines are drawn at the end of the error bars.
   //       '2' Error rectangles are drawn.
   //       '3' A filled area is drawn through the end points of the vertical
   //           error bars.
   //       '4' A smoothed filled area is drawn through the end points of the
   //           vertical error bars.
   //       '0' Turn off the symbols clipping.
   //
   //      'X0' When used with one of the "E" option, it suppress the error
   //           bar along X as gStyle->SetErrorX(0) would do.
   //
   //     Note that for all options, the line and fill attributes of the
   //     histogram are used for the errors or errors contours.
   //
   //     Use gStyle->SetErrorX(dx) to control the size of the error along x.
   //     set dx = 0 to suppress the error along x.
   //
   //     Use gStyle->SetEndErrorSize(np) to control the size of the lines
   //     at the end of the error bars (when option 1 is used).
   //     By default np=1. (np reprersents the number of pixels).
   //Begin_Html
   /*
   <img src="gif/PaintErrors.gif">
   */
   //End_Html

   const Int_t kBASEMARKER=8;
   Double_t xp, yp, ex1, ex2, ey1, ey2;
   Double_t delta;
   Double_t s2x, s2y, bxsize, bysize, symbolsize, xerror, sbase;
   Double_t xi1, xi2, xi3, xi4, yi1, yi2, yi3, yi4;
   Double_t xmin, xmax, ymin, ymax;
   Double_t logxmin = 0;
   Double_t logymin = 0;
   Int_t i, k, npoints, first, last, fixbin;
   Int_t if1 = 0;
   Int_t if2 = 0;
   Int_t drawmarker, errormarker;
   Int_t option0, option1, option2, option3, option4, optionE, optionEX0;

   Double_t *xline = 0;
   Double_t *yline = 0;
   option0 = option1 = option2 = option3 = option4 = optionE = optionEX0 = 0;
   if (Int_t(Hoption.Error/10) == 2) {optionEX0 = 1; Hoption.Error -= 10;}
   if (Hoption.Error == 31) {optionEX0 = 1; Hoption.Error = 1;}
   if (Hoption.Error == 10) option0 = 1;
   if (Hoption.Error == 11) option1 = 1;
   if (Hoption.Error == 12) option2 = 1;
   if (Hoption.Error == 13) option3 = 1;
   if (Hoption.Error == 14) {option4 = 1; option3 = 1;}
   if (option2+option3 == 0) optionE = 1;
   if (Hoption.Error == 0) optionE = 0;
   if (fXaxis->GetXbins()->fN) fixbin = 0;
   else                        fixbin = 1;

   errormarker = fH->GetMarkerStyle();
   if (optionEX0) {
      xerror = 0;
   } else {
      xerror = gStyle->GetErrorX();
   }
   symbolsize  = fH->GetMarkerSize();
   if (errormarker == 1) symbolsize = 0.01;
   sbase       = symbolsize*kBASEMARKER;
   // set the graphics attributes

   fH->TAttLine::Modify();
   fH->TAttFill::Modify();
   fH->TAttMarker::Modify();

   // set the first and last bin

   Double_t factor = Hparam.factor;
   first      = Hparam.xfirst;
   last       = Hparam.xlast;
   npoints    = last - first  +1;
   xmin       = gPad->GetUxmin();
   xmax       = gPad->GetUxmax();
   ymin       = gPad->GetUymin();
   ymax       = gPad->GetUymax();


   if (option3) {
      xline = new Double_t[2*npoints];
      yline = new Double_t[2*npoints];
      if (!xline || !yline) {
         Error("PaintErrors", "too many points, out of memory");
         return;
      }
      if1 = 1;
      if2 = 2*npoints;
   }

   //  compute the offset of the error bars due to the symbol size
   s2x    = gPad->PixeltoX(Int_t(0.5*sbase)) - gPad->PixeltoX(0);
   s2y    =-gPad->PixeltoY(Int_t(0.5*sbase)) + gPad->PixeltoY(0);

   // compute size of the lines at the end of the error bars
   Int_t dxend = Int_t(gStyle->GetEndErrorSize());
   bxsize    = gPad->PixeltoX(dxend) - gPad->PixeltoX(0);
   bysize    =-gPad->PixeltoY(dxend) + gPad->PixeltoY(0);


   if (fixbin) {
      if (Hoption.Logx) xp = TMath::Power(10,Hparam.xmin) + 0.5*Hparam.xbinsize;
      else              xp = Hparam.xmin + 0.5*Hparam.xbinsize;
   }
   else {
      delta = fH->GetBinWidth(first);
      xp    = fH->GetBinLowEdge(first) + 0.5*delta;
   }

   // if errormarker = 0 or symbolsize = 0. no symbol is drawn
   if (Hoption.Logx) logxmin = TMath::Power(10,Hparam.xmin);
   if (Hoption.Logy) logymin = TMath::Power(10,Hparam.ymin);

   //    ---------------------- Loop over the points---------------------
   for (k=first; k<=last; k++) {

      //          get the data
      //     xp      = X position of the current point
      //     yp      = Y position of the current point
      //     ex1   = Low X error
      //     ex2   = Up X error
      //     ey1   = Low Y error
      //     ey2   = Up Y error
      //     (xi,yi) = Error bars coordinates

      if (Hoption.Logx) {
         if (xp <= 0) goto L30;
         if (xp < logxmin) goto L30;
         if (xp > TMath::Power(10,xmax)) break;
      } else {
         if (xp < xmin) goto L30;
         if (xp > xmax) break;
      }
      yp = factor*fH->GetBinContent(k);
      if (fixbin) ex1 = xerror*Hparam.xbinsize;
      else {
         delta = fH->GetBinWidth(k);
         ex1 = xerror*delta;
      }
      ey1 = factor*fH->GetBinError(k);
      ex2 = ex1;
      ey2 = ey1;

      xi4 = xp;
      xi3 = xp;
      xi2 = xp + ex2;
      xi1 = xp - ex1;

      yi1 = yp;
      yi2 = yp;
      yi3 = yp - ey1;
      yi4 = yp + ey2;

      //          take the LOG if necessary
      if (Hoption.Logx) {
         xi1 = TMath::Log10(TMath::Max(xi1,logxmin));
         xi2 = TMath::Log10(TMath::Max(xi2,logxmin));
         xi3 = TMath::Log10(TMath::Max(xi3,logxmin));
         xi4 = TMath::Log10(TMath::Max(xi4,logxmin));
      }
      if (Hoption.Logy) {
         yi1 = TMath::Log10(TMath::Max(yi1,logymin));
         yi2 = TMath::Log10(TMath::Max(yi2,logymin));
         yi3 = TMath::Log10(TMath::Max(yi3,logymin));
         yi4 = TMath::Log10(TMath::Max(yi4,logymin));
      }

      // test if error bars are not outside the limits
      //  otherwise they are truncated

      xi1 = TMath::Max(xi1,xmin);
      xi2 = TMath::Min(xi2,xmax);
      yi3 = TMath::Max(yi3,ymin);
      yi4 = TMath::Min(yi4,ymax);

      //  test if the marker is on the frame limits. If "Yes", the
      //  marker will not be drawn and the error bars will be readjusted.

      drawmarker = kTRUE;
      if (!option0 && !option3) {
         if (Hoption.Logy && yp < logymin) goto L30; 
         if (yi1 < ymin || yi1 > ymax) goto L30;
         if (Hoption.Error != 0 && yp == 0 && ey1 <= 0) drawmarker = kFALSE;
      }
      if (!symbolsize || !errormarker) drawmarker = kFALSE;

      //  draw the error rectangles
      if (option2) gPad->PaintBox(xi1,yi3,xi2,yi4);

      //  keep points for fill area drawing
      if (option3) {
         xline[if1-1] = xi3;
         xline[if2-1] = xi3;
         yline[if1-1] = yi4;
         yline[if2-1] = yi3;
         if1++;
         if2--;
      }

      //          draw the error bars

      if (optionE && drawmarker) {
         if (yi3 < yi1 - s2y) gPad->PaintLine(xi3,yi3,xi4,yi1 - s2y);
         if (yi1 + s2y < yi4) gPad->PaintLine(xi3,yi1 + s2y,xi4,yi4);
         // don't duplicate the horizontal line
         if (Hoption.Hist != 2){
            if (xi1 < xi3 - s2x) gPad->PaintLine(xi1,yi1,xi3 - s2x,yi2);
            if (xi3 + s2x < xi2) gPad->PaintLine(xi3 + s2x,yi1,xi2,yi2);
         }
      }
      if (optionE && !drawmarker && ey1 != 0) {
         if (yi3 < yi4) gPad->PaintLine(xi3,yi3,xi4,yi4);
         if (yi1 < yi4) gPad->PaintLine(xi3,yi1,xi4,yi4);
         // don't duplicate the horizontal line
         if (Hoption.Hist != 2){
            if (xi1 < xi3) gPad->PaintLine(xi1,yi1,xi3,yi2);
            if (xi3 < xi2) gPad->PaintLine(xi3,yi1,xi2,yi2);
         }
      }

      //          draw line at the end of the error bars

      if (option1 && drawmarker) {
         if (yi3 < yi1-s2y) gPad->PaintLine(xi3 - bxsize,yi3,xi3 + bxsize,yi3);
         if (yi4 > yi1+s2y) gPad->PaintLine(xi3 - bxsize,yi4,xi3 + bxsize,yi4);
         if (xi1 < xi3-s2x) gPad->PaintLine(xi1,yi1 - bysize,xi1,yi1 + bysize);
         if (xi2 > xi3+s2x) gPad->PaintLine(xi2,yi1 - bysize,xi2,yi1 + bysize);
      }

      //          draw the marker

      if (drawmarker) gPad->PaintPolyMarker(1, &xi3, &yi1);

L30:
      if (fixbin) xp += Hparam.xbinsize;
      else {
         if (k < last) {
            delta = fH->GetBinWidth(k+1);
            xp    = fH->GetBinLowEdge(k+1) + 0.5*delta;
         }
      }
   }  //end of for loop

   //          draw the filled area

   if (option3) {
      TGraph graph;
      graph.SetLineStyle(fH->GetLineStyle());
      graph.SetLineColor(fH->GetLineColor());
      graph.SetLineWidth(fH->GetLineWidth());
      graph.SetFillStyle(fH->GetFillStyle());
      graph.SetFillColor(fH->GetFillColor());
      Int_t logx = gPad->GetLogx();
      Int_t logy = gPad->GetLogy();
      gPad->SetLogx(0);
      gPad->SetLogy(0);

      // In some cases the number of points in the fill area is smaller than
      // 2*npoints. In such cases the array xline and yline must be arranged
      // before being plotted. The next loop does that.
      if (if2 > npoints) {
         for(i=1; i<if1 ;i++) {
            xline[if1-2+i] = xline[if2-1+i];
            yline[if1-2+i] = yline[if2-1+i];
         }
         npoints = if1-1;
      }
      if (option4) graph.PaintGraph(2*npoints,xline,yline,"FC");
      else         graph.PaintGraph(2*npoints,xline,yline,"F");
      gPad->SetLogx(logx);
      gPad->SetLogy(logy);
      delete [] xline;
      delete [] yline;
   }
}

//______________________________________________________________________________
void THistPainter::Paint2DErrors(Option_t *)
{
   // Draw 2D histograms errors

   fH->TAttMarker::Modify();
   fH->TAttLine::Modify();

   // Define the 3D view
   fXbuf[0] = Hparam.xmin;
   fYbuf[0] = Hparam.xmax;
   fXbuf[1] = Hparam.ymin;
   fYbuf[1] = Hparam.ymax;
   fXbuf[2] = Hparam.zmin;
   fYbuf[2] = Hparam.zmax;
   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);
   TView *view = gPad->GetView();
   if (!view) {
      Error("Paint2DErrors", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   Int_t irep;
   view->SetView(phideg, thedeg, psideg, irep);

   // Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();
   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);
   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   // Paint the Back Box if needed
   if (Hoption.BackBox && !Hoption.Same) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   // Paint the Errors
   Double_t x, ex, x1, x2;
   Double_t y, ey, y1, y2;
   Double_t z, ez, z1, z2;
   Double_t temp1[3],temp2[3];
   Double_t xyerror;
   if (Hoption.Error == 110) {
      xyerror = 0 ;
   } else {
      xyerror = gStyle->GetErrorX();
   }

   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      y  = fYaxis->GetBinCenter(j);
      ey = fYaxis->GetBinWidth(j)*xyerror;
      y1 = y-ey;
      y2 = y+ey;
      if (Hoption.Logy) {
         if (y > 0)  y = TMath::Log10(y);
         else        continue;
         if (y1 > 0) y1 = TMath::Log10(y1);
         else        y1 = Hparam.ymin;
         if (y2 > 0) y2 = TMath::Log10(y2);
         else        y2 = Hparam.ymin;
      }
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin = fH->GetBin(i,j);
         x  = fXaxis->GetBinCenter(i);
         ex = fXaxis->GetBinWidth(i)*xyerror;
         x1 = x-ex;
         x2 = x+ex;
         if (Hoption.Logx) {
            if (x > 0)  x = TMath::Log10(x);
            else        continue;
            if (x1 > 0) x1 = TMath::Log10(x1);
            else        x1 = Hparam.xmin;
            if (x2 > 0) x2 = TMath::Log10(x2);
            else        x2 = Hparam.xmin;
         }
         z  = fH->GetBinContent(bin);
         ez = fH->GetBinError(bin);
         z1 = z-ez;
         z2 = z+ez;
         if (Hoption.Logz) {
            if (z > 0)   z = TMath::Log10(z);
            else         z = Hparam.zmin;
            if (z1 > 0) z1 = TMath::Log10(z1);
            else        z1 = Hparam.zmin;
            if (z2 > 0) z2 = TMath::Log10(z2);
            else        z2 = Hparam.zmin;

         }
         if (z <= Hparam.zmin) continue;
         if (z >  Hparam.zmax) z = Hparam.zmax;

         temp1[0] = x1;
         temp1[1] = y;
         temp1[2] = z;
         temp2[0] = x2;
         temp2[1] = y;
         temp2[2] = z;
         gPad->PaintLine3D(temp1, temp2);
         temp1[0] = x;
         temp1[1] = y1;
         temp1[2] = z;
         temp2[0] = x;
         temp2[1] = y2;
         temp2[2] = z;
         gPad->PaintLine3D(temp1, temp2);
         temp1[0] = x;
         temp1[1] = y;
         temp1[2] = z1;
         temp2[0] = x;
         temp2[1] = y;
         temp2[2] = z2;
         gPad->PaintLine3D(temp1, temp2);
         temp1[0] = x;
         temp1[1] = y;
         temp1[2] = z;
         view->WCtoNDC(temp1, &temp2[0]);
         gPad->PaintPolyMarker(1, &temp2[0], &temp2[1]);
      }
   }

   // Paint the Front Box if needed
   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }

   // Paint the Axis if needed
   if (!Hoption.Axis && !Hoption.Same) {
      TGaxis *axis = new TGaxis();
      PaintLegoAxis(axis, 90);
      delete axis;
   }

   delete fLego; fLego = 0;
}

//______________________________________________________________________________
void THistPainter::PaintFrame()
{
   // Calculate range and clear pad (canvas).

   if (Hoption.Same) return;

   RecalculateRange();

   if (Hoption.Lego || Hoption.Surf || Hoption.Tri ||
       Hoption.Contour == 14 || Hoption.Error >= 100) {
      TObject *frame = gPad->FindObject("TFrame");
      if (frame) gPad->GetListOfPrimitives()->Remove(frame);
      return;
   }
   gPad->PaintPadFrame(Hparam.xmin,Hparam.ymin,Hparam.xmax,Hparam.ymax);
}

//______________________________________________________________________________
void THistPainter::PaintFunction(Option_t *)
{
   // Paint functions associated to an histogram.
   //
   //   An associated function is created by THistPainter::Fit. Note that more than
   //   on fitted function can be associated with one histogram (see THistPainter::Fit).
   //
   //   A TF1 object can be added to the list of associated functions directly
   //   by a user without calling THistPainter::Fit.
   //   To add a new function to the list of associated functions, do
   //     h->GetListOfFunctions()->Add(f1);
   //        or
   //     h->GetListOfFunctions()->Add(f1,someoption);
   //   To retrieve a function by name from this list, do:
   //     TF1 *f1 = (TF1*)h->GetListOfFunctions()->FindObject(name);
   //   or
   //     TF1 *f1 = h->GetFunction(name);

   TObjOptLink *lnk = (TObjOptLink*)fFunctions->FirstLink();
   TObject *obj;

   while (lnk) {
      obj = lnk->GetObject();
      TVirtualPad *padsave = gPad;
      if (obj->InheritsFrom(TF1::Class())) {
         if (obj->TestBit(TF1::kNotDraw) == 0) obj->Paint("lsame");
      } else  {
         obj->Paint(lnk->GetOption());
      }
      lnk = (TObjOptLink*)lnk->Next();
      padsave->cd();
   }
}

//______________________________________________________________________________
void THistPainter::PaintHist(Option_t *)
{
   // Control routine to draw an histogram.

   static char chopth[17];

   Int_t htype, oldhtype;
   Int_t i, j, first, last, nbins, fixbin;
   Double_t c1, yb, y1, y2, ym1, ym2, yadd, ycur, ypre;
   Double_t ync, ynext, ypc;
   yb = ynext = 0;

   strcpy(chopth, "                ");

   Double_t ymin = Hparam.ymin;
   Double_t ymax = Hparam.ymax;
   Double_t baroffset = fH->GetBarOffset();
   Double_t barwidth  = fH->GetBarWidth();
   Double_t baroffsetsave = gStyle->GetBarOffset();
   Double_t barwidthsave  = gStyle->GetBarWidth();
   gStyle->SetBarOffset(baroffset);
   gStyle->SetBarWidth(barwidth);

   //       Create "LIFE" structure to keep current histogram status

   first = Hparam.xfirst;
   last  = Hparam.xlast;
   nbins = last - first + 1;

   Double_t *keepx = 0;
   Double_t *keepy = 0;
   if (fXaxis->GetXbins()->fN) fixbin = 0;
   else                        fixbin = 1;
   if (!Hoption.Plus) {
      if (fixbin) keepx = new Double_t[2];
      else        keepx = new Double_t[nbins+1];
      keepy = new Double_t[nbins];
   }
   Double_t logymin = 0;
   if (Hoption.Logy) logymin = TMath::Power(10,ymin);

   //      Loop on histogram bins

   for (j=first; j<=last;j++) {
      c1 = Hparam.factor*fH->GetBinContent(j);
      if (TMath::Abs(ymax-ymin) > 0) {
         if (Hoption.Logy) yb = TMath::Log10(TMath::Max(c1,.1*logymin));
         else              yb = c1;
      }
      yb = TMath::Max(yb, ymin);
      yb = TMath::Min(yb, ymax);
      if (Hoption.Plus) {
   //            compute y1, y2
         y1 = keepy[j-first];
         y2 = yb;
         if (Hoption.Plus == 1 || Hoption.Plus == -1) {
            if (Hoption.Logy)
               y2 = TMath::Log10(TMath::Power(10,y1) + Hoption.Plus*TMath::Power(10,y2));
            else
               y2 = y1 +y2*Hoption.Plus;
         }
   //            compute ym1, ym2
         if (!Hoption.Bar) {
            yadd = Hparam.factor*fH->GetBinContent(j);
            if (Hoption.Logy) ycur = TMath::Power(10,keepy[j-first]);
            else              ycur = keepy[j-first];
            if (j != last) {
               ync = Hparam.factor*fH->GetBinContent(j+1);
               if (Hoption.Logy) ycur = TMath::Power(10,keepy[j-first+1]);
               else              ycur = keepy[j-first+1];
            }
            else { ync = ynext = 0;}
            if (j != first) {
               ypc = Hparam.factor*fH->GetBinContent(j-1);
               if (Hoption.Logy) ypre = TMath::Power(10,keepy[j-first]);
               else              ypre = keepy[j-first];
            }
            else { ypc = ypre = 0;}
            if (Hoption.Plus == 1) {
               ym1 = TMath::Max(TMath::Min(ycur+yadd,ypre),ycur);
               ym2 = TMath::Max(TMath::Min(ycur+yadd,ynext+ync),ycur);
            }
            else if(Hoption.Plus == -1) {
               ym1 = TMath::Max(TMath::Min(ycur-yadd,ypre),ycur);
               ym2 = TMath::Max(TMath::Min(ycur-yadd,ynext-ync),ycur);
            }
            else {
               if (ycur > yadd) { ym1 = TMath::Max(yadd,ypc); ym2 = TMath::Max(yadd,ync); }
               else {             ym1 = TMath::Min(yadd,ypc); ym2 = TMath::Min(yadd,ync); }
            }
            if (Hoption.Logy) {
               if (ym1 > 0) ym1 = TMath::Log10(ym1);
               if (ym2 > 0) ym2 = TMath::Log10(ym2);
            }
         }

   //               Compute X1 X2
         if (Hoption.Logy && c1 <= 0) continue;

   //     Update the current plot

         if (Hoption.Plus == 1 || Hoption.Plus == -1) keepy[j-first-1] = y2;
      }
      else {
         keepy[j-first] = yb;
      }
   } // end of for loop

   if (Hoption.Plus) return;

   //              Draw histogram according to value of FillStyle and FillColor

   if (fixbin) { keepx[0] = Hparam.xmin; keepx[1] = Hparam.xmax; }
   else {
      for (i=0; i<=nbins; i++) keepx[i] = fXaxis->GetBinLowEdge(i+first);
   }

   //         Prepare Fill area (systematic with option "Bar").

   oldhtype = fH->GetFillStyle();
   htype    = oldhtype;
   if (Hoption.Bar) {
      if (htype == 0 || htype == 1000) htype = 1001;
   }

   Width_t lw = (Width_t)fH->GetLineWidth();

   //         Code option for GrapHist

   if (Hoption.Line) chopth[0] = 'L';
   if (Hoption.Star) chopth[1] = '*';
   if (Hoption.Mark) chopth[2] = 'P';
   if (Hoption.Mark == 10) chopth[3] = '0';
   if (Hoption.Line || Hoption.Curve || Hoption.Hist || Hoption.Bar) {
      if (Hoption.Curve)    chopth[3] = 'C';
      if (Hoption.Hist > 0) chopth[4] = 'H';
      else if (Hoption.Bar) chopth[5] = 'B';
      if (fH->GetFillColor() && htype) {
         if (Hoption.Logy) {
            chopth[6] = '1';
         }
         if (Hoption.Hist > 0 || Hoption.Curve || Hoption.Line) {
            chopth[7] = 'F';
         }
      }
   }
   if (!fixbin && strlen(chopth)) {
      chopth[8] = 'N';
   }

   if (Hoption.Fill == 2)    strcat(chopth,"2");
   if (Hoption.HighRes != 0) strcat(chopth,"9");

   //         Option LOGX

   if (Hoption.Logx) {
      chopth[9]  = 'G';
      chopth[10] = 'X';
      if (fixbin) {
         keepx[0] = TMath::Power(10,keepx[0]);
         keepx[1] = TMath::Power(10,keepx[1]);
      }
   }

   if (Hoption.Off) strcat(chopth,"][");

   //         Draw the histogram

   TGraph graph;
   graph.SetLineWidth(lw);
   graph.SetLineStyle(fH->GetLineStyle());
   graph.SetLineColor(fH->GetLineColor());
   graph.SetFillStyle(htype);
   graph.SetFillColor(fH->GetFillColor());
   graph.SetMarkerStyle(fH->GetMarkerStyle());
   graph.SetMarkerSize(fH->GetMarkerSize());
   graph.SetMarkerColor(fH->GetMarkerColor());
   if (!Hoption.Same) graph.ResetBit(TGraph::kClipFrame);

   graph.PaintGrapHist(nbins, keepx, keepy ,chopth);

   delete [] keepx;
   delete [] keepy;
   gStyle->SetBarOffset(baroffsetsave);
   gStyle->SetBarWidth(barwidthsave);

   htype=oldhtype;
}

//______________________________________________________________________________
void THistPainter::PaintH3(Option_t *option)
{
   // Control function to draw a 3-D histogram.

   char *cmd;
   if (fH->GetDrawOption() && strstr(fH->GetDrawOption(),"box")) {
      cmd = Form("TMarker3DBox::PaintH3((TH1 *)0x%lx,\"%s\");",(Long_t)fH,option);
   } else if (fH->GetDrawOption() && strstr(fH->GetDrawOption(),"iso")) {
      PaintH3Iso();
      return;
   } else if (strstr(option,"tf3")) {
      PaintTF3();
      return;
   } else {
      cmd = Form("TPolyMarker3D::PaintH3((TH1 *)0x%lx,\"%s\");",(Long_t)fH,option);
   }
   gROOT->ProcessLine(cmd);

   //Draw axis
   if (Hoption.Same) return;
   TView *view = gPad->GetView();
   if (!view) return;
   view->SetOutlineToCube();
   view->GetOutline()->Paint(option);
   Hoption.System = kCARTESIAN;
   TGaxis *axis = new TGaxis();
   PaintLegoAxis(axis,90);
   delete axis;

   //Draw title
   PaintTitle();
   
   //Draw stats and fit results
   TF1 *fit  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TF1::Class())) {
         fit = (TF1*)obj;
         break;
      }
   }
   if (Hoption.Same != 1) {
      if (!fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
         PaintStat3(gStyle->GetOptStat(),fit);
      }
   }

}

//______________________________________________________________________________
Int_t THistPainter::PaintInit()
{
   // Compute histogram parameters used by the drawing routines.

   if (fH->GetDimension() > 1 || Hoption.Lego || Hoption.Surf) return 1;

   static const char *where = "PaintInit";
   Double_t yMARGIN = gStyle->GetHistTopMargin();
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

   //     Compute X axis parameters

   Int_t last      = fXaxis->GetLast();
   Int_t first     = fXaxis->GetFirst();
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.xmin     = Hparam.xlowedge;
   Hparam.xmax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

   //       if log scale in X, replace xmin,max by the log
   if (Hoption.Logx) {
      if (Hparam.xlowedge <=0 ) {
         Hparam.xlowedge = 0.1*Hparam.xbinsize;
         Hparam.xmin  = Hparam.xlowedge;
      }
      if (Hparam.xmin <=0 || Hparam.xmax <=0) {
         Error(where, "cannot set X axis to log scale");
         return 0;
      }
      Hparam.xfirst= fXaxis->FindFixBin(Hparam.xmin);
      Hparam.xlast = fXaxis->FindFixBin(Hparam.xmax);
      Hparam.xmin  = TMath::Log10(Hparam.xmin);
      Hparam.xmax  = TMath::Log10(Hparam.xmax);
      if (Hparam.xlast  > last)  Hparam.xlast  = last;
      if (Hparam.xfirst < first) Hparam.xfirst = first;
   }

   //     Compute Y axis parameters
   Double_t bigp = TMath::Power(10,32);
   Double_t ymax = -bigp;
   Double_t ymin = bigp;
   Double_t c1, e1;
   Double_t xv[1];
   Double_t fval;
   Int_t i;
   TObject *f;
   TF1 *f1;
   Double_t allchan = 0;
   Int_t nonNullErrors = 0;
   TIter   next(fFunctions);
   for (i=first; i<=last;i++) {
      c1 = fH->GetBinContent(i);
      ymax = TMath::Max(ymax,c1);
      if (Hoption.Logy) {
         if (c1 > 0) ymin = TMath::Min(ymin,c1);
      } else {
         ymin = TMath::Min(ymin,c1);
      }
      if (Hoption.Error) {
         e1 = fH->GetBinError(i);
         if (e1 > 0) nonNullErrors++;
         ymax = TMath::Max(ymax,c1+e1);
         if (Hoption.Logy) {
            if (c1-e1>0.01*TMath::Abs(c1)) ymin = TMath::Min(ymin,c1-e1);
         } else {
            ymin = TMath::Min(ymin,c1-e1);
         }
      }
      if (Hoption.Func) {
         xv[0] = fXaxis->GetBinCenter(i);
         while ((f = (TObject*) next())) {
            if (f->IsA() == TF1::Class()) {
               f1 = (TF1*)f;
               if (xv[0] < f1->GetXmin() || xv[0] > f1->GetXmax()) continue;
               fval = f1->Eval(xv[0],0,0);
               ymax = TMath::Max(ymax,fval);
               if (Hoption.Logy) {
                  if (c1 > 0 && fval > 0.3*c1) ymin = TMath::Min(ymin,fval);
               }
            }
         }
         next.Reset();
      }
      allchan += c1;
   }
   if (!nonNullErrors) {
      if (Hoption.Error) {
         if (!Hoption.Mark && !Hoption.Line && !Hoption.Star && !Hoption.Curve) Hoption.Hist = 2;
         Hoption.Error=0;
      }
   }

   //     Take into account maximum , minimum

   if (Hoption.Logy && ymin <= 0) {
      if (ymax >= 1) ymin = TMath::Max(.005,ymax*1e-10);
      else           ymin = 0.001*ymax;
   }
   Double_t xm = ymin;
   if (maximum) ymax = fH->GetMaximumStored();
   if (minimum) xm   = fH->GetMinimumStored();
   if (Hoption.Logy && xm <= 0) {
      Error(where, "log scale requested with zero or negative argument (%f)", xm);
      return 0;
   }
   else ymin = xm;
   if (ymin >= ymax && !Hoption.Plus) {
      if (Hoption.Logy) {
         if (ymax > 0) ymin = 0.001*ymax;
         else {
            if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", ymax);
            return 0;
         }
      }
      else {
         if (ymin > 0) {
            ymin = 0;
            ymax *= 2;
         } else if (ymin < 0) {
            ymax = 0;
            ymin *= 2;
         } else {
            ymin = 0;
            ymax = 1;
         }
      }
   }

   //     take into account normalization factor
   Hparam.allchan = allchan;
   Double_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   ymax = factor*ymax;
   ymin = factor*ymin;
   //just in case the norm factor is negative
   // this may happen with a positive norm factor and a negative integral !
   if (ymax < ymin) {
      Double_t temp = ymax;
      ymax = ymin;
      ymin = temp;
   }

   //         For log scales, histogram coordinates are LOG10(ymin) and
   //         LOG10(ymax). Final adjustment (if not option "Same"
   //         or "+" for ymax) of ymax and ymin for logarithmic scale, if
   //         Maximum and Minimum are not defined.
   if (Hoption.Logy) {
      if (ymin <=0 || ymax <=0) {
         Error(where, "Cannot set Y axis to log scale");
         return 0;
      }
      ymin = TMath::Log10(ymin);
      if (!minimum) ymin += TMath::Log10(0.5);
      ymax = TMath::Log10(ymax);
      if (!maximum && !Hoption.Plus) ymax += TMath::Log10(2*(0.9/0.95));
      if (!Hoption.Same) {
         Hparam.ymin = ymin;
         Hparam.ymax = ymax;
      }
      return 1;
   }

   //         final adjustment of ymin for linear scale.
   //         if minimum is not set , then ymin is set to zero if >0
   //         or to ymin - margin if <0.
   if (!minimum) {
      if (gStyle->GetHistMinimumZero()) {
         if (ymin >= 0) ymin = 0;
         else           ymin -= yMARGIN*(ymax-ymin);
      } else {
         Double_t dymin = yMARGIN*(ymax-ymin);
         if (ymin >= 0 && (ymin-dymin <= 0)) ymin  = 0;
         else                                ymin -= dymin;
      }
   }

   //         final adjustment of YMAXI for linear scale (if not option "Same"):
   //         decrease histogram height to MAX% of allowed height if HMAXIM
   //         has not been called.
   if (!maximum && !Hoption.Plus) {
      ymax += yMARGIN*(ymax-ymin);
   }
   Hparam.ymin = ymin;
   Hparam.ymax = ymax;
   return 1;
}

//______________________________________________________________________________
Int_t THistPainter::PaintInitH()
{
   //    Compute histogram parameters used by the drawing routines
   //    for a rotated pad

   static const char *where = "PaintInitH";
   Double_t yMARGIN = gStyle->GetHistTopMargin();
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

   //     Compute X axis parameters

   Int_t last      = fXaxis->GetLast();
   Int_t first     = fXaxis->GetFirst();
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.ymin     = Hparam.xlowedge;
   Hparam.ymax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

   //       if log scale in Y, replace ymin,max by the log
   if (Hoption.Logy) {
      if (Hparam.xlowedge <=0 ) {
         Hparam.xlowedge = 0.1*Hparam.xbinsize;
         Hparam.ymin  = Hparam.xlowedge;
      }
      if (Hparam.ymin <=0 || Hparam.ymax <=0) {
         Error(where, "cannot set Y axis to log scale");
         return 0;
      }
      Hparam.xfirst= fXaxis->FindFixBin(Hparam.ymin);
      Hparam.xlast = fXaxis->FindFixBin(Hparam.ymax);
      Hparam.ymin  = TMath::Log10(Hparam.ymin);
      Hparam.ymax  = TMath::Log10(Hparam.ymax);
      if (Hparam.xlast > last) Hparam.xlast = last;
   }

   //     Compute Y axis parameters
   Double_t bigp = TMath::Power(10,32);
   Double_t xmax = -bigp;
   Double_t xmin = bigp;
   Double_t c1, e1;
   Double_t xv[1];
   Double_t fval;
   Int_t i;
   TObject *f;
   TF1 *f1;
   Double_t allchan = 0;
   TIter   next(fFunctions);
   for (i=first; i<=last;i++) {
      c1 = fH->GetBinContent(i);
      xmax = TMath::Max(xmax,c1);
      xmin = TMath::Min(xmin,c1);
      if (Hoption.Error) {
         e1 = fH->GetBinError(i);
         xmax = TMath::Max(xmax,c1+e1);
         xmin = TMath::Min(xmin,c1-e1);
      }
      if (Hoption.Func) {
         xv[0] = fXaxis->GetBinCenter(i);
         while ((f = (TObject*) next())) {
            if (f->IsA() == TF1::Class()) {
               f1 = (TF1*)f;
               if (xv[0] < f1->GetXmin() || xv[0] > f1->GetXmax()) continue;
               fval = f1->Eval(xv[0],0,0);
               xmax = TMath::Max(xmax,fval);
               if (Hoption.Logy) {
                  if (fval > 0.3*c1) xmin = TMath::Min(xmin,fval);
               }
            }
         }
         next.Reset();
      }
      allchan += c1;
   }

   //     Take into account maximum , minimum

   if (Hoption.Logx && xmin <= 0) {
      if (xmax >= 1) xmin = TMath::Max(.5,xmax*1e-10);
      else           xmin = 0.001*xmax;
   }
   Double_t xm = xmin;
   if (maximum) xmax = fH->GetMaximumStored();
   if (minimum) xm   = fH->GetMinimumStored();
   if (Hoption.Logx && xm <= 0) {
      Error(where, "log scale requested with zero or negative argument (%f)", xm);
      return 0;
   }
   else xmin = xm;
   if (xmin >= xmax && !Hoption.Plus) {
      if (Hoption.Logx) {
         if (xmax > 0) xmin = 0.001*xmax;
         else {
            if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", xmax);
            return 0;
         }
      }
      else {
         if (xmin > 0) {
            xmin = 0;
            xmax *= 2;
         } else if (xmin < 0) {
            xmax = 0;
            xmin *= 2;
         } else {
            xmin = -1;
            xmax = 1;
         }
      }
   }

   //     take into account normalization factor
   Hparam.allchan = allchan;
   Double_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   xmax = factor*xmax;
   xmin = factor*xmin;

   //         For log scales, histogram coordinates are LOG10(ymin) and
   //         LOG10(ymax). Final adjustment (if not option "Same"
   //         or "+" for ymax) of ymax and ymin for logarithmic scale, if
   //         Maximum and Minimum are not defined.
   if (Hoption.Logx) {
      if (xmin <=0 || xmax <=0) {
         Error(where, "Cannot set Y axis to log scale");
         return 0;
      }
      xmin = TMath::Log10(xmin);
      if (!minimum) xmin += TMath::Log10(0.5);
      xmax = TMath::Log10(xmax);
      if (!maximum && !Hoption.Plus) xmax += TMath::Log10(2*(0.9/0.95));
      if (!Hoption.Same) {
         Hparam.xmin = xmin;
         Hparam.xmax = xmax;
      }
      return 1;
   }

   //         final adjustment of ymin for linear scale.
   //         if minimum is not set , then ymin is set to zero if >0
   //         or to ymin - margin if <0.
   if (!minimum) {
      if (xmin >= 0) xmin = 0;
      else           xmin -= yMARGIN*(xmax-xmin);
   }

   //         final adjustment of YMAXI for linear scale (if not option "Same"):
   //         decrease histogram height to MAX% of allowed height if HMAXIM
   //         has not been called.
   if (!maximum && !Hoption.Plus) {
      xmax += yMARGIN*(xmax-xmin);
   }
   Hparam.xmin = xmin;
   Hparam.xmax = xmax;
   return 1;
}


//______________________________________________________________________________
void THistPainter::PaintH3Iso()
{
   // Control function to draw a 3d histogram with Iso Surfaces.
   //
   // Thanks to the function IsoSurface of the TPainter3dAlgorithms class, this
   // function paints a Gouraud shaded 3d iso surface though a 3d histogram.
   //
   // This first implementation paint one surface at the value computed as follow:
   // SumOfWeights/(NbinsX*NbinsY*NbinsZ)
   //
   // Example:
   //
   //  #include "TH3.h"
   //  #include "TRandom.h"
   //
   //  void hist3d() {
   //     TH3F *h3 = new TH3F("h3","h3",20,-2,2,20,-2,2,20,0,4);
   //     Double_t x, y, z;
   //     for (Int_t i=0;i<10000;i++) {
   //        gRandom->Rannor(x, y);
   //        z = x*x + y*y;
   //        h3->Fill(x,y,z);
   //     }
   //     h3->Draw("iso");
   //  }
   //
   //Begin_Html
   /*
   <img src="gif/PaintIso.gif">
   */
   //End_Html

   const Double_t ydiff = 1;
   const Double_t yligh1 = 10;
   const Double_t qa = 0.15;
   const Double_t qd = 0.15;
   const Double_t qs = 0.8;
   Double_t fmin, fmax;
   Int_t i, irep;
   Int_t nbcol = 28;
   Int_t icol1 = 201;
   Int_t ic1 = icol1;
   Int_t ic2 = ic1+nbcol;
   Int_t ic3 = ic2+nbcol;

   TGaxis *axis = new TGaxis();
   TAxis *xaxis = fH->GetXaxis();
   TAxis *yaxis = fH->GetYaxis();
   TAxis *zaxis = fH->GetZaxis();

   Int_t nx = fH->GetNbinsX();
   Int_t ny = fH->GetNbinsY();
   Int_t nz = fH->GetNbinsZ();

   Double_t *x = new Double_t[nx];
   Double_t *y = new Double_t[ny];
   Double_t *z = new Double_t[nz];

   for ( i=0 ; i<nx ; i++) x[i] = xaxis->GetBinCenter(i+1);
   for ( i=0 ; i<ny ; i++) y[i] = yaxis->GetBinCenter(i+1);
   for ( i=0 ; i<nz ; i++) z[i] = zaxis->GetBinCenter(i+1);

   fXbuf[0] = xaxis->GetBinLowEdge(xaxis->GetFirst());
   fYbuf[0] = xaxis->GetBinUpEdge(xaxis->GetLast());
   fXbuf[1] = yaxis->GetBinLowEdge(yaxis->GetFirst());
   fYbuf[1] = yaxis->GetBinUpEdge(yaxis->GetLast());
   fXbuf[2] = zaxis->GetBinLowEdge(zaxis->GetFirst());
   fYbuf[2] = zaxis->GetBinUpEdge(zaxis->GetLast());

   Double_t s[3];
   s[0] = fH->GetSumOfWeights()/(fH->GetNbinsX()*fH->GetNbinsY()*fH->GetNbinsZ());
   s[1] = 0.5*s[0];
   s[2] = 1.5*s[0];

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintH3Iso", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   Double_t dcol = 0.5/Double_t(nbcol);
   TColor *colref = gROOT->GetColor(fH->GetFillColor());
   Float_t r, g, b, hue, light, satur;
   colref->GetRGB(r,g,b);
   TColor::RGBtoHLS(r,g,b,hue,light,satur);
   TColor *acol;
   for (Int_t col=0;col<nbcol;col++) {
      acol = gROOT->GetColor(col+icol1);
      TColor::HLStoRGB(hue, .4+col*dcol, satur, r, g, b);
      acol->SetRGB(r, g, b);
   }

   fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.BackBox) {
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   fLego->LightSource(0, ydiff, 0, 0, 0, irep);
   fLego->LightSource(1, yligh1, 1, 1, 1, irep);
   fLego->SurfaceProperty(qa, qd, qs, 1, irep);
   fmin = ydiff*qa;
   fmax = ydiff*qa + (yligh1+0.1)*(qd+qs);
   fLego->SetIsoSurfaceParameters(fmin, fmax, nbcol, ic1, ic2, ic3);

   fLego->IsoSurface(1, s, nx, ny, nz, x, y, z, "BF");

   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);

   PaintTitle();

   delete axis;
   delete fLego; fLego = 0;
   delete [] x;
   delete [] y;
   delete [] z;
}


//______________________________________________________________________________
void THistPainter::PaintLego(Option_t *)
{
   // Control function to draw a table as a lego plot.
   //
   //      In a lego plot, cell contents are represented as 3-d boxes.
   //      The height of the box is proportional to the cell content.
   //
   //      A lego plot can be represented in several coordinate systems.
   //      Default system is Cartesian coordinates.
   //      Possible systems are CYL,POL,SPH,PSR.
   //
   //      See THistPainter::Draw for the list of Lego options.
   //      See TPainter3dAlgorithms for more examples of lego options.
   //
   //      See TStyle::SeTPaletteAxis to change the color palette.
   //      It is suggested to use palette 1 via the call
   //      gStyle->SetColorPalette(1)
   //
   //Begin_Html
   /*
   <img src="gif/PaintLego1.gif">
   */
   //End_Html

   Int_t raster = 1;
   if (Hparam.zmin == 0 && Hparam.zmax == 0) {Hparam.zmin = -1; Hparam.zmax = 1;}
   Int_t   nx      = Hparam.xlast - Hparam.xfirst + 1;
   Int_t   ny      = Hparam.ylast - Hparam.yfirst + 1;
   Double_t zmin   = Hparam.zmin;
   Double_t zmax   = Hparam.zmax;
   Double_t xlab1  = Hparam.xmin;
   Double_t xlab2  = Hparam.xmax;
   Double_t ylab1  = Hparam.ymin;
   Double_t ylab2  = Hparam.ymax;
   Double_t dangle = 10*3.141592/180; //Delta angle for Rapidity option
   Double_t deltaz = TMath::Abs(zmin);
   if (deltaz == 0) deltaz = 1;
   if (zmin >= zmax) {
      zmin -= 0.5*deltaz;
      zmax += 0.5*deltaz;
   }
   Double_t z1c = zmin;
   Double_t z2c = zmin + (zmax-zmin)*(1+gStyle->GetHistTopMargin());

   //     Compute the lego limits and instantiate a lego object
   fXbuf[0] = -1;
   fYbuf[0] =  1;
   fXbuf[1] = -1;
   fYbuf[1] =  1;
   if (Hoption.System == kPOLAR) {
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
   } else if (Hoption.System == kCYLINDRICAL) {
      if (Hoption.Logy) {
         if (ylab1 > 0) fXbuf[2] = TMath::Log10(ylab1);
         else           fXbuf[2] = 0;
         if (ylab2 > 0) fYbuf[2] = TMath::Log10(ylab2);
         else           fYbuf[2] = 0;
      } else {
         fXbuf[2] = ylab1;
         fYbuf[2] = ylab2;
      }
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kSPHERICAL) {
      fXbuf[2] = -1;
      fYbuf[2] =  1;
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kRAPIDITY) {
      fXbuf[2] = -1/TMath::Tan(dangle);
      fYbuf[2] =  1/TMath::Tan(dangle);
   } else {
      fXbuf[0] = xlab1;
      fYbuf[0] = xlab2;
      fXbuf[1] = ylab1;
      fYbuf[1] = ylab2;
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
      raster  = 0;
   }

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf, Hoption.System);

   //          Create axis object

   TGaxis *axis = new TGaxis();

   //                  Initialize the levels on the Z axis
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);

   //     Initialize colors for the lighting model (option Lego1 only)
   if (Hoption.Lego == 1) {
         Color_t colormain = fH->GetLineColor();
         fLego->SetColorMain(colormain,0);
   }
   if (Hoption.Lego == 11) {
      Int_t nids = 1;
      if (fStack) nids = fStack->GetSize();
      TH1 *hid = fH;
      for (Int_t id=0;id<=nids;id++) {
         if (id > 0 && fStack) hid = (TH1*)fStack->At(id-1);
         Color_t colormain = hid->GetFillColor();
         if (colormain == 1) colormain = 17; //avoid drawing with black
         Color_t colordark = colormain + 100;
         fLego->SetColorMain(colormain,id);
         fLego->SetColorDark(colordark,id);
         if (id == 0)    fLego->SetColorMain(colormain,-1);  // Set Bottom color
         if (id == nids) fLego->SetColorMain(colormain,99);  // Set Top color
      }
   }

   //     Now ready to draw the lego plot
   Int_t irep = 0;

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintLego", "no TView in current pad");
      return;
   }

   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   fLego->SetLineColor(fH->GetLineColor());
   fLego->SetFillStyle(fH->GetFillStyle());

   //     Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();

   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);

   if (raster) fLego->InitRaster(-1.1,-1.1,1.1,1.1,1000,800);
   else        fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.Lego == 11 || Hoption.Lego == 12) {
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }

   if (Hoption.Lego == 12) DefineColorLevels(ndivz);

   fLego->SetLegoFunction(&TPainter3dAlgorithms::LegoFunction);
   if (Hoption.Lego ==  1) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceRaster2);
   if (Hoption.Lego == 11) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode3);
   if (Hoption.Lego == 12) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
   if (Hoption.System == kPOLAR) {
      if (Hoption.Lego ==  1) fLego->LegoPolar(1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoPolar(1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoPolar(1,nx,ny,"BF");
   } else if (Hoption.System == kCYLINDRICAL) {
      if (Hoption.Lego ==  1) fLego->LegoCylindrical(1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoCylindrical(1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoCylindrical(1,nx,ny,"BF");
   } else if (Hoption.System == kSPHERICAL) {
      if (Hoption.Lego ==  1) fLego->LegoSpherical(0,1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoSpherical(0,1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoSpherical(0,1,nx,ny,"BF");
   } else if (Hoption.System == kRAPIDITY) {
      if (Hoption.Lego ==  1) fLego->LegoSpherical(1,1,nx,ny,"FB");
      if (Hoption.Lego == 11) fLego->LegoSpherical(1,1,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoSpherical(1,1,nx,ny,"BF");
   } else {
      if (Hoption.Lego ==  1) {
                              fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
                              fLego->LegoCartesian(90,nx,ny,"FB");}
      if (Hoption.Lego == 11) fLego->LegoCartesian(90,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoCartesian(90,nx,ny,"BF");
   }

   if (Hoption.Lego == 1 || Hoption.Lego == 11) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }
   if (Hoption.System == kCARTESIAN) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      if (Hoption.FrontBox) fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);
   if (Hoption.Zscale) PaintPalette();
   delete axis;
   delete fLego; fLego = 0;
}

//______________________________________________________________________________
void THistPainter::PaintLegoAxis(TGaxis *axis, Double_t ang)
{
   // Draw the axis for legos and surface plots

   static Double_t epsil = 0.001;

   Double_t cosa, sina;
   Double_t bmin, bmax;
   Double_t r[24]        /* was [3][8] */;
   Int_t ndivx, ndivy, ndivz, i;
   Double_t x1[3], x2[3], y1[3], y2[3], z1[3], z2[3], av[24]  /*  was [3][8] */;
   static char chopax[8], chopay[8], chopaz[8];
   Int_t ix1, ix2, iy1, iy2, iz1, iz2;
   Double_t rad;

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintLegoAxis", "no TView in current pad");
      return;
   }

   // In polar coordinates, draw a short line going from the external circle
   // corresponding to r = 1 up to r = 1.1
   if (Hoption.System == kPOLAR) {
      r[0] = 1;
      r[1] = 0;
      r[2] = 0;
      view->WCtoNDC(r, x1);
      r[0] = 1.1;
      r[1] = 0;
      r[2] = 0;
      view->WCtoNDC(r, x2);
      gPad->PaintLine(x1[0],x1[1],x2[0],x2[1]);
      return ;
   }

   if (Hoption.System != kCARTESIAN) return ;

   rad = TMath::ATan(1.) * 4. /180.;
   cosa = TMath::Cos(ang*rad);
   sina = TMath::Sin(ang*rad);

   view->AxisVertex(ang, av, ix1, ix2, iy1, iy2, iz1, iz2);
   for (i = 1; i <= 8; ++i) {
      r[i*3 - 3] = av[i*3 - 3] + av[i*3 - 2]*cosa;
      r[i*3 - 2] = av[i*3 - 2]*sina;
      r[i*3 - 1] = av[i*3 - 1];
   }

   view->WCtoNDC(&r[ix1*3 - 3], x1);
   view->WCtoNDC(&r[ix2*3 - 3], x2);
   view->WCtoNDC(&r[iy1*3 - 3], y1);
   view->WCtoNDC(&r[iy2*3 - 3], y2);
   view->WCtoNDC(&r[iz1*3 - 3], z1);
   view->WCtoNDC(&r[iz2*3 - 3], z2);

   view->SetAxisNDC(x1, x2, y1, y2, z1, z2);

   Double_t *rmin = view->GetRmin();
   Double_t *rmax = view->GetRmax();

   // Initialize the axis options
   if (x1[0] > x2[0]) strcpy(chopax, "SDH=+");
   else               strcpy(chopax, "SDH=-");
   if (y1[0] > y2[0]) strcpy(chopay, "SDH=+");
   else               strcpy(chopay, "SDH=-");
   strcpy(chopaz, "SDH+=");

   // Option LOG is required ?
   if (Hoption.Logx) strcat(chopax,"G");
   if (Hoption.Logy) strcat(chopay,"G");
   if (Hoption.Logz) strcat(chopaz,"G");

   // Initialize the number of divisions. If the
   // number of divisions is negative, option 'N' is required.
   ndivx = fXaxis->GetNdivisions();
   ndivy = fYaxis->GetNdivisions();
   ndivz = fZaxis->GetNdivisions();
   if (ndivx < 0) {
      ndivx = TMath::Abs(ndivx);
      strcat(chopax, "N");
   }
   if (ndivy < 0) {
      ndivy = TMath::Abs(ndivy);
      strcat(chopay, "N");
   }
   if (ndivz < 0) {
      ndivz = TMath::Abs(ndivz);
      strcat(chopaz, "N");
   }

   // Set Axis attributes.
   // The variable SCALE  rescales the VSIZ
   // in order to have the same label size for all angles.

   axis->SetLineWidth(1);

   // X axis drawing
   if (TMath::Abs(x1[0] - x2[0]) >= epsil || TMath::Abs(x1[1] - x2[1]) > epsil) {
      axis->ImportAxisAttributes(fXaxis);
      axis->SetLabelOffset(fXaxis->GetLabelOffset()+fXaxis->GetTickLength());
      if (Hoption.Logx) {
         bmin = TMath::Power(10, rmin[0]);
         bmax = TMath::Power(10, rmax[0]);
      } else {
         bmin = rmin[0];
         bmax = rmax[0];
      }
      // Option time display is required ?
      if (fXaxis->GetTimeDisplay()) {
         strcat(chopax,"t");
         if (strlen(fXaxis->GetTimeFormatOnly()) == 0) {
            axis->SetTimeFormat(fXaxis->ChooseTimeFormat(bmax-bmin));
         } else {
            axis->SetTimeFormat(fXaxis->GetTimeFormat());
         }
      }
      axis->SetOption(chopax);
      axis->PaintAxis(x1[0], x1[1], x2[0], x2[1], bmin, bmax, ndivx, chopax);
   }

   // Y axis drawing
   if (TMath::Abs(y1[0] - y2[0]) >= epsil || TMath::Abs(y1[1] - y2[1]) > epsil) {
      axis->ImportAxisAttributes(fYaxis);
      axis->SetLabelOffset(fYaxis->GetLabelOffset()+fYaxis->GetTickLength());

      if (fH->GetDimension() < 2) {
         strcpy(chopay, "V=+UN");
         ndivy = 0;
      }
      if (TMath::Abs(y1[0] - y2[0]) < epsil) {
         y2[0] = y1[0];
      }
      if (Hoption.Logy) {
         bmin = TMath::Power(10, rmin[1]);
         bmax = TMath::Power(10, rmax[1]);
      } else {
         bmin = rmin[1];
         bmax = rmax[1];
      }
      // Option time display is required ?
      if (fYaxis->GetTimeDisplay()) {
         strcat(chopay,"t");
         if (strlen(fYaxis->GetTimeFormatOnly()) == 0) {
            axis->SetTimeFormat(fYaxis->ChooseTimeFormat(bmax-bmin));
         } else {
            axis->SetTimeFormat(fYaxis->GetTimeFormat());
         }
      }
      axis->SetOption(chopay);
      axis->PaintAxis(y1[0], y1[1], y2[0], y2[1], bmin, bmax, ndivy, chopay);
   }

   // Z axis drawing
   if (TMath::Abs(z1[0] - z2[0]) >= 100*epsil || TMath::Abs(z1[1] - z2[1]) > 100*epsil) {
      axis->ImportAxisAttributes(fZaxis);
      if (Hoption.Logz) {
         bmin = TMath::Power(10, rmin[2]);
         bmax = TMath::Power(10, rmax[2]);
      } else {
         bmin = rmin[2];
         bmax = rmax[2];
      }
      // Option time display is required ?
      if (fZaxis->GetTimeDisplay()) {
         strcat(chopaz,"t");
         if (strlen(fZaxis->GetTimeFormatOnly()) == 0) {
            axis->SetTimeFormat(fZaxis->ChooseTimeFormat(bmax-bmin));
         } else {
            axis->SetTimeFormat(fZaxis->GetTimeFormat());
         }
      }
      axis->SetOption(chopaz);
      axis->PaintAxis(z1[0], z1[1], z2[0], z2[1], bmin, bmax, ndivz, chopaz);
   }

   fH->SetLineStyle(1);
}

//______________________________________________________________________________
void THistPainter::PaintPalette()
{
   // Paint the color palette on the right side of the pad.

   TPaletteAxis *palette = (TPaletteAxis*)fFunctions->FindObject("palette");
   TView *view = gPad->GetView();
   if (palette) {
      if (view) {
         if (!palette->TestBit(TPaletteAxis::kHasView)) {
            delete palette; palette = 0;
         }
      } else {
         if (palette->TestBit(TPaletteAxis::kHasView)) {
            delete palette; palette = 0;
         }
      }
   }

   if (!palette) {
      Double_t xup  = gPad->GetUxmax();
      Double_t x2   = gPad->PadtoX(gPad->GetX2());
      Double_t ymin = gPad->PadtoY(gPad->GetUymin());
      Double_t ymax = gPad->PadtoY(gPad->GetUymax());
      Double_t xr   = 0.05*(gPad->GetX2() - gPad->GetX1());
      Double_t xmin = gPad->PadtoX(xup +0.1*xr);
      Double_t xmax = gPad->PadtoX(xup + xr);
      if (xmax > x2) xmax = gPad->PadtoX(gPad->GetX2()-0.01*xr);
      palette = new TPaletteAxis(xmin,ymin,xmax,ymax,fH);
      fFunctions->Add(palette);
      palette->Paint();
   }
}

//______________________________________________________________________________
void THistPainter::PaintScatterPlot(Option_t *option)
{
   // Control function to draw a table as a scatter plot.
   //
   //       For each cell (i,j) a number of points proportional to the cell
   //       content is drawn.
   //       A maximum of kNMAX points per cell is drawn. If the maximum is above kNMAX
   //       contents are normalized to kNMAX. (kNMAX=2000)/
   //       if option is of the form "scat=ff", (eg scat=1.8, scat=1e-3), then
   //       ff is used as a scale factor to compute the number of dots.
   //       "scat=1" is the default.
   //Begin_Html
   /*
   <img src="gif/PaintScatterPlot.gif">
   */
   //End_Html

   fH->TAttMarker::Modify();

   Int_t k, marker;
   Double_t dz, z, xk,xstep, yk, ystep;
   Double_t scale = 1;
   Bool_t ltest  = kFALSE;
   Double_t zmax  = fH->GetMaximum();
   Double_t zmin  = fH->GetMinimum();
   if (zmin == 0 && zmax == 0) return;
   if (zmin == zmax) {
      zmax += 0.01*TMath::Abs(zmax);
      zmin -= 0.01*TMath::Abs(zmin);
   }
   Int_t ncells = (Hparam.ylast-Hparam.yfirst)*(Hparam.xlast-Hparam.xfirst);
   if (Hoption.Logz) {
      if (zmin > 0) zmin = TMath::Log10(zmin);
      else          zmin = 0;
      if (zmax > 0) zmax = TMath::Log10(zmax);
      else          zmax = 0;
      if (zmin == 0 && zmax == 0) return;
      dz = zmax - zmin;
      scale = 100/dz;
      if (ncells > 10000) scale /= 5;
      ltest = kTRUE;
   } else {
      dz = zmax - zmin;
      if (dz >= kNMAX || zmax < 1) {
         scale = (kNMAX-1)/dz;
         if (ncells > 10000) scale /= 5;
         ltest = kTRUE;
      }
   }
   if (fH->GetMinimumStored() == -1111) {
      Double_t yMARGIN = gStyle->GetHistTopMargin();
      if (gStyle->GetHistMinimumZero()) {
         if (zmin >= 0) zmin = 0;
         else           zmin -= yMARGIN*(zmax-zmin);
      } else {
         Double_t dzmin = yMARGIN*(zmax-zmin);
         if (zmin >= 0 && (zmin-dzmin <= 0)) zmin  = 0;
         else                                zmin -= dzmin;
      }
   }

   TString opt = option;
   opt.ToLower();
   if (opt.Contains("scat=")) {
      char optscat[100];
      strcpy(optscat,opt.Data());
      char *oscat = strstr(optscat,"scat=");
      char *blank = strstr(oscat," "); if (blank) *blank = 0;
      sscanf(oscat+5,"%lg",&scale);
   }
   UInt_t seedsave = gRandom->GetSeed();
   gRandom->SetSeed();
   marker=0;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         if (!IsInside(xk+0.5*xstep,yk+0.5*ystep)) continue;
         z     = fH->GetBinContent(bin);
         if (z < zmin) z = zmin;
         if (z > zmax) z = zmax;
         if (Hoption.Logz) {
            if (z > 0) z = TMath::Log10(z) - zmin;
         } else {
            z    -=  zmin;
         }
         if (z <= 0) continue;
         k = Int_t(z*scale);
         if (ltest) k++;
         if (k > 0) {
            for (Int_t loop=0; loop<k; loop++) {
               if (k+marker >= kNMAX) {
                  gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
                  marker=0;
               }
               fXbuf[marker] = (gRandom->Rndm(loop)*xstep) + xk;
               fYbuf[marker] = (gRandom->Rndm(loop)*ystep) + yk;
               if (Hoption.Logx){
                  if (fXbuf[marker] > 0) fXbuf[marker] = TMath::Log10(fXbuf[marker]);
                  else                   break;
               }
               if (Hoption.Logy){
                  if (fYbuf[marker] > 0) fYbuf[marker] = TMath::Log10(fYbuf[marker]);
                  else                  break;
               }
               if (fXbuf[marker] < gPad->GetUxmin()) break;
               if (fYbuf[marker] < gPad->GetUymin()) break;
               if (fXbuf[marker] > gPad->GetUxmax()) break;
               if (fYbuf[marker] > gPad->GetUymax()) break;
               marker++;
            }
         }
      }
   }
   if (marker > 0) gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
   gRandom->SetSeed(seedsave);

   if (Hoption.Zscale) PaintPalette();
}


//______________________________________________________________________________
void THistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
   // Draw the statistics box for 1D and profile histograms
   // The type of information printed in the histogram statistics box
   // can be selected via gStyle->SetOptStat(mode).
   // The parameter mode can be = iourmen  (default = 0001111)
   //    n = 1;  name of histogram is printed
   //    e = 1;  number of entries printed
   //    m = 1;  mean value printed
   //    m = 2;  mean and mean error values printed
   //    r = 1;  rms printed
   //    r = 2;  rms and rms error printed
   //    u = 1;  number of underflows printed
   //    o = 1;  number of overflows printed
   //    i = 1;  integral of bins printed
   //    s = 1;  skewness printed
   //    s = 2;  skewness and skewness error printed
   //    k = 1;  kurtosis printed
   //    k = 2;  kurtosis and kurtosis error printed
   //  Example: gStyle->SetOptStat(11);
   //           print only name of histogram and number of entries.
   //
   // The type of information about fit parameters printed in the histogram
   // statistics box can be selected via gStyle->SetOptFit(mode).
   // The parameter mode can be = pcev  (default = 0111)
   //    v = 1;  print name/values of parameters
   //    e = 1;  print errors (if e=1, v must be 1)
   //    c = 1;  print Chisquare/Number of degrees of freedom
   //    p = 1;  print Probability
   //    When "v"=1 is specified, only the non-fixed parameters are shown.
   //    When "v"=2 all parameters are shown.
   //  Example: gStyle->SetOptFit(1011);
   //           print fit probability, parameter names/values and errors.
   //
   //  Note: gStyle->SetOptFit(1) means "default value", so it is equivalent to
   //        gStyle->SetOptFit(111)
   //
   //  When option "same" is specified, the statistic box is not drawn.
   //  Specify option "sames" to force painting statistics with option "same"
   //  When option "sames" is given, one can use the following technique
   //  to move a previous "stats" box to a new position
   //  Root > TPaveStats *st = (TPaveStats*)gPad->GetPrimitive("stats")
   //  Root > st->SetX1NDC(newx1); //new x start position
   //  Root > st->SetX2NDC(newx2); //new x end position
   //  Root > newhist->Draw("sames")

   static char t[64];
   Int_t dofit;
   TPaveStats *stats  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TPaveStats::Class())) {
         stats = (TPaveStats*)obj;
         break;
      }
   }

   if (stats && dostat) {
      dofit  = stats->GetOptFit();
      dostat = stats->GetOptStat();
   } else {
      dofit  = gStyle->GetOptFit();
   }
   if (!dofit) fit = 0;
   if (dofit  == 1) dofit  =  111;
   if (dostat == 1) dostat = 1111;
   Int_t print_name    = dostat%10;
   Int_t print_entries = (dostat/10)%10;
   Int_t print_mean    = (dostat/100)%10;
   Int_t print_rms     = (dostat/1000)%10;
   Int_t print_under   = (dostat/10000)%10;
   Int_t print_over    = (dostat/100000)%10;
   Int_t print_integral= (dostat/1000000)%10;
   Int_t print_skew    = (dostat/10000000)%10;
   Int_t print_kurt    = (dostat/100000000)%10;
   Int_t nlines = print_name + print_entries + print_mean + print_rms + 
                  print_under + print_over + print_integral +
                  print_skew + print_kurt;
   Int_t print_fval    = dofit%10;
   Int_t print_ferrors = (dofit/10)%10;
   Int_t print_fchi2   = (dofit/100)%10;
   Int_t print_fprob   = (dofit/1000)%10;
   Int_t nlinesf = print_fval + print_fchi2 + print_fprob;
   if (fit) {
      if (print_fval < 2) nlinesf += fit->GetNumberFreeParameters();
      else                nlinesf += fit->GetNpar();
   }
   if (fH->InheritsFrom("TProfile")) nlinesf += print_mean + print_rms;

   // Pavetext with statistics
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) { delete stats; fFunctions->Remove(stats); }
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = (nlines+nlinesf)*gStyle->GetStatFontSize();
   if (stath <= 0 || 3 == (gStyle->GetStatFont()%10)) {
      stath = 0.25*(nlines+nlinesf)*gStyle->GetStatH();
   }
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");

      stats->SetParent(fH);
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetTextFont(gStyle->GetStatFont());
      if (gStyle->GetStatFont()%10 > 2)
         stats->SetTextSize(gStyle->GetStatFontSize());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetBit(kCanDelete);
      stats->SetBit(kMustCleanup);
   }
   if (print_name)  stats->AddText(fH->GetName());
   if (print_entries) {
      if (fH->GetEntries() < 1e7) sprintf(t,"%s = %-7d",gStringEntries.Data(),Int_t(fH->GetEntries()));
      else                        sprintf(t,"%s = %14.7g",gStringEntries.Data(),Float_t(fH->GetEntries()));
      stats->AddText(t);
   }
   char textstats[50];
   if (print_mean) {
      if (print_mean == 1) {
         sprintf(textstats,"%s  = %s%s",gStringMean.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetMean(1));
      } else {
         sprintf(textstats,"%s  = %s%s #pm %s%s",gStringMean.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetMean(1),fH->GetMeanError(1));
      }
      stats->AddText(t);
      if (fH->InheritsFrom("TProfile")) {
         if (print_mean == 1) {
            sprintf(textstats,"%s = %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat());
            sprintf(t,textstats,fH->GetMean(2));
         } else {
            sprintf(textstats,"%s = %s%s #pm %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat()
                                                      ,"%",stats->GetStatFormat());
            sprintf(t,textstats,fH->GetMean(2),fH->GetMeanError(2));
         }
         stats->AddText(t);
      }
   }
   if (print_rms) {
      if (print_rms == 1) {
         sprintf(textstats,"%s   = %s%s",gStringRMS.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetRMS(1));
      } else {
         sprintf(textstats,"%s   = %s%s #pm %s%s",gStringRMS.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetRMS(1),fH->GetRMSError(1));
      }
      stats->AddText(t);
      if(fH->InheritsFrom("TProfile")) {
         if (print_rms == 1) {
            sprintf(textstats,"%s = %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat());
            sprintf(t,textstats,fH->GetRMS(2));
         } else {
            sprintf(textstats,"%s = %s%s #pm %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat()
                                                     ,"%",stats->GetStatFormat());
            sprintf(t,textstats,fH->GetRMS(2),fH->GetRMSError(2));
         }
         stats->AddText(t);
      }
   }
   if (print_under) {
      sprintf(textstats,"%s = %s%s",gStringUnderflow.Data(),"%",stats->GetStatFormat());
      sprintf(t,textstats,fH->GetBinContent(0));
      stats->AddText(t);
   }
   if (print_over) {
      sprintf(textstats,"%s  = %s%s",gStringOverflow.Data(),"%",stats->GetStatFormat());
      sprintf(t,textstats,fH->GetBinContent(fXaxis->GetNbins()+1));
      stats->AddText(t);
   }
   if (print_integral) {
      sprintf(textstats,"%s = %s%s",gStringIntegral.Data(),"%",stats->GetStatFormat());
      sprintf(t,textstats,fH->Integral());
      stats->AddText(t);
   }
   if (print_skew) {
      if (print_skew == 1) {
         sprintf(textstats,"%s = %s%s",gStringSkewness.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetSkewness(1));
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringSkewness.Data(),"%",stats->GetStatFormat()
                                                     ,"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetSkewness(1),fH->GetSkewness(11));
      }
      stats->AddText(t);
   }
   if (print_kurt) {
      if (print_kurt == 1) {
         sprintf(textstats,"%s = %s%s",gStringKurtosis.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetKurtosis(1));
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringKurtosis.Data(),"%",stats->GetStatFormat()
                                                     ,"%",stats->GetStatFormat());
         sprintf(t,textstats,fH->GetKurtosis(1),fH->GetKurtosis(11));
      }
      stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNDF();
      sprintf(textstats,"#chi^{2} / ndf = %s%s / %d","%",stats->GetFitFormat(),ndf);
      sprintf(t,textstats,(Float_t)fit->GetChisquare());
      if (print_fchi2) stats->AddText(t);
      if (print_fprob) {
         sprintf(textstats,"Prob  = %s%s","%",stats->GetFitFormat());
         sprintf(t,textstats,(Float_t)TMath::Prob(fit->GetChisquare(),ndf));
         stats->AddText(t);
      }
      if (print_fval || print_ferrors) {
         Double_t parmin,parmax;
         for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
            fit->GetParLimits(ipar,parmin,parmax);
            if (print_fval < 2 && parmin*parmax != 0 && parmin >= parmax) continue;
            if (print_ferrors) {
               sprintf(textstats,"%-8s = %s%s #pm %s ",fit->GetParName(ipar), "%",stats->GetFitFormat(),
                       GetBestFormat(fit->GetParameter(ipar), fit->GetParError(ipar), stats->GetFitFormat()));
               sprintf(t,textstats,(Float_t)fit->GetParameter(ipar)
                               ,(Float_t)fit->GetParError(ipar));
            } else {
               sprintf(textstats,"%-8s = %s%s ",fit->GetParName(ipar),"%",stats->GetFitFormat());
               sprintf(t,textstats,(Float_t)fit->GetParameter(ipar));
            }
            t[63] = 0;
            stats->AddText(t);
         }
      }
   }

   if (!done) fFunctions->Add(stats);
   stats->Paint();
}

//______________________________________________________________________________
void THistPainter::PaintStat2(Int_t dostat, TF1 *fit)
{
   // Draw the statistics box for 2D histogram.
   // The type of information printed in the histogram statistics box
   // can be selected via gStyle->SetOptStat(mode).
   // The parameter mode can be = ourmen  (default = 001111)
   //    n = 1;  name of histogram is printed
   //    e = 1;  number of entries printed
   //    m = 1;  mean value printed
   //    m = 2;  mean and mean error values printed
   //    r = 1;  rms printed
   //    r = 2;  rms and rms error printed
   //    u = 1;  number of underflows printed
   //    o = 1;  number of overflows printed
   //    i = 1;  integral of bins printed
   //    s = 1;  skewness printed
   //    s = 2;  skewness and skewness error printed
   //    k = 1;  kurtosis printed
   //    k = 2;  kurtosis and kurtosis error printed
   //  Example: gStyle->SetOptStat(11);
   //           print only name of histogram and number of entries.

   if (fH->GetDimension() != 2) return;
   TH2 *h2 = (TH2*)fH;

   static char t[64];
   Int_t dofit;
   TPaveStats *stats  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TPaveStats::Class())) {
         stats = (TPaveStats*)obj;
         break;
      }
   }
   if (stats && dostat) {
      dofit  = stats->GetOptFit();
      dostat = stats->GetOptStat();
   } else {
      dofit  = gStyle->GetOptFit();
   }
   if (dostat == 1) dostat = 1111;
   Int_t print_name    = dostat%10;
   Int_t print_entries = (dostat/10)%10;
   Int_t print_mean    = (dostat/100)%10;
   Int_t print_rms     = (dostat/1000)%10;
   Int_t print_under   = (dostat/10000)%10;
   Int_t print_over    = (dostat/100000)%10;
   Int_t print_integral= (dostat/1000000)%10;
   Int_t print_skew    = (dostat/10000000)%10;
   Int_t print_kurt    = (dostat/100000000)%10;
   Int_t nlines = print_name + print_entries + 2*print_mean + 2*print_rms + print_integral;
   if (print_under || print_over) nlines += 3;

   // Pavetext with statistics
   if (!gStyle->GetOptFit()) fit = 0;
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) delete stats;
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = nlines*gStyle->GetStatFontSize();
   if (stath <= 0 || 3 == (gStyle->GetStatFont()%10)) {
      stath = 0.25*nlines*gStyle->GetStatH();
   }
   if (fit) stath += gStyle->GetStatH();
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");

      stats->SetParent(fH);
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetTextFont(gStyle->GetStatFont());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetBit(kCanDelete);
      stats->SetBit(kMustCleanup);
   }
   if (print_name)  stats->AddText(h2->GetName());
   if (print_entries) {
      if (h2->GetEntries() < 1e7) sprintf(t,"%s = %-7d",gStringEntries.Data(),Int_t(h2->GetEntries()));
      else                        sprintf(t,"%s = %14.7g",gStringEntries.Data(),Float_t(h2->GetEntries()));
      stats->AddText(t);
   }
   char textstats[50];
   if (print_mean) {
      if (print_mean == 1) {
         sprintf(textstats,"%s = %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetMean(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetMean(2));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetMean(1),h2->GetMeanError(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetMean(2),h2->GetMeanError(2));
         stats->AddText(t);
      }
   }
   if (print_rms) {
      if (print_rms == 1) {
         sprintf(textstats,"%s = %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetRMS(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetRMS(2));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetRMS(1),h2->GetRMSError(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetRMS(2),h2->GetRMSError(2));
         stats->AddText(t);
      }
   }
   if (print_integral) {
      sprintf(t,"%s  = %6.4g",gStringIntegral.Data(),h2->Integral());
      stats->AddText(t);
   }
   if (print_skew) {
      if (print_skew == 1) {
         sprintf(textstats,"%s = %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetSkewness(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetSkewness(2));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetSkewness(1),h2->GetSkewness(11));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetSkewness(2),h2->GetSkewness(12));
         stats->AddText(t);
      }
   }
   if (print_kurt) {
      if (print_kurt == 1) {
         sprintf(textstats,"%s = %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetKurtosis(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetKurtosis(2));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetKurtosis(1),h2->GetKurtosis(11));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h2->GetKurtosis(2),h2->GetKurtosis(12));
         stats->AddText(t);
      }
   }
   if (print_under || print_over) {
      //get 3*3 under/overflows for 2d hist
      Double_t unov[9];

      unov[0] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetNbins()+1);
      unov[1] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetNbins()+1);
      unov[2] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetNbins()+1,h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetNbins()+1);
      unov[3] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[4] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[5] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetNbins()+1,h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[6] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,0,h2->GetYaxis()->GetFirst()-1);
      unov[7] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),0,h2->GetYaxis()->GetFirst()-1);
      unov[8] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetNbins()+1,0,h2->GetYaxis()->GetFirst()-1);

      sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[0], (Int_t)unov[1], (Int_t)unov[2]);
      stats->AddText(t);
      if (h2->GetEntries() < 1e7)
         sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[3], (Int_t)unov[4], (Int_t)unov[5]);
      else
         sprintf(t, " %7d|%14.7g|%7d\n", (Int_t)unov[3], (Float_t)unov[4], (Int_t)unov[5]);
      stats->AddText(t);
      sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[6], (Int_t)unov[7], (Int_t)unov[8]);
      stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNDF();
      sprintf(t,"#chi^{2} / ndf = %6.4g / %d",(Float_t)fit->GetChisquare(),ndf);
      stats->AddText(t);
      for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
         sprintf(t,"%-8s = %5.4g #pm %5.4g ",fit->GetParName(ipar)
                                   ,(Float_t)fit->GetParameter(ipar)
                                   ,(Float_t)fit->GetParError(ipar));
         t[32] = 0;
         stats->AddText(t);
      }
   }

   if (!done) fFunctions->Add(stats);
   stats->Paint();
}

//______________________________________________________________________________
void THistPainter::PaintStat3(Int_t dostat, TF1 *fit)
{
   // Draw the statistics box for 3D histogram.
   // The type of information printed in the histogram statistics box
   // can be selected via gStyle->SetOptStat(mode).
   // The parameter mode can be = ourmen  (default = 001111)
   //    n = 1;  name of histogram is printed
   //    e = 1;  number of entries printed
   //    m = 1;  mean value printed
   //    m = 2;  mean and mean error values printed
   //    r = 1;  rms printed
   //    r = 2;  rms and rms error printed
   //    u = 1;  number of underflows printed
   //    o = 1;  number of overflows printed
   //    i = 1;  integral of bins printed
   //    s = 1;  skewness printed
   //    s = 2;  skewness and skewness error printed
   //    k = 1;  kurtosis printed
   //    k = 2;  kurtosis and kurtosis error printed
   //  Example: gStyle->SetOptStat(11);
   //           print only name of histogram and number of entries.

   if (fH->GetDimension() != 3) return;
   TH3 *h3 = (TH3*)fH;

   static char t[64];
   Int_t dofit;
   TPaveStats *stats  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TPaveStats::Class())) {
         stats = (TPaveStats*)obj;
         break;
      }
   }
   if (stats && dostat) {
      dofit  = stats->GetOptFit();
      dostat = stats->GetOptStat();
   } else {
      dofit  = gStyle->GetOptFit();
   }
   if (dostat == 1) dostat = 1111;
   Int_t print_name    = dostat%10;
   Int_t print_entries = (dostat/10)%10;
   Int_t print_mean    = (dostat/100)%10;
   Int_t print_rms     = (dostat/1000)%10;
   Int_t print_under   = (dostat/10000)%10;
   Int_t print_over    = (dostat/100000)%10;
   Int_t print_integral= (dostat/1000000)%10;
   Int_t print_skew    = (dostat/10000000)%10;
   Int_t print_kurt    = (dostat/100000000)%10;
   Int_t nlines = print_name + print_entries + 3*print_mean + 3*print_rms + print_integral;
   if (print_under || print_over) nlines += 3;

   // Pavetext with statistics
   if (!gStyle->GetOptFit()) fit = 0;
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) delete stats;
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = nlines*gStyle->GetStatFontSize();
   if (stath <= 0 || 3 == (gStyle->GetStatFont()%10)) {
      stath = 0.25*nlines*gStyle->GetStatH();
   }
   if (fit) stath += gStyle->GetStatH();
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");

      stats->SetParent(fH);
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetTextFont(gStyle->GetStatFont());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetBit(kCanDelete);
      stats->SetBit(kMustCleanup);
   }
   if (print_name)  stats->AddText(h3->GetName());
   if (print_entries) {
      if (h3->GetEntries() < 1e7) sprintf(t,"%s = %-7d",gStringEntries.Data(),Int_t(h3->GetEntries()));
      else                        sprintf(t,"%s = %14.7g",gStringEntries.Data(),Float_t(h3->GetEntries()));
      stats->AddText(t);
   }
   char textstats[50];
   if (print_mean) {
      if (print_mean == 1) {
         sprintf(textstats,"%s = %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetMean(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetMean(2));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringMeanZ.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetMean(3));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringMeanX.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetMean(1),h3->GetMeanError(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringMeanY.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetMean(2),h3->GetMeanError(2));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringMeanZ.Data(),"%",stats->GetStatFormat()
                                                   ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetMean(3),h3->GetMeanError(3));
         stats->AddText(t);
      }
   }
   if (print_rms) {
      if (print_rms == 1) {
         sprintf(textstats,"%s = %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetRMS(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetRMS(2));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringRMSZ.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetRMS(3));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringRMSX.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetRMS(1),h3->GetRMSError(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringRMSY.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetRMS(2),h3->GetRMSError(2));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringRMSZ.Data(),"%",stats->GetStatFormat()
                                                  ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetRMS(3),h3->GetRMSError(3));
         stats->AddText(t);
      }
   }
   if (print_integral) {
      sprintf(t,"%s  = %6.4g",gStringIntegral.Data(),h3->Integral());
      stats->AddText(t);
   }
   if (print_skew) {
      if (print_skew == 1) {
         sprintf(textstats,"%s = %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetSkewness(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetSkewness(2));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringSkewnessZ.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetSkewness(3));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringSkewnessX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetSkewness(1),h3->GetSkewness(11));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringSkewnessY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetSkewness(2),h3->GetSkewness(12));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringSkewnessZ.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetSkewness(3),h3->GetSkewness(13));
         stats->AddText(t);
      }
   }
   if (print_kurt) {
      if (print_kurt == 1) {
         sprintf(textstats,"%s = %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetKurtosis(1));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetKurtosis(2));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s",gStringKurtosisZ.Data(),"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetKurtosis(3));
         stats->AddText(t);
      } else {
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringKurtosisX.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetKurtosis(1),h3->GetKurtosis(11));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringKurtosisY.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetKurtosis(2),h3->GetKurtosis(12));
         stats->AddText(t);
         sprintf(textstats,"%s = %s%s #pm %s%s",gStringKurtosisZ.Data(),"%",stats->GetStatFormat()
                                                       ,"%",stats->GetStatFormat());
         sprintf(t,textstats,h3->GetKurtosis(3),h3->GetKurtosis(13));
         stats->AddText(t);
      }
   }
   if (print_under || print_over) {
      //get 3*3 under/overflows for 2d hist
      Double_t unov[9];

      unov[0] = h3->Integral(0,h3->GetXaxis()->GetFirst()-1,h3->GetYaxis()->GetLast()+1,h3->GetYaxis()->GetNbins()+1);
      unov[1] = h3->Integral(h3->GetXaxis()->GetFirst(),h3->GetXaxis()->GetLast(),h3->GetYaxis()->GetLast()+1,h3->GetYaxis()->GetNbins()+1);
      unov[2] = h3->Integral(h3->GetXaxis()->GetLast()+1,h3->GetXaxis()->GetNbins()+1,h3->GetYaxis()->GetLast()+1,h3->GetYaxis()->GetNbins()+1);
      unov[3] = h3->Integral(0,h3->GetXaxis()->GetFirst()-1,h3->GetYaxis()->GetFirst(),h3->GetYaxis()->GetLast());
      unov[4] = h3->Integral(h3->GetXaxis()->GetFirst(),h3->GetXaxis()->GetLast(),h3->GetYaxis()->GetFirst(),h3->GetYaxis()->GetLast());
      unov[5] = h3->Integral(h3->GetXaxis()->GetLast()+1,h3->GetXaxis()->GetNbins()+1,h3->GetYaxis()->GetFirst(),h3->GetYaxis()->GetLast());
      unov[6] = h3->Integral(0,h3->GetXaxis()->GetFirst()-1,0,h3->GetYaxis()->GetFirst()-1);
      unov[7] = h3->Integral(h3->GetXaxis()->GetFirst(),h3->GetXaxis()->GetLast(),0,h3->GetYaxis()->GetFirst()-1);
      unov[8] = h3->Integral(h3->GetXaxis()->GetLast()+1,h3->GetXaxis()->GetNbins()+1,0,h3->GetYaxis()->GetFirst()-1);

      sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[0], (Int_t)unov[1], (Int_t)unov[2]);
      stats->AddText(t);
      if (h3->GetEntries() < 1e7)
         sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[3], (Int_t)unov[4], (Int_t)unov[5]);
      else
         sprintf(t, " %7d|%14.7g|%7d\n", (Int_t)unov[3], (Float_t)unov[4], (Int_t)unov[5]);
      stats->AddText(t);
      sprintf(t, " %7d|%7d|%7d\n", (Int_t)unov[6], (Int_t)unov[7], (Int_t)unov[8]);
      stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNDF();
      sprintf(t,"#chi^{2} / ndf = %6.4g / %d",(Float_t)fit->GetChisquare(),ndf);
      stats->AddText(t);
      for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
         sprintf(t,"%-8s = %5.4g #pm %5.4g ",fit->GetParName(ipar)
                                   ,(Float_t)fit->GetParameter(ipar)
                                   ,(Float_t)fit->GetParError(ipar));
         t[32] = 0;
         stats->AddText(t);
      }
   }

   if (!done) fFunctions->Add(stats);
   stats->Paint();
}

//______________________________________________________________________________
void THistPainter::PaintSurface(Option_t *)
{
   // Control function to draw a table as a surface plot.
   //
   //     In a surface plot, cell contents are represented as a mesh.
   //     The height of the mesh is proportional to the cell content.
   //
   //     A surface plot can be represented in several coordinate systems.
   //     Default system is Cartesian coordinates.
   //     Possible systems are CYL,POL,SPH,PSR.
   //
   //     See THistPainter::Draw for a list of Surface options
   //     The following picture is generated with option SURF1.
   //
   //     See TStyle::SeTPaletteAxis to change the color palette.
   //     It is suggested to use palette 1 via the call
   //     gStyle->SetColorPalette(1)
   //
   //Begin_Html
   /*
   <img src="gif/PaintSurface1.gif">
   */
   //End_Html
   //     The following picture is generated with option SURF3.
   //Begin_Html
   /*
   <img src="gif/PaintSurface3.gif">
   */
   //End_Html

   const Double_t ydiff = 1;
   const Double_t yligh1 = 10;
   const Double_t qa = 0.15;
   const Double_t qd = 0.15;
   const Double_t qs = 0.8;
   Double_t fmin, fmax;
   Int_t raster = 0;
   Int_t irep   = 0;

   if (Hparam.zmin == 0 && Hparam.zmax == 0) {Hparam.zmin = -1; Hparam.zmax = 1;}
   Int_t   nx      = Hparam.xlast - Hparam.xfirst;
   Int_t   ny      = Hparam.ylast - Hparam.yfirst;
   Double_t zmin   = Hparam.zmin;
   Double_t zmax   = Hparam.zmax;
   Double_t xlab1  = Hparam.xmin;
   Double_t xlab2  = Hparam.xmax;
   Double_t ylab1  = Hparam.ymin;
   Double_t ylab2  = Hparam.ymax;
   Double_t dangle = 10*3.141592/180; //Delta angle for Rapidity option
   Double_t deltaz = TMath::Abs(zmin);
   if (deltaz == 0) deltaz = 1;
   if (zmin >= zmax) {
      zmin -= 0.5*deltaz;
      zmax += 0.5*deltaz;
   }
   Double_t z1c = zmin;
   Double_t z2c = zmin + (zmax-zmin)*(1+gStyle->GetHistTopMargin());
   //     Compute the lego limits and instantiate a lego object
   fXbuf[0] = -1;
   fYbuf[0] =  1;
   fXbuf[1] = -1;
   fYbuf[1] =  1;
   if (Hoption.System >= kPOLAR && (Hoption.Surf == 1 || Hoption.Surf == 13)) raster = 1;
   if (Hoption.System == kPOLAR) {
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
   } else if (Hoption.System == kCYLINDRICAL) {
      if (Hoption.Logy) {
         if (ylab1 > 0) fXbuf[2] = TMath::Log10(ylab1);
         else           fXbuf[2] = 0;
         if (ylab2 > 0) fYbuf[2] = TMath::Log10(ylab2);
         else           fYbuf[2] = 0;
      } else {
         fXbuf[2] = ylab1;
         fYbuf[2] = ylab2;
      }
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kSPHERICAL) {
      fXbuf[2] = -1;
      fYbuf[2] =  1;
      z1c = 0; z2c = 1;
   } else if (Hoption.System == kRAPIDITY) {
      fXbuf[2] = -1/TMath::Tan(dangle);
      fYbuf[2] =  1/TMath::Tan(dangle);
   } else {
      fXbuf[0] = xlab1;
      fYbuf[0] = xlab2;
      fXbuf[1] = ylab1;
      fYbuf[1] = ylab2;
      fXbuf[2] = z1c;
      fYbuf[2] = z2c;
   }

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf, Hoption.System);
   fLego->SetLineColor(fH->GetLineColor());
   fLego->SetFillColor(fH->GetFillColor());

   //          Create axis object

   TGaxis *axis = new TGaxis();

   //                  Initialize the levels on the Z axis
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = gStyle->GetNumberContours();
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);

   if (Hoption.Surf == 13 || Hoption.Surf == 15) fLego->SetMesh(3);
   if (Hoption.Surf == 12 || Hoption.Surf == 14) fLego->SetMesh(0);

   //     Close the surface in case of non cartesian coordinates.

   if (Hoption.System != kCARTESIAN) {nx++; ny++;}

   //     Now ready to draw the surface plot

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintSurface", "no TView in current pad");
      return;
   }

   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   //     Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();

   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   //     Draw the filled contour on top
   Int_t icol1 = fH->GetFillColor();

   Int_t hoption35 = Hoption.Surf;
   if (Hoption.Surf == 13 || Hoption.Surf == 15) {
      DefineColorLevels(ndivz);
      Hoption.Surf = 23;
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::SurfaceFunction);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"BF");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"BF");
      Hoption.Surf = hoption35;
      fLego->SetMesh(1);
   }

   if (raster) fLego->InitRaster(-1.1,-1.1,1.1,1.1,1000,800);
   else        fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 14) {
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }

   //     Gouraud Shading surface
   if (Hoption.Surf == 14) {
   //    Set light sources
      fLego->LightSource(0, ydiff, 0,0,0,irep);
      fLego->LightSource(1, yligh1 ,1,1,1,irep);
      fLego->SurfaceProperty(qa, qd, qs, 1, irep);
      fmin = ydiff*qa;
      fmax = fmin + (yligh1+0.1)*(qd+qs);
      Int_t nbcol = 28;
      icol1 = 201;
      Double_t dcol = 0.5/Double_t(nbcol);
      TColor *colref = gROOT->GetColor(fH->GetFillColor());
      Float_t r,g,b,hue,light,satur;
      colref->GetRGB(r,g,b);
      TColor::RGBtoHLS(r,g,b,hue,light,satur);
      TColor *acol;
      for (Int_t col=0;col<nbcol;col++) {
         acol = gROOT->GetColor(col+icol1);
         TColor::HLStoRGB(hue,.4+col*dcol,satur,r,g,b);
         acol->SetRGB(r,g,b);
      }
      fLego->Spectrum(nbcol, fmin, fmax, icol1, 1, irep);
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::GouraudFunction);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"BF");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"BF");
   } else if (Hoption.Surf == 15) {
   // The surface is not drawn in this case.
   } else {
   //     Draw the surface
      if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 16) {
         DefineColorLevels(ndivz);
      } else {
         fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      }
      fLego->SetSurfaceFunction(&TPainter3dAlgorithms::SurfaceFunction);
      if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceRaster1);
      if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode2);
      if (Hoption.System == kPOLAR) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfacePolar(1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SurfacePolar(1,nx,ny,"BF");
      } else if (Hoption.System == kCYLINDRICAL) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceCylindrical(1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      } else if (Hoption.System == kSPHERICAL) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceSpherical(0,1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      } else if (Hoption.System == kRAPIDITY) {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceSpherical(1,1,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      } else {
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         if (Hoption.Surf == 16) fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove3);
         if (Hoption.Surf ==  1 || Hoption.Surf == 13 || Hoption.Surf == 16) fLego->SurfaceCartesian(90,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SurfaceCartesian(90,nx,ny,"BF");
      }
   }

   if ((!Hoption.Same) &&
       (Hoption.Surf == 1 || Hoption.Surf == 13 || Hoption.Surf == 16)) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }
   if (Hoption.System == kCARTESIAN) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      if (Hoption.FrontBox) fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);

   if (Hoption.Zscale) PaintPalette();

   delete axis;
   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintTriangles(Option_t *option)
{
   // Control function to draw a table using Delaunay triangles.

   TGraphDelaunay *dt;

   // Check if fH contains a TGraphDelaunay
   TList *hl = fH->GetListOfFunctions();
   dt = (TGraphDelaunay*)hl->FindObject("TGraphDelaunay");
   if (!dt) return;

   // If needed, create a TGraphPainter
   if (!fGraphPainter) fGraphPainter = new TGraphPainter(dt);

   // Define the 3D view
   fXbuf[0] = Hparam.xmin;
   fYbuf[0] = Hparam.xmax;
   fXbuf[1] = Hparam.ymin;
   fYbuf[1] = Hparam.ymax;
   fXbuf[2] = Hparam.zmin;
   fYbuf[2] = Hparam.zmax;
   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);
   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintTriangles", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   Int_t irep;
   view->SetView(phideg, thedeg, psideg, irep);

   // Set color/style for back box
   fLego->SetFillStyle(gPad->GetFrameFillStyle());
   fLego->SetFillColor(gPad->GetFrameFillColor());
   fLego->TAttFill::Modify();
   Int_t backcolor = gPad->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);
   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   // Paint the Back Box if needed
   if (Hoption.BackBox && !Hoption.Same) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   // Paint the triangles
   fGraphPainter->Paint(option);

   // Paint the Front Box if needed
   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }

   // Paint the Axis if needed
   if (!Hoption.Axis && !Hoption.Same) {
      TGaxis *axis = new TGaxis();
      PaintLegoAxis(axis, 90);
      delete axis;
   }

   if (Hoption.Zscale) PaintPalette();

   delete fLego; fLego = 0;
}

//______________________________________________________________________________
void THistPainter::DefineColorLevels(Int_t ndivz)
{
   // Define the color levels used to paint legos, surfaces etc..

   Int_t i, irep;

   // Initialize the color levels
   if (ndivz >= 100) {
      Warning("PaintSurface", "too many color levels, %d, reset to 8", ndivz);
      ndivz = 8;
   }
   Double_t *funlevel = new Double_t[ndivz+1];
   Int_t *colorlevel = new Int_t[ndivz+1];
   Int_t theColor;
   Int_t ncolors = gStyle->GetNumberOfColors();
   for (i = 0; i < ndivz; ++i) {
      funlevel[i] = fH->GetContourLevelPad(i);
      theColor = Int_t((i+0.99)*Float_t(ncolors)/Float_t(ndivz));
      colorlevel[i] = gStyle->GetColorPalette(theColor);
   }
   colorlevel[ndivz] = gStyle->GetColorPalette(ncolors-1);
   fLego->ColorFunction(ndivz, funlevel, colorlevel, irep);
   delete [] colorlevel;
   delete [] funlevel;
}


//______________________________________________________________________________
void THistPainter::PaintTable(Option_t *option)
{
   // Control function to draw 2-D/3-D tables

   if (!TableInit()) return;  //fill Hparam structure with histo parameters

   PaintFrame();

   //if palette option not specified, delete a possible existing palette
   if (!Hoption.Zscale) {
      delete fFunctions->FindObject("palette");
   }

   if (fH->GetEntries() != 0 && Hoption.Axis<=0) {
      if (Hoption.Scat)    PaintScatterPlot(option);
      if (Hoption.Arrow)   PaintArrows(option);
      if (Hoption.Box)     PaintBoxes(option);
      if (Hoption.Color)   PaintColorLevels(option);
      if (Hoption.Contour) PaintContour(option);
      if (Hoption.Text)    PaintText(option);
      if (Hoption.Error >= 100)   Paint2DErrors(option);
   }

   if (Hoption.Lego) PaintLego(option);
   if (Hoption.Surf && !Hoption.Contour) PaintSurface(option);
   if (Hoption.Tri) PaintTriangles(option);

   if (!Hoption.Lego && !Hoption.Surf &&
       !Hoption.Tri  && !(Hoption.Error >= 100)) PaintAxis(kFALSE); // Draw the axes

   PaintTitle();    //    Draw histogram title

   TF1 *fit  = 0;
   TIter next(fFunctions);
   TObject *obj;
   while ((obj = next())) {
      if (obj->InheritsFrom(TF1::Class())) {
         fit = (TF1*)obj;
         break;
      }
   }
   if (Hoption.Same != 1) {
      if (!fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
         PaintStat2(gStyle->GetOptStat(),fit);
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintText(Option_t *)
{
   //  Control function to draw a histogram with the bin values
   //
   //  For each bin the content is printed.
   //
   //  The text attributes are:
   //      - text font  = current TStyle font
   //      - text size  = 0.02*padheight*markersize
   //      - text color = marker color
   //
   //   By default the format "g" is used. This format can be redefined
   //   by calling gStyle->SetPaintTextFormat
   //
   //Begin_Html
   /*
   <img src="gif/PaintText.gif">
   */
   //End_Html

   TText text;
   text.SetTextFont(gStyle->GetTextFont());
   text.SetTextColor(fH->GetMarkerColor());
   text.SetTextSize(0.02*fH->GetMarkerSize());

   Double_t x, y, z, angle = 0;
   char value[50];
   char format[32];
   sprintf(format,"%s%s","%",gStyle->GetPaintTextFormat());
   if (Hoption.Text >= 1000) angle = Hoption.Text-1000;

   // 1D histograms
   if (fH->GetDimension() == 1) {
      if (Hoption.Text ==  1) angle = 90;
      text.SetTextAngle(angle);
      text.SetTextAlign(11);
      if (angle == 90) text.SetTextAlign(12);
      if (angle ==  0) text.SetTextAlign(21);
      text.TAttText::Modify();
      Double_t dt = 0.02*(gPad->GetY2()-gPad->GetY1());
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         x = fH->GetXaxis()->GetBinCenter(i);
         y = fH->GetBinContent(i);
         sprintf(value,format,y);
         if (Hoption.Logx) {
            if (x > 0)  x  = TMath::Log10(x);
            else continue;
         }
         if (Hoption.Logy) {
            if (y > 0)  y  = TMath::Log10(y);
            else continue;
         }
         if (y >= gPad->GetY2()) continue;
         if (y <= gPad->GetY1()) continue;
         gPad->PaintText(x,y+0.2*dt,value);
      }

   // 2D histograms
   } else {
      text.SetTextAlign(22);
      if (Hoption.Text ==  1) angle = 0;
      text.SetTextAngle(angle);
      text.TAttText::Modify();
      for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
         y    = fYaxis->GetBinCenter(j);
         if (Hoption.Logy) {
            if (y > 0)  y  = TMath::Log10(y);
            else continue;
         }
         for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
            Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
            x    = fXaxis->GetBinCenter(i);
            if (Hoption.Logx) {
               if (x > 0)  x  = TMath::Log10(x);
               else continue;
            }
            if (!IsInside(x,y)) continue;
            z     = fH->GetBinContent(bin);
            if (z < Hparam.zmin) continue;
            sprintf(value,format,z);
            gPad->PaintText(x,y,value);
         }
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintTF3()
{
   // Control function to draw a 3d implicit functions.
   //
   // Thanks to the function ImplicitFunction of the TPainter3dAlgorithms class,
   // this function paints 3d representation of an implicit function.
   //
   // Example:
   //
   //   TF3 *fun3 = new TF3("fun3","sin(x*x+y*y+z*z-36)",-2,2,-2,2,-2,2);
   //   fun3->Draw();
   //
   //Begin_Html
   /*
   <img src="gif/PaintTF3.gif">
   */
   //End_Html

   Int_t irep;

   TGaxis *axis = new TGaxis();
   TAxis *xaxis = fH->GetXaxis();
   TAxis *yaxis = fH->GetYaxis();
   TAxis *zaxis = fH->GetZaxis();

   fXbuf[0] = xaxis->GetBinLowEdge(xaxis->GetFirst());
   fYbuf[0] = xaxis->GetBinUpEdge(xaxis->GetLast());
   fXbuf[1] = yaxis->GetBinLowEdge(yaxis->GetFirst());
   fYbuf[1] = yaxis->GetBinUpEdge(yaxis->GetLast());
   fXbuf[2] = zaxis->GetBinLowEdge(zaxis->GetFirst());
   fYbuf[2] = zaxis->GetBinUpEdge(zaxis->GetLast());

   fLego = new TPainter3dAlgorithms(fXbuf, fYbuf);

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintTF3", "no TView in current pad");
      return;
   }
   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

   fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.BackBox) {
      fLego->DefineGridLevels(fZaxis->GetNdivisions()%100);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove1);
      fLego->BackBox(90);
   }

   fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMode1);

   fLego->ImplicitFunction(fXbuf, fYbuf, fH->GetNbinsX(),
                                         fH->GetNbinsY(),
                                         fH->GetNbinsZ(), "BF");

   if (Hoption.FrontBox) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TPainter3dAlgorithms::DrawFaceMove2);
      fLego->FrontBox(90);
   }
   if (!Hoption.Axis && !Hoption.Same) PaintLegoAxis(axis, 90);

   PaintTitle();

   delete axis;
   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintTitle()
{
   // Draw the histogram title
   //
   // The title is drawn according to the title alignment returned by
   // GetTitleAlign. It is a 2 digits integer): hv
   //
   // where "h" is the horizontal alignment and "v" is the vertical alignment.
   // "h" can get the values 1 2 3 for left, center, and right
   // "v" can get the values 1 2 3 for bottom, middle and top
   //
   // for instance the default alignment is: 13 (left top)

   if (Hoption.Same) return;
   if (fH->TestBit(TH1::kNoTitle)) return;
   Int_t nt = strlen(fH->GetTitle());
   TPaveText *title = 0;
   TObject *obj;
   TIter next(gPad->GetListOfPrimitives());
   while ((obj = next())) {
      if (!obj->InheritsFrom(TPaveText::Class())) continue;
      title = (TPaveText*)obj;
      if (strcmp(title->GetName(),"title")) {title = 0; continue;}
      break;
   }
   if (nt == 0 || gStyle->GetOptTitle() <= 0) {
      if (title) delete title;
      return;
   }
   Double_t ht = gStyle->GetTitleH();
   Double_t wt = gStyle->GetTitleW();
   if (ht <= 0) ht = 1.1*gStyle->GetTitleFontSize();
   if (ht <= 0) ht = 0.05;
   if (wt <= 0) {
      TLatex l;
      l.SetTextSize(ht);
      l.SetTitle(fH->GetTitle());
      // adjustment in case the title has several lines (#splitline)
      ht = TMath::Max(ht, 1.2*l.GetYsize()/(gPad->GetY2() - gPad->GetY1()));
      Double_t wndc = l.GetXsize()/(gPad->GetX2() - gPad->GetX1());
      wt = TMath::Min(0.7, 0.02+wndc);
   }
   if (title) {
      TText *t0 = (TText*)title->GetLine(0);
      if (t0) {
         if (!strcmp(t0->GetTitle(),fH->GetTitle())) return;
         t0->SetTitle(fH->GetTitle());
         if (wt > 0) title->SetX2NDC(title->GetX1NDC()+wt);
      }
      return;
   }

   Int_t talh = gStyle->GetTitleAlign()/10;
   if (talh < 1) talh = 1; if (talh > 3) talh = 3;
   Int_t talv = gStyle->GetTitleAlign()%10;
   if (talv < 1) talv = 1; if (talv > 3) talv = 3;
   Double_t xpos, ypos;
   xpos = gStyle->GetTitleX();
   ypos = gStyle->GetTitleY();
   if (talh == 2) xpos = xpos-wt/2.;
   if (talh == 3) xpos = xpos-wt;
   if (talv == 2) ypos = ypos+ht/2.;
   if (talv == 1) ypos = ypos+ht;

   TPaveText *ptitle = new TPaveText(xpos, ypos-ht, xpos+wt, ypos,"blNDC");

   //     box with the histogram title
   ptitle->SetFillColor(gStyle->GetTitleFillColor());
   ptitle->SetFillStyle(gStyle->GetTitleStyle());
   ptitle->SetName("title");
   ptitle->SetBorderSize(gStyle->GetTitleBorderSize());
   ptitle->SetTextColor(gStyle->GetTitleTextColor());
   ptitle->SetTextFont(gStyle->GetTitleFont(""));
   if (gStyle->GetTitleFont("")%10 > 2)
      ptitle->SetTextSize(gStyle->GetTitleFontSize());
   ptitle->AddText(fH->GetTitle());
   ptitle->SetBit(kCanDelete);
   ptitle->Draw();
   ptitle->Paint();

}

//______________________________________________________________________________
void THistPainter::ProcessMessage(const char *mess, const TObject *obj)
{
   //  Process message mess

   if (!strcmp(mess,"SetF3")) {
      TPainter3dAlgorithms::SetF3((TF3*)obj);
   } else if (!strcmp(mess,"SetF3ClippingBoxOff")) {
      TPainter3dAlgorithms::SetF3ClippingBoxOff();
   } else if (!strcmp(mess,"SetF3ClippingBoxOn")) {
      TVectorD &v =  (TVectorD&)(*obj);
      Double_t xclip = v(0);
      Double_t yclip = v(1);
      Double_t zclip = v(2);
      TPainter3dAlgorithms::SetF3ClippingBoxOn(xclip,yclip,zclip);
   }
}

//______________________________________________________________________________
Int_t THistPainter::ProjectAitoff2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   // Static function
   // Convert Right Ascension, Declination to X,Y using an AITOFF projection.
   // This procedure can be used to create an all-sky map in Galactic
   // coordinates with an equal-area Aitoff projection.  Output map
   // coordinates are zero longitude centered.
   // Also called Hammer-Aitoff projection (first presented by Ernst von Hammer in 1892)
   // source: GMT
   // code from  Ernst-Jan Buis

   Double_t x, y;

   Double_t alpha2 = (l/2)*TMath::DegToRad();
   Double_t delta  = b*TMath::DegToRad();
   Double_t r2     = TMath::Sqrt(2.);
   Double_t f      = 2*r2/TMath::Pi();
   Double_t cdec   = TMath::Cos(delta);
   Double_t denom  = TMath::Sqrt(1. + cdec*TMath::Cos(alpha2));
   x      = cdec*TMath::Sin(alpha2)*2.*r2/denom;
   y      = TMath::Sin(delta)*r2/denom;
   x     *= TMath::RadToDeg()/f;
   y     *= TMath::RadToDeg()/f;
   //  x *= -1.; // for a skymap swap left<->right
   Al = x;
   Ab = y;

   return 0;
}

//______________________________________________________________________________
Int_t THistPainter::ProjectMercator2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   // Static function
   // Probably the most famous of the various map projections, the Mercator projection
   // takes its name from Mercator who presented it in 1569. It is a cylindrical, conformal projection
   // with no distortion along the equator.
   // The Mercator projection has been used extensively for world maps in which the distortion towards
   // the polar regions grows rather large, thus incorrectly giving the impression that, for example,
   // Greenland is larger than South America. In reality, the latter is about eight times the size of
   // Greenland. Also, the Former Soviet Union looks much bigger than Africa or South America. One may wonder
   // whether this illusion has had any influence on U.S. foreign policy.' (Source: GMT)
   // code from  Ernst-Jan Buis

   Al = l;
   Double_t aid = TMath::Tan((TMath::PiOver2() + b*TMath::DegToRad())/2);
   Ab = TMath::Log(aid);
   return 0;
}

//______________________________________________________________________________
Int_t THistPainter::ProjectSinusoidal2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   // Static function
   // code from  Ernst-Jan Buis

   Al = l*cos(b*TMath::DegToRad());
   Ab = b;
   return 0;
}

//______________________________________________________________________________
Int_t THistPainter::ProjectParabolic2xy(Double_t l, Double_t b, Double_t &Al, Double_t &Ab)
{
   // Static function
   // code from  Ernst-Jan Buis

   Al = l*(2.*TMath::Cos(2*b*TMath::DegToRad()/3) - 1);
   Ab = 180*TMath::Sin(b*TMath::DegToRad()/3);
   return 0;
}

//______________________________________________________________________________
void THistPainter::RecalculateRange()
{
   // Recompute the histogram range following graphics operations.

   if (Hoption.Same) return;

   //     Compute x,y range
   Double_t xmin = Hparam.xmin;
   Double_t xmax = Hparam.xmax;
   Double_t ymin = Hparam.ymin;
   Double_t ymax = Hparam.ymax;

   Double_t xmin_aid, ymin_aid, xmax_aid, ymax_aid;
   if (Hoption.Proj ==1) {
      // TODO : check x range not lower than -180 and not higher than 180
      THistPainter::ProjectAitoff2xy(Hparam.xmin, Hparam.ymin, xmin_aid, ymin_aid);
      THistPainter::ProjectAitoff2xy(Hparam.xmin, Hparam.ymax, xmin,     ymax_aid);
      THistPainter::ProjectAitoff2xy(Hparam.xmax, Hparam.ymax, xmax_aid, ymax);
      THistPainter::ProjectAitoff2xy(Hparam.xmax, Hparam.ymin, xmax,     ymin);

      if (xmin > xmin_aid) xmin = xmin_aid;
      if (ymin > ymin_aid) ymin = ymin_aid;
      if (xmax < xmax_aid) xmax = xmax_aid;
      if (ymax < ymax_aid) ymax = ymax_aid;
      if (Hparam.ymin<0 && Hparam.ymax>0) {
         // there is an  'equator', check its range in the plot..
         THistPainter::ProjectAitoff2xy(Hparam.xmin*0.9999, 0, xmin_aid, ymin_aid);
         THistPainter::ProjectAitoff2xy(Hparam.xmax*0.9999, 0, xmax_aid, ymin_aid);
         if (xmin >xmin_aid) xmin = xmin_aid;
         if (xmax <xmax_aid) xmax = xmax_aid;
      }
      if (Hparam.xmin<0 && Hparam.xmax>0) {
         THistPainter::ProjectAitoff2xy(0, Hparam.ymin, xmin_aid, ymin_aid);
         THistPainter::ProjectAitoff2xy(0, Hparam.ymax, xmax_aid, ymax_aid);
         if (ymin >ymin_aid) ymin = ymin_aid;
         if (ymax <ymax_aid) ymax = ymax_aid;
      }
   } else if ( Hoption.Proj ==2) {
      if (Hparam.ymin <= -90 | Hparam.ymax >=90) {
         Warning("Mercator Projection", "Latitude out of range %f or %f", Hparam.ymin, Hparam.ymax);
         Hoption.Proj = 0;
      } else {
         THistPainter::ProjectMercator2xy(Hparam.xmin, Hparam.ymin, xmin, ymin);
         THistPainter::ProjectMercator2xy(Hparam.xmax, Hparam.ymax, xmax, ymax);
      }
   } else if (Hoption.Proj == 3) {
      THistPainter::ProjectSinusoidal2xy(Hparam.xmin, Hparam.ymin, xmin_aid, ymin_aid);
      THistPainter::ProjectSinusoidal2xy(Hparam.xmin, Hparam.ymax, xmin,     ymax_aid);
      THistPainter::ProjectSinusoidal2xy(Hparam.xmax, Hparam.ymax, xmax_aid, ymax);
      THistPainter::ProjectSinusoidal2xy(Hparam.xmax, Hparam.ymin, xmax,     ymin);

      if (xmin > xmin_aid) xmin = xmin_aid;
      if (ymin > ymin_aid) ymin = ymin_aid;
      if (xmax < xmax_aid) xmax = xmax_aid;
      if (ymax < ymax_aid) ymax = ymax_aid;
      if (Hparam.ymin<0 && Hparam.ymax>0) {
         THistPainter::ProjectSinusoidal2xy(Hparam.xmin, 0, xmin_aid, ymin_aid);
         THistPainter::ProjectSinusoidal2xy(Hparam.xmax, 0, xmax_aid, ymin_aid);
         if (xmin >xmin_aid) xmin = xmin_aid;
         if (xmax <xmax_aid) xmax = xmax_aid;
      }
      if (Hparam.xmin<0 && Hparam.xmax>0) {
         THistPainter::ProjectSinusoidal2xy(0,Hparam.ymin, xmin_aid, ymin_aid);
         THistPainter::ProjectSinusoidal2xy(0, Hparam.ymax, xmax_aid, ymin_aid);
         if (ymin >ymin_aid) ymin = ymin_aid;
         if (ymax <ymax_aid) ymax = ymax_aid;
      }
   } else if (Hoption.Proj == 4) {
      THistPainter::ProjectParabolic2xy(Hparam.xmin, Hparam.ymin, xmin_aid, ymin_aid);
      THistPainter::ProjectParabolic2xy(Hparam.xmin, Hparam.ymax, xmin,     ymax_aid);
      THistPainter::ProjectParabolic2xy(Hparam.xmax, Hparam.ymax, xmax_aid, ymax);
      THistPainter::ProjectParabolic2xy(Hparam.xmax, Hparam.ymin, xmax,     ymin);

      if (xmin > xmin_aid) xmin = xmin_aid;
      if (ymin > ymin_aid) ymin = ymin_aid;
      if (xmax < xmax_aid) xmax = xmax_aid;
      if (ymax < ymax_aid) ymax = ymax_aid;
      if (Hparam.ymin<0 && Hparam.ymax>0) {
         THistPainter::ProjectParabolic2xy(Hparam.xmin, 0, xmin_aid, ymin_aid);
         THistPainter::ProjectParabolic2xy(Hparam.xmax, 0, xmax_aid, ymin_aid);
         if (xmin >xmin_aid) xmin = xmin_aid;
         if (xmax <xmax_aid) xmax = xmax_aid;
      }
      if (Hparam.xmin<0 && Hparam.xmax>0) {
         THistPainter::ProjectParabolic2xy(0, Hparam.ymin, xmin_aid, ymin_aid);
         THistPainter::ProjectParabolic2xy(0, Hparam.ymax, xmax_aid, ymin_aid);
         if (ymin >ymin_aid) ymin = ymin_aid;
         if (ymax <ymax_aid) ymax = ymax_aid;
      }
   }
   Hparam.xmin= xmin;
   Hparam.xmax= xmax;
   Hparam.ymin= ymin;
   Hparam.ymax= ymax;

   Double_t dx   = xmax-xmin;
   Double_t dy   = ymax-ymin;
   Double_t dxr  = dx/(1 - gPad->GetLeftMargin()   - gPad->GetRightMargin());
   Double_t dyr  = dy/(1 - gPad->GetBottomMargin() - gPad->GetTopMargin());

   // Range() could change the size of the pad pixmap and therefore should
   // be called before the other paint routines
   gPad->Range(xmin - dxr*gPad->GetLeftMargin(),
                      ymin - dyr*gPad->GetBottomMargin(),
                      xmax + dxr*gPad->GetRightMargin(),
                      ymax + dyr*gPad->GetTopMargin());
   gPad->RangeAxis(xmin, ymin, xmax, ymax);
}

//______________________________________________________________________________
void THistPainter::SetHistogram(TH1 *h)
{
   //  set current histogram to h

   if (h == 0)  return;
   fH = h;
   fXaxis = h->GetXaxis();
   fYaxis = h->GetYaxis();
   fZaxis = h->GetZaxis();
   fFunctions = fH->GetListOfFunctions();
}

//______________________________________________________________________________
Int_t THistPainter::TableInit()
{
   // Initialize various options to draw tables.

   static const char *where = "TableInit";

   Int_t first, last;
   Double_t yMARGIN= gStyle->GetHistTopMargin();
   Double_t zmin, zmax;
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

   //    -----------------  Compute X axis parameters
   first           = fXaxis->GetFirst();
   last            = fXaxis->GetLast();
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xmin     = Hparam.xlowedge;
   Hparam.xmax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

   //       if log scale in X, replace xmin,max by the log
   if (Hoption.Logx) {
   //   find the first edge of a bin that is > 0
      if (Hparam.xlowedge <=0 ) {
         Hparam.xlowedge = fXaxis->GetBinUpEdge(fXaxis->FindFixBin(0.01*Hparam.xbinsize));
         Hparam.xmin  = Hparam.xlowedge;
      }
      if (Hparam.xmin <=0 || Hparam.xmax <=0) {
         Error(where, "cannot set X axis to log scale");
         return 0;
      }
      Hparam.xfirst= fXaxis->FindFixBin(Hparam.xmin);
      if (Hparam.xfirst < first) Hparam.xfirst = first;
      Hparam.xlast = fXaxis->FindFixBin(Hparam.xmax);
      if (Hparam.xlast > last) Hparam.xlast = last;
      Hparam.xmin  = TMath::Log10(Hparam.xmin);
      Hparam.xmax  = TMath::Log10(Hparam.xmax);
   }

   //    -----------------  Compute Y axis parameters
   first           = fYaxis->GetFirst();
   last            = fYaxis->GetLast();
   Hparam.ylast    = last;
   Hparam.yfirst   = first;
   Hparam.ylowedge = fYaxis->GetBinLowEdge(first);
   Hparam.ybinsize = fYaxis->GetBinWidth(first);
   if (!Hparam.ybinsize) Hparam.ybinsize = 1;
   Hparam.ymin     = Hparam.ylowedge;
   Hparam.ymax     = fYaxis->GetBinLowEdge(last)+fYaxis->GetBinWidth(last);

   //       if log scale in Y, replace ymin,max by the log
   if (Hoption.Logy) {
      if (Hparam.ylowedge <=0 ) {
         Hparam.ylowedge = fYaxis->GetBinUpEdge(fYaxis->FindFixBin(0.01*Hparam.ybinsize));
         Hparam.ymin  = Hparam.ylowedge;
      }
      if (Hparam.ymin <=0 || Hparam.ymax <=0) {
         Error(where, "cannot set Y axis to log scale");
         return 0;
      }
      Hparam.yfirst= fYaxis->FindFixBin(Hparam.ymin);
      if (Hparam.yfirst < first) Hparam.yfirst = first;
      Hparam.ylast = fYaxis->FindFixBin(Hparam.ymax);
      if (Hparam.ylast > last) Hparam.ylast = last;
      Hparam.ymin  = TMath::Log10(Hparam.ymin);
      Hparam.ymax  = TMath::Log10(Hparam.ymax);
   }


   //    -----------------  Compute Z axis parameters
   Double_t bigp = TMath::Power(10,32);
   zmax = -bigp;
   zmin = bigp;
   Double_t c1, e1;
   Double_t allchan = 0;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         c1 = fH->GetCellContent(i,j);
         zmax = TMath::Max(zmax,c1);
         if (Hoption.Error) {
            e1 = fH->GetCellError(i,j);
            zmax = TMath::Max(zmax,c1+e1);
         }
         zmin = TMath::Min(zmin,c1);
         allchan += c1;
      }
   }

   //     Take into account maximum , minimum

   if (maximum) zmax = fH->GetMaximumStored();
   if (minimum) zmin = fH->GetMinimumStored();
   if (Hoption.Logz && zmax <= 0) {
      if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", zmax);
      return 0;
   }
   if (zmin >= zmax && !Hoption.Plus) {
      if (Hoption.Logz) {
         if (zmax > 0) zmin = 0.001*zmax;
         else {
            if (!Hoption.Same) Error(where, "log scale is requested but maximum is less or equal 0 (%f)", zmax);
            return 0;
         }
      }
   }

   //     take into account normalization factor
   Hparam.allchan = allchan;
   Double_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   zmax = factor*zmax;
   zmin = factor*zmin;
   c1 = zmax;
   if (TMath::Abs(zmin) > TMath::Abs(c1)) c1 = zmin;

   //         For log scales, histogram coordinates are log10(ymin) and
   //         log10(ymax). Final adjustment (if not option "Same")
   //         or "+" for ymax) of ymax and ymin for logarithmic scale, if
   //         Maximum and Minimum are not defined.
   if (Hoption.Logz) {
      if (zmin <= 0) {
         zmin = TMath::Min((Double_t)1, (Double_t)0.001*zmax);
         fH->SetMinimum(zmin);
      }
      zmin = TMath::Log10(zmin);
      if (!minimum) zmin += TMath::Log10(0.5);
      zmax = TMath::Log10(zmax);
      if (!maximum && !Hoption.Plus) zmax += TMath::Log10(2*(0.9/0.95));
      goto LZMIN;
   }

   //         final adjustment of YMAXI for linear scale (if not option "Same"):
   //         decrease histogram height to MAX% of allowed height if HMAXIM
   //         has not been called.
   //         MAX% is the value in percent which has been set in HPLSET
   //         (default is 90%).
   if (!maximum && !Hoption.Plus) {
      zmax += yMARGIN*(zmax-zmin);
   }

   //         final adjustment of ymin for linear scale.
   //         if minimum is not set , then ymin is set to zero if >0
   //         or to ymin - yMARGIN if <0.
   if (!minimum) {
      if (gStyle->GetHistMinimumZero()) {
         if (zmin >= 0) zmin = 0;
         else           zmin -= yMARGIN*(zmax-zmin);
      } else {
         Double_t dzmin = yMARGIN*(zmax-zmin);
         if (zmin >= 0 && (zmin-dzmin <= 0)) zmin  = 0;
         else                                zmin -= dzmin;
      }
   }

LZMIN:
   Hparam.zmin = zmin;
   Hparam.zmax = zmax;

   //     Set bar offset and width
   Hparam.baroffset = fH->GetBarOffset();
   Hparam.barwidth  = fH->GetBarWidth();

   return 1;
}

//______________________________________________________________________________
const char * THistPainter::GetBestFormat(Double_t v, Double_t e, const char *f)
{
   // This function returns the best format to print the error value (e)
   // knowing the parameter value (v) and the format (f) used to print it.

   static char ef[20];
   char tf[20], tv[64];

   // print v with the format f in tv.
   sprintf(tf,"%s%s","%",f);
   sprintf(tv,tf,v);

   // Analyse tv.
   TString sv = tv;
   int ie = sv.Index("e");
   int iE = sv.Index("E");
   int id = sv.Index(".");

   // v has been printed with the exponent notation.
   // There is 2 cases, the exponent is positive or negative
   if (ie >= 0 || iE >= 0) {
      if (sv.Index("+") >= 0) {
         if (e < 1) {
            sprintf(ef,"%s.1f","%");
         } else {
            sprintf(ef,"%s.0f","%");
         }
      } else {
         if (ie >= 0) {
            sprintf(ef,"%s.%de","%",ie-id-1);
         } else {
            sprintf(ef,"%s.%dE","%",iE-id-1);
         }
      }

   // The is not '.' in tv. e will be printed with one decimal digit.
   } else if (id < 0) {
      sprintf(ef,"%s.1f","%");

   // There is a '.' in tv and no exponent notation. e's decimal part will
   // have the same number of digits as v's one.
   } else {
      sprintf(ef,"%s.%df","%",sv.Length()-id-1);
   }

   return ef;
}

//______________________________________________________________________________
void THistPainter::SetShowProjection(const char *option,Int_t nbins)
{
   // Set projection onto X

   if (fShowProjection) return;
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("x"))  fShowProjection = 1+100*nbins;
   if (opt.Contains("y"))  fShowProjection = 2+100*nbins;
   if (opt.Contains("z"))  fShowProjection = 3+100*nbins;
   if (opt.Contains("xy")) fShowProjection = 4+100*nbins;
   if (opt.Contains("yx")) fShowProjection = 5+100*nbins;
   if (opt.Contains("xz")) fShowProjection = 6+100*nbins;
   if (opt.Contains("zx")) fShowProjection = 7+100*nbins;
   if (opt.Contains("yz")) fShowProjection = 8+100*nbins;
   if (opt.Contains("zy")) fShowProjection = 9+100*nbins;
   if (fShowProjection < 4) fShowOption = option+1;
   else                     fShowOption = option+2;
   if (!gROOT->GetMakeDefCanvas()) return;
   (gROOT->GetMakeDefCanvas())();
   gPad->SetName(Form("c_projection_%d",fShowProjection));
   gPad->SetGrid();
}

//______________________________________________________________________________
void THistPainter::ShowProjectionX(Int_t /*px*/, Int_t py)
{
   // Show projection onto X
   Int_t nbins = (Int_t)fShowProjection/100;
   gPad->SetDoubleBuffer(0); // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert); // set the drawing mode to XOR mode
      
   // Erase old position and draw a line at current position
   static int pyold1 = 0;
   static int pyold2 = 0;
   float uxmin = gPad->GetUxmin();
   float uxmax = gPad->GetUxmax();
   int pxmin   = gPad->XtoAbsPixel(uxmin);
   int pxmax   = gPad->XtoAbsPixel(uxmax);
   Float_t upy = gPad->AbsPixeltoY(py);
   Float_t y   = gPad->PadtoY(upy);
   Int_t biny1 = fH->GetYaxis()->FindBin(y);
   Int_t biny2 = TMath::Min(biny1+nbins-1, fH->GetYaxis()->GetNbins());
   Int_t py1   = gPad->YtoAbsPixel(fH->GetYaxis()->GetBinLowEdge(biny1));
   Int_t py2   = gPad->YtoAbsPixel(fH->GetYaxis()->GetBinUpEdge(biny2));

   if (pyold1 || pyold2) gVirtualX->DrawBox(pxmin,pyold1,pxmax,pyold2,TVirtualX::kFilled);
   gVirtualX->DrawBox(pxmin,py1,pxmax,py2,TVirtualX::kFilled);
   pyold1 = py1;
   pyold2 = py2;

   // Create or set the new canvas proj x
   TVirtualPad *padsav = gPad;
   TVirtualPad *c = (TVirtualPad*)gROOT->GetListOfCanvases()->FindObject(Form("c_projection_%d",fShowProjection));
   if (c) {
      c->Clear();
   } else {
      fShowProjection = 0;
      pyold1 = 0;
      pyold2 = 0;
      return;
   }
   c->cd();

   // Draw slice corresponding to mouse position
   TH1D *hp = ((TH2*)fH)->ProjectionX("_px", biny1, biny2);
   hp->SetFillColor(38);
   if (biny1 == biny2) hp->SetTitle(Form("ProjectionX of biny=%d", biny1));
   else hp->SetTitle(Form("ProjectionX of biny=[%d,%d]", biny1,biny2));
   hp->SetXTitle(fH->GetXaxis()->GetTitle());
   hp->SetYTitle("Number of Entries");
   hp->Draw();
   c->Update();
   padsav->cd();
}

//______________________________________________________________________________
void THistPainter::ShowProjectionY(Int_t px, Int_t /*py*/)
{
   // Show projection onto Y
   Int_t nbins = (Int_t)fShowProjection/100;
   gPad->SetDoubleBuffer(0);             // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert);  // set the drawing mode to XOR mode
      
   // Erase old position and draw a line at current position
   static int pxold1 = 0;
   static int pxold2 = 0;
   float uymin = gPad->GetUymin();
   float uymax = gPad->GetUymax();
   int pymin   = gPad->YtoAbsPixel(uymin);
   int pymax   = gPad->YtoAbsPixel(uymax);
   Float_t upx = gPad->AbsPixeltoX(px);
   Float_t x   = gPad->PadtoX(upx);
   Int_t binx1 = fH->GetXaxis()->FindBin(x);
   Int_t binx2 = TMath::Min(binx1+nbins-1, fH->GetXaxis()->GetNbins());
   Int_t px1   = gPad->XtoAbsPixel(fH->GetXaxis()->GetBinLowEdge(binx1));
   Int_t px2   = gPad->XtoAbsPixel(fH->GetXaxis()->GetBinUpEdge(binx2));
   
   if (pxold1 || pxold2) gVirtualX->DrawBox(pxold1,pymin,pxold2,pymax,TVirtualX::kFilled);
   gVirtualX->DrawBox(px1,pymin,px2,pymax,TVirtualX::kFilled);
   pxold1 = px1;
   pxold2 = px2;

   // Create or set the new canvas proj y
   TVirtualPad *padsav = gPad;
   TVirtualPad *c = (TVirtualPad*)gROOT->GetListOfCanvases()->FindObject(Form("c_projection_%d",fShowProjection));
   if(c) {
      c->Clear();
   } else {
      fShowProjection = 0;
      pxold1 = 0;
      pxold2 = 0;
      return;
   }
   c->cd();

   // Draw slice corresponding to mouse position
   TH1D *hp = ((TH2*)fH)->ProjectionY("_py", binx1, binx2);
   hp->SetFillColor(38);
   if (binx1 == binx2) hp->SetTitle(Form("ProjectionY of binx=%d", binx1));
   else hp->SetTitle(Form("ProjectionY of binx=[%d,%d]", binx1,binx2));
   hp->SetXTitle(fH->GetYaxis()->GetTitle());
   hp->SetYTitle("Number of Entries");
   hp->Draw();
   c->Update();
   padsav->cd();
}

//______________________________________________________________________________
void THistPainter::ShowProjection3(Int_t px, Int_t py)
{
   // Show projection (specified by fShowProjection) of a TH3
   // The drawing option for the projection is in fShowOption.
   //
   // First implementation; R.Brun
   // Full implementation: Tim Tran (timtran@jlab.org)  April 2006
   
   Int_t nbins=(Int_t)fShowProjection/100; //decode nbins
   if (fH->GetDimension() < 3) {
      if (fShowProjection%100 == 1) {ShowProjectionX(px,py); return;}
      if (fShowProjection%100 == 2) {ShowProjectionY(px,py); return;}
   }  

   gPad->SetDoubleBuffer(0);             // turn off double buffer mode
   gVirtualX->SetDrawMode(TVirtualX::kInvert);  // set the drawing mode to XOR mode
      
   // Erase old position and draw a line at current position
   TView *view = gPad->GetView();
   TH3 *h3 = (TH3*)fH;
   TAxis *xaxis = h3->GetXaxis();
   TAxis *yaxis = h3->GetYaxis();
   TAxis *zaxis = h3->GetZaxis();
   Double_t u[3],xx[3];
   
   static TPoint line1[2];//store end points of a line, initialised 0 by default
   static TPoint line2[2];// second line when slice thickness > 1 bin thickness 
   static TPoint line3[2];
   static TPoint line4[2];
   static TPoint endface1[5];
   static TPoint endface2[5];
   static TPoint rect1[5];//store vertices of the polyline (rectangle), initialsed 0 by default
   static TPoint rect2[5];// second rectangle when slice thickness > 1 bin thickness

   Double_t value1=0, value2=0; //bin values cooresponding to the lower and upper bins of the slice  
   Double_t uxmin = gPad->GetUxmin();
   Double_t uxmax = gPad->GetUxmax();
   Double_t uymin = gPad->GetUymin();
   Double_t uymax = gPad->GetUymax();
 
   int pxmin = gPad->XtoAbsPixel(uxmin);
   int pxmax = gPad->XtoAbsPixel(uxmax);
   int pymin = gPad->YtoAbsPixel(uymin);
   int pymax = gPad->YtoAbsPixel(uymax);
   Double_t cx    = (pxmax-pxmin)/(uxmax-uxmin);
   Double_t cy    = (pymax-pymin)/(uymax-uymin);
   TVirtualPad *padsav = gPad;
   TVirtualPad *c = (TVirtualPad*)gROOT->GetListOfCanvases()->FindObject(Form("c_projection_%d",fShowProjection));
   if(!c) {   
      fShowProjection = 0;
      return;
   }
     
   switch ((Int_t)fShowProjection%100) {
      case 1:
         // "x"      
         {
            Int_t firstY = yaxis->GetFirst();
            Int_t lastY  = yaxis->GetLast();
            Int_t biny = firstY + Int_t((lastY-firstY)*(px-pxmin)/(pxmax-pxmin));
            yaxis->SetRange(biny,biny+nbins-1);         
            Int_t firstZ = zaxis->GetFirst();
            Int_t lastZ  = zaxis->GetLast();
            Int_t binz = firstZ + Int_t((lastZ-firstZ)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);                   
            if(line1[0].GetX()) gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1 && line1[0].GetX()) {
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            xx[0] = xaxis->GetXmin();
            xx[2] = zaxis->GetBinCenter(binz);         
            xx[1] = yaxis->GetBinCenter(biny); 
            view->WCtoNDC(xx,u);         
            line1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
            xx[0] = xaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            line1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
            gVirtualX->DrawPolyLine(2,line1);          
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);         
               xx[1] = yaxis->GetBinCenter(biny); 
               view->WCtoNDC(xx,u);         
               line2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               
               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);         
               xx[1] = yaxis->GetBinCenter(biny+nbins-1); 
               view->WCtoNDC(xx,u);         
               line3[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line3[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
                 
               xx[0] = xaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz);         
               xx[1] = yaxis->GetBinCenter(biny+nbins-1); 
               view->WCtoNDC(xx,u);         
               line4[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line4[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[1].SetY(pymin + Int_t((u[1]-uymin)*cy));   
               
               endface1[0].SetX(line1[0].GetX());
               endface1[0].SetY(line1[0].GetY());
               endface1[1].SetX(line2[0].GetX());
               endface1[1].SetY(line2[0].GetY());
               endface1[2].SetX(line3[0].GetX());
               endface1[2].SetY(line3[0].GetY());
               endface1[3].SetX(line4[0].GetX());
               endface1[3].SetY(line4[0].GetY());
               endface1[4].SetX(line1[0].GetX());
               endface1[4].SetY(line1[0].GetY());
                  
               endface2[0].SetX(line1[1].GetX());
               endface2[0].SetY(line1[1].GetY());
               endface2[1].SetX(line2[1].GetX());
               endface2[1].SetY(line2[1].GetY());
               endface2[2].SetX(line3[1].GetX());
               endface2[2].SetY(line3[1].GetY());
               endface2[3].SetX(line4[1].GetX());
               endface2[3].SetY(line4[1].GetY());
               endface2[4].SetX(line1[1].GetX());
               endface2[4].SetY(line1[1].GetY());
                        
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);    
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("x");
            yaxis->SetRange(firstY,lastY);
            zaxis->SetRange(firstZ,lastZ);
            hp->SetFillColor(38);
            hp->SetTitle(Form("ProjectionX of biny=%d binz=%d", biny, binz));
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }
         break;
 
      case 2:
         // "y"
         {
            Int_t firstX = xaxis->GetFirst();
            Int_t lastX  = xaxis->GetLast();
            Int_t binx = firstX + Int_t((lastX-firstX)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);         
            Int_t firstZ = zaxis->GetFirst();
            Int_t lastZ  = zaxis->GetLast();
            Int_t binz = firstZ + Int_t((lastZ-firstZ)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);                   
            if(line1[0].GetX()) gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1 && line1[0].GetX()) {
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            xx[0]=xaxis->GetBinCenter(binx);
            xx[2] = zaxis->GetBinCenter(binz);         
            xx[1] = yaxis->GetXmin(); 
            view->WCtoNDC(xx,u);         
            line1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
            xx[1] = yaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            line1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
            gVirtualX->DrawPolyLine(2,line1);          
            if (nbins>1) {                
               xx[1] = yaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);         
               xx[0] = xaxis->GetBinCenter(binx); 
               view->WCtoNDC(xx,u);         
               line2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[1] = yaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               
               xx[1] = yaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);         
               xx[0] = xaxis->GetBinCenter(binx+nbins-1); 
               view->WCtoNDC(xx,u);         
               line3[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[1] = yaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line3[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
                 
               xx[1] = yaxis->GetXmin();
               xx[2] = zaxis->GetBinCenter(binz);         
               xx[0] = xaxis->GetBinCenter(binx+nbins-1); 
               view->WCtoNDC(xx,u);         
               line4[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[1] = yaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line4[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[1].SetY(pymin + Int_t((u[1]-uymin)*cy));   
               
               endface1[0].SetX(line1[0].GetX());
               endface1[0].SetY(line1[0].GetY());
               endface1[1].SetX(line2[0].GetX());
               endface1[1].SetY(line2[0].GetY());
               endface1[2].SetX(line3[0].GetX());
               endface1[2].SetY(line3[0].GetY());
               endface1[3].SetX(line4[0].GetX());
               endface1[3].SetY(line4[0].GetY());
               endface1[4].SetX(line1[0].GetX());
               endface1[4].SetY(line1[0].GetY());
                  
               endface2[0].SetX(line1[1].GetX());
               endface2[0].SetY(line1[1].GetY());
               endface2[1].SetX(line2[1].GetX());
               endface2[1].SetY(line2[1].GetY());
               endface2[2].SetX(line3[1].GetX());
               endface2[2].SetY(line3[1].GetY());
               endface2[3].SetX(line4[1].GetX());
               endface2[3].SetY(line4[1].GetY());
               endface2[4].SetX(line1[1].GetX());
               endface2[4].SetY(line1[1].GetY());
                       
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);    
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("y");
            xaxis->SetRange(firstX,lastX);
            zaxis->SetRange(firstZ,lastZ);
            hp->SetFillColor(38);
            hp->SetTitle(Form("ProjectionY of binx=%d binz=%d", binx, binz));
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }   
         break;
 
      case 3:
         // "z"
         {
            Int_t firstX = xaxis->GetFirst();
            Int_t lastX  = xaxis->GetLast();
            Int_t binx = firstX + Int_t((lastX-firstX)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);         
            Int_t firstY = yaxis->GetFirst();
            Int_t lastY  = yaxis->GetLast();
            Int_t biny = firstY + Int_t((lastY-firstY)*(py-pymin)/(pymax-pymin));
            yaxis->SetRange(biny,biny+nbins-1);                   
            if(line1[0].GetX()) gVirtualX->DrawPolyLine(2,line1);
            if (nbins>1 && line1[0].GetX()) {
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);
            }
            xx[0] = xaxis->GetBinCenter(binx);
            xx[1] = yaxis->GetBinCenter(biny);         
            xx[2] = zaxis->GetXmin(); 
            view->WCtoNDC(xx,u);         
            line1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
            xx[2] = zaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            line1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            line1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
            gVirtualX->DrawPolyLine(2,line1);          
            if (nbins>1) {                
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);         
               xx[0] = xaxis->GetBinCenter(binx); 
               view->WCtoNDC(xx,u);         
               line2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[2] = zaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);         
               xx[0] = xaxis->GetBinCenter(binx+nbins-1); 
               view->WCtoNDC(xx,u);         
               line3[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[2] = zaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line3[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line3[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
                 
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetBinCenter(biny);         
               xx[0] = xaxis->GetBinCenter(binx+nbins-1); 
               view->WCtoNDC(xx,u);         
               line4[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[0].SetY(pymin + Int_t((u[1]-uymin)*cy));                   
               xx[2] = zaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               line4[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               line4[1].SetY(pymin + Int_t((u[1]-uymin)*cy));   
               
               endface1[0].SetX(line1[0].GetX());
               endface1[0].SetY(line1[0].GetY());
               endface1[1].SetX(line2[0].GetX());
               endface1[1].SetY(line2[0].GetY());
               endface1[2].SetX(line3[0].GetX());
               endface1[2].SetY(line3[0].GetY());
               endface1[3].SetX(line4[0].GetX());
               endface1[3].SetY(line4[0].GetY());
               endface1[4].SetX(line1[0].GetX());
               endface1[4].SetY(line1[0].GetY());
                  
               endface2[0].SetX(line1[1].GetX());
               endface2[0].SetY(line1[1].GetY());
               endface2[1].SetX(line2[1].GetX());
               endface2[1].SetY(line2[1].GetY());
               endface2[2].SetX(line3[1].GetX());
               endface2[2].SetY(line3[1].GetY());
               endface2[3].SetX(line4[1].GetX());
               endface2[3].SetY(line4[1].GetY());
               endface2[4].SetX(line1[1].GetX());
               endface2[4].SetY(line1[1].GetY());
                       
               gVirtualX->DrawPolyLine(2,line2);
               gVirtualX->DrawPolyLine(2,line3);
               gVirtualX->DrawPolyLine(2,line4);
               gVirtualX->DrawPolyLine(5,endface1);
               gVirtualX->DrawPolyLine(5,endface2);                              
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("z");
            xaxis->SetRange(firstX,lastX);
            yaxis->SetRange(firstY,lastY);
            hp->SetFillColor(38);
            hp->SetTitle(Form("ProjectionZ of binx=%d biny=%d", binx, biny));
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }             
         break;
 
      case 4:
         // "xy"
         {
            Int_t first = zaxis->GetFirst();
            Int_t last  = zaxis->GetLast();
            Int_t binz  = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);         
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect2[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();
            xx[1] = yaxis->GetXmax();          
            xx[2] = zaxis->GetBinCenter(binz);
            value1=xx[2]; // for screen display
            view->WCtoNDC(xx,u);         
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());          
            xx[0] = xaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));          
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);          
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
            xx[0] = xaxis->GetXmin();       
            view->WCtoNDC(xx,u);         
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[1] = yaxis->GetXmax();          
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               value2=xx[2];
               view->WCtoNDC(xx,u);          
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());          
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));          
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);          
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
               xx[0] = xaxis->GetXmin();       
               view->WCtoNDC(xx,u);         
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }  
                       
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("xy");
            zaxis->SetRange(first,last);
            hp->SetFillColor(38);
            if(nbins==1)hp->SetTitle(Form("ProjectionXY of binz=%d (%.1f)", binz,value1));
            else        hp->SetTitle(Form("ProjectionXY, binz range=%d-%d (%.1f-%.1f)", binz,binz+nbins-1,value1,value2));
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }
         break;
 
      case 5:
         // "yx"
         {
            Int_t first = zaxis->GetFirst();
            Int_t last  = zaxis->GetLast();
            Int_t binz = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            zaxis->SetRange(binz,binz+nbins-1);         
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect2[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();
            xx[1] = yaxis->GetXmax();          
            xx[2] = zaxis->GetBinCenter(binz);
            value1=xx[2]; // for screen display
            view->WCtoNDC(xx,u);          
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());          
            xx[0] = xaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));          
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);          
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
            xx[0] = xaxis->GetXmin();       
            view->WCtoNDC(xx,u);         
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();
               xx[1] = yaxis->GetXmax();          
               xx[2] = zaxis->GetBinCenter(binz+nbins-1);
               value2=xx[2];
               view->WCtoNDC(xx,u);          
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());          
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));          
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);          
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
               xx[0] = xaxis->GetXmin();       
               view->WCtoNDC(xx,u);         
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }                        
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("yx");
            zaxis->SetRange(first,last);
            hp->SetFillColor(38);
            if(nbins==1)hp->SetTitle(Form("ProjectionYX of binz=%d (%.1f)", binz,value1));
            else        hp->SetTitle(Form("ProjectionXY, binz range=%d-%d (%.1f-%.1f)", binz,binz+nbins-1,value1,value2));
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }
         break;
 
      case 6:
         // "xz"        
         {
            Int_t first = yaxis->GetFirst();
            Int_t last  = yaxis->GetLast();
            Int_t biny = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            yaxis->SetRange(biny,biny+nbins-1);         
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();          
            xx[2] = zaxis->GetXmax();
            xx[1] = yaxis->GetBinCenter(biny);
            value1=xx[1];                
            view->WCtoNDC(xx,u);                 
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());          
            xx[0] = xaxis->GetXmax();           
            view->WCtoNDC(xx,u);           
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));              
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);        
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));           
            xx[0] = xaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();              
               xx[2] = zaxis->GetXmax();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               value2=xx[1];
               view->WCtoNDC(xx,u);                 
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());          
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);           
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[2] = zaxis->GetXmin();
               view->WCtoNDC(xx,u);        
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));
               xx[0] = xaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }                    
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("xz");
            yaxis->SetRange(first,last);
            hp->SetFillColor(38);         
            if(nbins==1)hp->SetTitle(Form("ProjectionXZ of biny=%d (%.1f)", biny,value1));
            else        hp->SetTitle(Form("ProjectionXZ, biny range=%d-%d (%.1f-%.1f)", biny,biny+nbins-1,value1,value2));          
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }        
         break;
 
      case 7:
         // "zx"       
         {
            Int_t first = yaxis->GetFirst();
            Int_t last  = yaxis->GetLast();
            Int_t biny = first + Int_t((last-first)*(py-pymin)/(pymax-pymin));
            yaxis->SetRange(biny,biny+nbins-1);         
            if(rect1[0].GetX())            gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);
            xx[0] = xaxis->GetXmin();          
            xx[2] = zaxis->GetXmax();
            xx[1] = yaxis->GetBinCenter(biny);
            value1=xx[1];                
            view->WCtoNDC(xx,u);                 
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());          
            xx[0] = xaxis->GetXmax();           
            view->WCtoNDC(xx,u);           
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));              
            xx[2] = zaxis->GetXmin();
            view->WCtoNDC(xx,u);        
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));           
            xx[0] = xaxis->GetXmin();
            view->WCtoNDC(xx,u);
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[0] = xaxis->GetXmin();              
               xx[2] = zaxis->GetXmax();
               xx[1] = yaxis->GetBinCenter(biny+nbins-1);
               value2=xx[1];
               view->WCtoNDC(xx,u);                 
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());          
               xx[0] = xaxis->GetXmax();           
               view->WCtoNDC(xx,u);           
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));              
               xx[2] = zaxis->GetXmin();
               view->WCtoNDC(xx,u);        
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));           
               xx[0] = xaxis->GetXmin();
               view->WCtoNDC(xx,u);
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));
               gVirtualX->DrawPolyLine(5,rect2);
            }                    
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("zx");
            yaxis->SetRange(first,last);
            hp->SetFillColor(38);         
            if(nbins==1)hp->SetTitle(Form("ProjectionZX of biny=%d (%.1f)", biny,value1));
            else        hp->SetTitle(Form("ProjectionZX, binY range=%d-%d (%.1f-%.1f)", biny,biny+nbins-1,value1,value2));          
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }     
         break;
 
      case 8:
         // "yz"        
         {
            Int_t first = xaxis->GetFirst();
            Int_t last  = xaxis->GetLast();
            Int_t binx = first + Int_t((last-first)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);
            if(rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);         
            xx[2] = zaxis->GetXmin();
            xx[1] = yaxis->GetXmax();         
            xx[0] = xaxis->GetBinCenter(binx);
            value1=xx[0];          
            view->WCtoNDC(xx,u);         
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());          
            xx[2] = zaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);        
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
            xx[2] = zaxis->GetXmin();       
            view->WCtoNDC(xx,u);       
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));     
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {        
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetXmax();         
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               value2=xx[0];          
               view->WCtoNDC(xx,u);         
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());          
               xx[2] = zaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);        
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
               xx[2] = zaxis->GetXmin();       
               view->WCtoNDC(xx,u);       
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));     
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("yz");
            xaxis->SetRange(first,last);
            hp->SetFillColor(38);
            if(nbins==1)hp->SetTitle(Form("ProjectionYZ of binx=%d (%.1f)", binx,value1));
            else        hp->SetTitle(Form("ProjectionYZ, binx range=%d-%d (%.1f-%.1f)", binx,binx+nbins-1,value1,value2)); 
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }
         break;
 
      case 9:
         // "zy"        
         {
            Int_t first = xaxis->GetFirst();
            Int_t last  = xaxis->GetLast();
            Int_t binx = first + Int_t((last-first)*(px-pxmin)/(pxmax-pxmin));
            xaxis->SetRange(binx,binx+nbins-1);
            if(rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect1);
            if(nbins>1 && rect1[0].GetX()) gVirtualX->DrawPolyLine(5,rect2);         
            xx[2] = zaxis->GetXmin();
            xx[1] = yaxis->GetXmax();         
            xx[0] = xaxis->GetBinCenter(binx);
            value1=xx[0];          
            view->WCtoNDC(xx,u);         
            rect1[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
            rect1[4].SetX(rect1[0].GetX());
            rect1[4].SetY(rect1[0].GetY());          
            xx[2] = zaxis->GetXmax();           
            view->WCtoNDC(xx,u);         
            rect1[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
            xx[1] = yaxis->GetXmin();
            view->WCtoNDC(xx,u);        
            rect1[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
            xx[2] = zaxis->GetXmin();       
            view->WCtoNDC(xx,u);       
            rect1[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
            rect1[3].SetY(pymin + Int_t((u[1]-uymin)*cy));     
            gVirtualX->DrawPolyLine(5,rect1);
            if (nbins>1) {
               xx[2] = zaxis->GetXmin();
               xx[1] = yaxis->GetXmax();         
               xx[0] = xaxis->GetBinCenter(binx+nbins-1);
               value2=xx[0];          
               view->WCtoNDC(xx,u);         
               rect2[0].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[0].SetY(pymin + Int_t((u[1]-uymin)*cy));
               rect2[4].SetX(rect2[0].GetX());
               rect2[4].SetY(rect2[0].GetY());          
               xx[2] = zaxis->GetXmax();           
               view->WCtoNDC(xx,u);         
               rect2[1].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[1].SetY(pymin + Int_t((u[1]-uymin)*cy));        
               xx[1] = yaxis->GetXmin();
               view->WCtoNDC(xx,u);        
               rect2[2].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[2].SetY(pymin + Int_t((u[1]-uymin)*cy));          
               xx[2] = zaxis->GetXmin();       
               view->WCtoNDC(xx,u);       
               rect2[3].SetX(pxmin + Int_t((u[0]-uxmin)*cx));
               rect2[3].SetY(pymin + Int_t((u[1]-uymin)*cy));     
               gVirtualX->DrawPolyLine(5,rect2);
            }
            c->Clear();
            c->cd();
            TH2 *hp = (TH2*)h3->Project3D("zy");
            xaxis->SetRange(first,last);
            hp->SetFillColor(38);
           
            if(nbins==1)hp->SetTitle(Form("ProjectionZY of binx=%d (%.1f)", binx,value1));
            else        hp->SetTitle(Form("ProjectionZY, binx range=%d-%d (%.1f-%.1f)", binx,binx+nbins-1,value1,value2)); 
            hp->SetXTitle(fH->GetYaxis()->GetTitle());
            hp->SetZTitle("Number of Entries");
            hp->Draw(fShowOption.Data());
         }
         break;
      
   }
   c->Update();
   padsav->cd();
}
