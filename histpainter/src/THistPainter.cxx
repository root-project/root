// @(#)root/histpainter:$Name:  $:$Id: THistPainter.cxx,v 1.28 2001/02/01 17:43:00 brun Exp $
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
#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "THistPainter.h"
#include "TH2.h"
#include "TF2.h"
#include "TF3.h"
#include "TVirtualPad.h"
#include "TPaveStats.h"
#include "TFrame.h"
#include "TText.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TPoints.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TGaxis.h"
#include "TColor.h"
#include "TLego.h"
#include "TView.h"
#include "TMath.h"
#include "TRandom.h"
#include "TObjArray.h"
#include "Hoption.h"
#include "Hparam.h"


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

static TH1 *hlist[10];
const Double_t kHMAX = 1.05;
const Int_t kMAXCONTOUR  = 104;
const Int_t kCannotRotate = BIT(11);

ClassImp(THistPainter)

//______________________________________________________________________________
THistPainter::THistPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Histogram default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
   fH = 0;
   fXaxis = 0;
   fYaxis = 0;
   fZaxis = 0;
   fFunctions = 0;
   fNIDS  = 0;
   fXbuf  = new Double_t[kNMAX];
   fYbuf  = new Double_t[kNMAX];
}

//______________________________________________________________________________
THistPainter::~THistPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Histogram default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

   delete [] fXbuf;
   delete [] fYbuf;
}


//______________________________________________________________________________
Int_t THistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a line*-*-*-*-*-*
//*-*                  ===========================================
//*-*  Compute the closest distance of approach from point px,py to elements
//*-*  of an histogram.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  Algorithm:
//*-*  Currently, this simple model computes the distance from the mouse
//*-*  to the histogram contour only.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t big = 9999;
   const Int_t kMaxDiff = 7;

   Double_t x = gPad->AbsPixeltoX(px);

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());

//*-*- return if point is not in the histogram area

//*-*- If a 3-D view exists, check distance to axis
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
      if ( px > puxmin && px < puxmax && py > puymax && py < puymin) return 0;
      return big;
   }
//*-*- check if point is close to an axis
   Int_t xyaxis = puxmin - Int_t((puxmax-puxmin)*fYaxis->GetLabelOffset());
   Int_t dyaxis = Int_t(2*(puymin-puymax)*fYaxis->GetLabelSize());
   if (px >= xyaxis-dyaxis && px <= xyaxis && py >puymax && py < puymin) {
      if (!strstr(gPad->GetPadPointer()->GetDrawOption(),"same")) {
         gPad->SetSelected(fYaxis);
         return 0;
      }
   }
   Int_t yxaxis = puymin + Int_t((puymin-puymax)*fXaxis->GetLabelOffset());
   if (yxaxis < puymin) yxaxis = puymin;
   Int_t dxaxis = Int_t((puymin-puymax)*fXaxis->GetLabelSize());
   if (py <= yxaxis+dxaxis && py >= yxaxis && px <puxmax && px > puxmin) {
      if (!strstr(gPad->GetPadPointer()->GetDrawOption(),"same")) {
         gPad->SetSelected(fXaxis);
         return 0;
       }
    }

//*-* check if point is on the color palette
   if (strcmp(fH->GetDrawOption(),"colz") == 0 || strcmp(fH->GetDrawOption(),"COLZ") == 0) {
      if (py <= puymin && py > puymax) {
         Double_t xup  = gPad->GetUxmax();
         Double_t x2   = gPad->GetX2();
         Double_t xr   = 0.05*(x2 - gPad->GetX1());
         Double_t xmin = xup +0.1*xr;
         Double_t xmax = xmin + xr;
         if (xmax > x2) xmax = x2-0.01*xr;
         Int_t xzaxis = gPad->XtoAbsPixel(xmax);
         if (TMath::Abs(px-xzaxis) < kMaxDiff) {
            gPad->SetSelected(fZaxis);
            return 0;
         }   
      }   
   }    
//*-*- if object is 2-D or 3-D return this object
   if (fH->GetDimension() == 2) {
      Int_t delta2 = 5; //Give a margin of delta2 pixels to be in the 2-d area
      if ( px > puxmin + delta2
        && px < puxmax - delta2
        && py > puymax + delta2
        && py < puymin - delta2) return 0;
   }

//*-*- point is inside histogram area. Find channel number
   Int_t bin      = fXaxis->FindFixBin(gPad->PadtoX(x));
   Double_t binval = fH->GetBinContent(bin);
   Int_t pybin    = gPad->YtoAbsPixel(gPad->YtoPad(binval));
   if (TMath::Abs(py - pybin) <= kMaxDiff) return TMath::Abs(py - pybin);

//*-*- Loop on the list of associated functions and user objects
   TObject *f;
   TIter   next(fFunctions);
   while ((f = (TObject*) next())) {
      Int_t dist = f->DistancetoPrimitive(px,py);
      if (dist < kMaxDiff) {gPad->SetSelected(f); return dist;}
   }
   return big;
}

//______________________________________________________________________________
void THistPainter::DrawPanel()
{
//*-*-*-*-*-*-*Display a panel with all histogram drawing options*-*-*-*-*-*
//*-*          ==================================================
//*-*
//*-*   See class TDrawPanelHist for example

   gCurrentHist = fH;
   if (gPad) {
      //TObject *pr = gPad->GetSelected();
      TObject *pr = 0;
      if (!pr) pr = fH;
      //TVirtualPad *pad = gPad->GetSelectedPad();
      TVirtualPad *pad = 0;
      if (!pad) pad = gPad;
      gROOT->SetSelectedPrimitive(pr);
      gROOT->SetSelectedPad(pad);
   } else {
      Error("DrawPanel", "need to draw histogram first");
      return;
   }

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (!lc->FindObject("R__drawpanelhist")) {
      gROOT->ProcessLine("TDrawPanelHist *R__drawpanelhist = "
                         "new TDrawPanelHist(\"R__drawpanelhist\",\"Hist Draw Panel\","
                         "330,450);");
      return;
   }
   gROOT->ProcessLine("R__drawpanelhist->SetDefaults(); R__drawpanelhist->Show();");
}

//______________________________________________________________________________
void THistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function is called when a histogram is clicked with the locator
//*-*
//*-*  If Left button clicked on the bin top value, then the content of this bin
//*-*  is modified according to the new position of the mouse when it is released.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   static Int_t bin, px1, py1, px2, py2, pyold;
   Double_t xlow, xup, ylow, binval, x, baroffset, barwidth, binwidth;

   if (!gPad->IsEditable()) return;

//*-*- come here if we have a lego/surface in the pad
   TView *view = gPad->GetView();
   if (view && view->TestBit(kCannotRotate) == 0) {
      view->ExecuteRotateView(event, px, py);
      return;
   }

   switch (event) {

   case kButton1Down:

      gVirtualX->SetLineColor(-1);
      fH->TAttLine::Modify();

      // No break !!!

   case kMouseMotion:

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
         gVirtualX->DrawBox(px1, py1, px2, py2,TVirtualX::kHollow);  //*-* Draw the old box
         py2 += py - pyold;
         gVirtualX->DrawBox(px1, py1, px2, py2,TVirtualX::kHollow);  //*-* Draw the new box
         pyold = py;
      }

      break;

   case kButton1Up:

      if (gROOT->GetEditHistograms()) {
         binval = gPad->PadtoY(gPad->AbsPixeltoY(py2));
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
//*-*-*-*-*-*-*Display a panel with all histogram fit options*-*-*-*-*-*
//*-*          ==============================================
//*-*
//*-*   See class TFitPanel for example

   gCurrentHist = fH;
   if (gPad) {
      gROOT->SetSelectedPrimitive(fH);
      gROOT->SetSelectedPad(gPad);
   } else {
      Error("FitPanel", "need to draw histogram first");
      return;
   }

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   if (!lc->FindObject("R__fitpanel")) {
      gROOT->ProcessLine("TFitPanel *R__fitpanel = "
                         "new TFitPanel(\"R__fitpanel\",\"Fit Panel\",300,400);");
      return;
   }
   gROOT->ProcessLine("R__fitpanel->SetDefaults(); R__fitpanel->Show();");
}

//______________________________________________________________________________
char *THistPainter::GetObjectInfo(Int_t px, Int_t py) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the histogram info (bin number, contents, integral up to bin
//   corresponding to cursor position px,py
//
   if (!gPad) return (char*)"";
   static char info[64];
   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y = gPad->PadtoY(gPad->AbsPixeltoY(py));
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
   Int_t binx,biny,binmin;
   binx   = fXaxis->FindFixBin(x);
   binmin = fXaxis->GetFirst();
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
Int_t THistPainter::MakeChopt(Option_t *choptin)
{
//*-*-*-*-*-*-*-*-*Decode string chopt and fill Hoption structure*-*-*-*-*-*-*
//*-*              ==============================================

   char *l;
   static char chopt[33];
   Int_t nch = strlen(choptin);
   for (Int_t i=0;i<nch;i++) chopt[i] = toupper(choptin[i]);
   chopt[nch] = 0;

   Hoption.Axis = Hoption.Bar    = Hoption.Curve   = Hoption.Error = 0;
   Hoption.Hist = Hoption.Line   = Hoption.Mark    = Hoption.Fill  = 0;
   Hoption.Same = Hoption.Func   = Hoption.Plus    = Hoption.Scat  = 0;
   Hoption.Star = Hoption.Arrow  = Hoption.Box     = Hoption.Text  = 0;
   Hoption.Char = Hoption.Color  = Hoption.Contour = Hoption.Logx  = 0;
   Hoption.Logy = Hoption.Logz   = Hoption.Lego    = Hoption.Surf  = 0;

//*-* special 2-D options
   Hoption.List        = 0;
   Hoption.Zscale      = 0;
   Hoption.FrontBox    = 1;
   Hoption.BackBox     = 1;
   Hoption.System      = kCARTESIAN;

   if (fH->GetDimension() > 1) Hoption.Scat = 1;
   if (!nch) Hoption.Hist = 1;
   if (fH->GetSumw2N()) Hoption.Error = 1;
   if (fFunctions->First()) Hoption.Func = 2;
   if (fH->GetSumw2N()) Hoption.Error = 2;

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
   }
   l = strstr(chopt,"+-");   if (l) { Hoption.Plus = 2; strncpy(l,"  ",2); }
   l = strstr(chopt,"-+");   if (l) { Hoption.Plus = 2; strncpy(l,"  ",2); }

   l = strstr(chopt,"ARR" ); if (l) { Hoption.Arrow  = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"BOX" ); if (l) { Hoption.Box    = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"COLZ"); if (l) { Hoption.Color  = 2; strncpy(l,"   ", 4); Hoption.Scat = 0; Hoption.Zscale = 1;}
   l = strstr(chopt,"COL" ); if (l) { Hoption.Color  = 1; strncpy(l,"   ", 3); Hoption.Scat = 0; }
   l = strstr(chopt,"CHAR"); if (l) { Hoption.Char   = 1; strncpy(l,"    ",4); Hoption.Scat = 0; }
   l = strstr(chopt,"FUNC"); if (l) { Hoption.Func   = 2; strncpy(l,"    ",4); Hoption.Hist = 0; }
   l = strstr(chopt,"HIST"); if (l) { Hoption.Hist   = 2; strncpy(l,"    ",4); Hoption.Func = 0; Hoption.Error = 0;}
   l = strstr(chopt,"AXIS"); if (l) { Hoption.Axis   = 1; strncpy(l,"    ",4); }
   l = strstr(chopt,"SCAT"); if (l) { Hoption.Scat   = 1; strncpy(l,"    ",4); }
   l = strstr(chopt,"TEXT"); if (l) { Hoption.Text   = 1; strncpy(l,"    ",4); Hoption.Scat = 0; }
   l = strstr(chopt,"POL");  if (l) { Hoption.System = kPOLAR;       strncpy(l,"   ",3); }
   l = strstr(chopt,"CYL");  if (l) { Hoption.System = kCYLINDRICAL; strncpy(l,"   ",3); }
   l = strstr(chopt,"SPH");  if (l) { Hoption.System = kSPHERICAL;   strncpy(l,"   ",3); }
   l = strstr(chopt,"PSR");  if (l) { Hoption.System = kRAPIDITY;    strncpy(l,"   ",3); }

   if (strstr(chopt,"A")) Hoption.Axis = -1;
   if (strstr(chopt,"B")) Hoption.Bar  = 1;
   if (strstr(chopt,"C")) { Hoption.Curve =1; Hoption.Hist = -1;}
   if (strstr(chopt,"E")) Hoption.Error =1;
   if (strstr(chopt,"F")) Hoption.Fill =1;
   if (strstr(chopt,"L")) { Hoption.Line =1; Hoption.Hist = -1;}
   if (strstr(chopt,"P")) { Hoption.Mark =1; Hoption.Hist = 0;}
   if (strstr(chopt,"Z")) Hoption.Zscale =1;
   if (strstr(chopt,"*")) Hoption.Star =1;
   if (strstr(chopt,"+")) Hoption.Plus =1;
   if (strstr(chopt,"-")) Hoption.Plus =-1;
   if (strstr(chopt,"H")) Hoption.Hist =2;
   if (strstr(chopt,"E0")) Hoption.Error =10;
   if (strstr(chopt,"E1")) Hoption.Error =11;
   if (strstr(chopt,"E2")) Hoption.Error =12;
   if (strstr(chopt,"E3")) Hoption.Error =13;
   if (strstr(chopt,"E4")) Hoption.Error =14;

   if (Hoption.Surf == 14 && Hoption.System != kCARTESIAN) {
      Hoption.System = kCARTESIAN;
      Warning("MakeChopt","option SURF4 only supported in Cartesian mode");
   }
//
//   if (fSumw2.fN && !Hoption.Error) Hoption.Error = 2;

//*-*   Copy options from current style
   Hoption.Logx = gPad->GetLogx();
   Hoption.Logy = gPad->GetLogy();
   Hoption.Logz = gPad->GetLogz();

//*-*-   Check options incompatibilities
   if (Hoption.Bar) Hoption.Hist = -1;
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
void THistPainter::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Control routine to paint any kind of histograms*-*-*-*-*-*-*
//*-*              ===============================================
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
// A pad is updated after one of these three actions:
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
//    "HIST"   : Draw only histo contour. (if histo has errors, errors are not drawn)
//    "SAME"   : Superimpose on previous picture in the same pad
//    "CYL"    : Use Cylindrical coordinates
//    "POL"    : Use Polar coordinates
//    "SPH"    : Use Spherical coordinates
//    "PSR"    : Use PseudoRapidity/Phi coordinates
//    "LEGO"   : Draw a lego plot with hidden line removal
//    "LEGO1"  : Draw a lego plot with hidden surface removal
//    "LEGO2"  : Draw a lego plot using colors to show the cell contents
//    "SURF"   : Draw a surface plot with hidden line removal
//    "SURF1"  : Draw a surface plot with hidden surface removal
//    "SURF2"  : Draw a surface plot using colors to show the cell contents
//    "SURF3"  : same as SURF with in addition a contour view drawn on the top
//    "SURF4"  : Draw a surface using Gouraud shading
//
//  The following options are supported for 1-D types:
//    "AH"     : Draw histogram, but not the axis labels and tick marks
//    "B"      : Bar chart option
//    "C"      : Draw a smooth Curve througth the histogram bins
//    "E"      : Draw error bars
//    "E0"     : Draw error bars including bins with o contents
//    "E1"     : Draw error bars with perpendicular lines at the edges
//    "E2"     : Draw error bars with rectangles
//    "E3"     : Draw a fill area througth the end points of the vertical error bars
//    "E4"     : Draw a smoothed filled area through the end points of the error bars
//    "L"      : Draw a line througth the bin contents
//    "P"      : Draw current marker at each bin
//    "*H"     : Draw histogram with a * at each bin
//
//
//  The following options are supported for 2-D types:
//    "ARR"    : arrow mode. Shows gradient between adjacent cells
//    "BOX"    : a box is drawn for each cell with surface proportional to contents
//    "COL"    : a box is drawn for each cell with a color scale varying with contents
//    "COLZ"   : same as "COL". In addition the color palette is also drawn
//    "CONT"   : Draw a contour plot (same as CONT0)
//    "CONT0"  : Draw a contour plot using surface colors to distinguish contours
//    "CONT1"  : Draw a contour plot using line styles to distinguish contours
//    "CONT2"  : Draw a contour plot using the same line style for all contours
//    "CONT3"  : Draw a contour plot using fill area colors
//    "CONT4"  : Draw a contour plot using surface colors (SURF option at theta = 0)
//    "LIST"   : Generate a list of TGraph objects for each contour
//    "FB"     : With LEGO or SURFACE, suppress the Front-Box
//    "BB"     : With LEGO or SURFACE, suppress the Back-Box
//    "SCAT"   : Draw a scatter-plot (default)
//
// Most options can be concatenated without spaces or commas, for example:
//        h->Draw("E1SAME");
//
// The options are not case sensitive:
//	h->Draw("e1same");
//
// The options "BOX", "COL" or "COLZ", use the color palette 
// defined in the current style (see TStyle::SetPalette)
//
// The options "CONT" or "SURF" or "LEGO" have by default 20 equidistant contour
// levels, you can change the number of levels with TH1::SetContour.
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
// When trailing digits is left out, they are assumed to be 0.
// For example: gStyle->SetOptStat(11);
// displays only the name of histogram and the number of entries.
// 
//When the option "same", the statistic box is not redrawn, and hence 
// the statistics from the previously drawn hostgram will still show. 
// With the option "sames", you can rename a previous "stats" box 
// and/or change its position with these lines:
//
//  Root > TPaveStats *st = (TPaveStats*)gPad->FindObject("stats")
//  Root > st->SetName(newname)
//  Root > st->SetX1NDC(newx1); //new x start position
//  Root > st->SetX2NDC(newx2); //new x end position
//  Root > newhist->Draw("sames")
//
// Fit Statistics
// ==============
// You can change the statistics box to display the fit paramters with
// the TH1::SetOptFit(mode) method. This mode has four digits.
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
//    The color palette in TStyle can be modified via TStyle::SetPalette.
//Begin_Html
/*
<img src="gif/h2_c2h.gif">
*/
//End_Html
//
//
//  The TEXT Option
//  =============== 
//    For each cell (i,j) the cell content is printed.
//    The text attributes are:
//      - text font = current TStyle font
//      - text size = 0.02*padheight*markersize
//      - text color= marker color
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
// 
//  The default number of contour levels is 20 equidistant levels and can
//  be changed with TH1::SetContour.
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
//      See TStyle::SetPalette to change the color palette.
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
//   ==================
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
//      See TStyle::SetPalette to change the color palette.
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
// For the options "BOX", "COL", "CONT", "SURF", "LEGO"
// you can display the color palette with an axis indicating the value
// of the corresponding color.
//  
//  Setting the color palette
//  =========================
// You can set the color palette with TStyle::SetPalette, eg
//
//      gStyle->SetPalette(ncolors,colors);
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
//
//   Drawing options for 3-D histograms
//   ==================================
//   By default a 3-d scatter plot is drawn
//   If option "BOX" is specified, a 3-D box with a volume proportional
//   to the cell content is drawn.
//
//
//--------------------------------------------------------------------
   TH1 *oldhist = gCurrentHist;
   gCurrentHist = fH;
   TH1 *hsave   = fH;
   Double_t minsav = fH->GetMinimumStored();

   if (!MakeChopt(option)) return; //check options and fill Hoption structure

   if (fH->GetDimension() > 2) {
      PaintH3(option);
      fH->SetMinimum(minsav);
      if (Hoption.Func) {
         Hoption_t hoptsave = Hoption;
         Hparam_t  hparsave = Hparam;
         PaintFunction();
         SetHistogram(hsave);
         Hoption = hoptsave;
         Hparam  = hparsave;
      }
      gCurrentHist = oldhist;
      return;
   }
   TView *view = gPad->GetView();
   if (view) {
      if (!Hoption.Lego && !Hoption.Surf ) {
         delete view;
         gPad->SetView(0);
      }
   }
   if (fH->GetDimension() > 1 || Hoption.Lego || Hoption.Surf ) {
      PaintTable();
      fH->SetMinimum(minsav);
      if (Hoption.Func) {
         Hoption_t hoptsave = Hoption;
         Hparam_t  hparsave = Hparam;
         PaintFunction();
         SetHistogram(hsave);
         Hoption = hoptsave;
         Hparam  = hparsave;
      }
      gCurrentHist = oldhist;
      return;
   }

   if (!PaintInit()) return;  //fill Hparam structure with histo parameters

//*-*-      Picture surround (if new page) and page number (if requested).
//*-*-      Histogram surround (if not option "Same").

   PaintFrame();
//*-*------
//*-*-      Paint histogram axis only
   if (Hoption.Axis > 0) {
      PaintAxis();
      return;
   }

//*-*------
//*-*-      do not draw histogram if error bars required
   if (!Hoption.Error) {
      if (Hoption.Hist) PaintHist();
   }
//*-*------

//*-*-     test for error bars or option E
   if (Hoption.Error || Hoption.Mark) {
      PaintErrors();
      if (Hoption.Hist == 2) PaintHist();
   }

//*-*-     test for associated function
   if (Hoption.Func) {
      Hoption_t hoptsave = Hoption;
      Hparam_t  hparsave = Hparam;
      PaintFunction();
      SetHistogram(hsave);
      Hoption = hoptsave;
      Hparam  = hparsave;
   }

   PaintAxis();     //*-* Draw the axes
   PaintTitle();    //*-* Draw histogram title
//   PaintFile();     //*-* Draw Current File name corresp to current directory
//   PaintDate();     //*-* Draw date/time
     //*-* Draw box with histogram statistics and/or fit parameters
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

}

//______________________________________________________________________________
void THistPainter::PaintArrows()
{
//*-*-*-*-*-*-*-*Control function to draw a table as an arrow plot*-*-*-*-*-*
//*-*            =================================================
//*-*
//*-*    For each cell (i,j) an arrow is drawn
//*-*    The orientation of the arrow follows the cell gradient
//Begin_Html
/*
<img src="gif/PaintArrows.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
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
   fH->SetLineStyle(linesav);
   fH->SetLineWidth(widthsav);
   fH->TAttLine::Modify();
}

//______________________________________________________________________________
void THistPainter::PaintAxis()
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw axis of an histogram*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =========================
//
//   Assume tx = gPad->GetTickx() and ty = gPad->GetTicky()
//   by default only the left Y axis and X bottom axis are drawn (tx = ty = 0)
//    tx = 1 ;  tick marks on top side are drawn (inside)
//    tx = 2;   tick marks and labels on top side are drawn
//    ty = 1;   tick marks on right side are drawn (inside)
//    ty = 2;   tick marks and labels on right side are drawn
//       Use TPad::SetTicks(tx,ty) to set these options

   if (Hoption.Same && Hoption.Axis <= 0) return;

   static char chopt[10] = "";
   Double_t gridl = 0;
   Int_t ndiv, ndivx, ndivy, nx1, nx2,ndivsave;
   Double_t umin, umax, uminsave, umaxsave;

   Double_t axmin = gPad->GetUxmin();
   Double_t axmax = gPad->GetUxmax();
   Double_t aymin = gPad->GetUymin();
   Double_t aymax = gPad->GetUymax();
   char *cw = 0;

//*-*- X axis
   ndivx = fH->GetNdivisions("X");
   ndivy = fH->GetNdivisions("Y");
   if (ndivx > 1000) {
      nx2   = ndivx/100;
      nx1   = TMath::Max(1, ndivx%100);
      ndivx = 100*nx2 + Int_t(Float_t(nx1)*gPad->GetAbsWNDC());
   }
   TGaxis axis;
   axis.SetTextAngle(0);
   axis.SetLineColor(fXaxis->GetAxisColor());
   axis.SetTextColor(fXaxis->GetTitleColor());
   axis.SetTextFont(fXaxis->GetTitleFont());
   axis.SetLabelColor(fXaxis->GetLabelColor());
   axis.SetLabelFont(fXaxis->GetLabelFont());
   axis.SetLabelSize(fXaxis->GetLabelSize());
   axis.SetLabelOffset(fXaxis->GetLabelOffset());
   axis.SetTickSize(fXaxis->GetTickLength());
   axis.SetTitle(fXaxis->GetTitle());
   axis.SetTitleOffset(fXaxis->GetTitleOffset());
   axis.SetTitleSize(fXaxis->GetTitleSize());
   axis.SetBit(TGaxis::kCenterTitle, fXaxis->TestBit(TGaxis::kCenterTitle));
   axis.SetBit(TGaxis::kRotateTitle, fXaxis->TestBit(TGaxis::kRotateTitle));
   axis.SetTimeFormat(fXaxis->GetTimeFormat());

   chopt[0] = 0;
   strcat(chopt, "SDH");
   if (ndivx < 0) strcat(chopt, "N");
   if (gPad->GetGridx()) {
      gridl = (aymax-aymin)/(gPad->GetY2() - gPad->GetY1());
      strcat(chopt, "W");
   }

   if (Hoption.Logx) {
      strcat(chopt, "G");
      ndiv =TMath::Abs(ndivx);
      umin = TMath::Power(10,axmin);
      umax = TMath::Power(10,axmax);
   } else {
      ndiv =TMath::Abs(ndivx);
      umin = axmin;
      umax = axmax;
   }

//   Display axis as time
   if (fXaxis->GetTimeDisplay()) {
      strcat(chopt,"t");
      if (strlen(fXaxis->GetTimeFormat()) == 0) {
         axis.SetTimeFormat(fXaxis->ChooseTimeFormat(Hparam.xmax-Hparam.xmin));
      }
   }
   uminsave = umin;
   umaxsave = umax;
   ndivsave = ndiv;
   axis.SetOption(chopt);
   axis.PaintAxis(axmin, aymin,
                  axmax, aymin,
                  umin, umax,  ndiv, chopt, gridl);
   if (gPad->GetTickx()) {
      strcat(chopt, "-");
      if (gPad->GetTickx() < 2) strcat(chopt, "U");
      if ((cw=strstr(chopt,"W"))) *cw='z';
      axis.SetTitle("");
      axis.PaintAxis(axmin, aymax,
                     axmax, aymax,
                     uminsave, umaxsave,  ndivsave, chopt, gridl);
   }
//*-*- Y axis
   axis.SetLineColor(fYaxis->GetAxisColor());
   axis.SetTextColor(fYaxis->GetTitleColor());
   axis.SetTextFont(fYaxis->GetTitleFont());
   axis.SetLabelColor(fYaxis->GetLabelColor());
   axis.SetLabelFont(fYaxis->GetLabelFont());
   axis.SetLabelSize(fYaxis->GetLabelSize());
   axis.SetLabelOffset(fYaxis->GetLabelOffset());
   axis.SetTickSize(fYaxis->GetTickLength());
   axis.SetTitle(fYaxis->GetTitle());
   axis.SetTitleOffset(fYaxis->GetTitleOffset());
   axis.SetTitleSize(fYaxis->GetTitleSize());
   axis.SetBit(TGaxis::kCenterTitle, fYaxis->TestBit(TGaxis::kCenterTitle));
   axis.SetBit(TGaxis::kRotateTitle, fYaxis->TestBit(TGaxis::kRotateTitle));
   axis.SetTimeFormat(fYaxis->GetTimeFormat());

      chopt[0] = 0;
   strcat(chopt, "SDH");
   if (ndivy < 0) {
      nx2   = ndivy/100;
      nx1   = TMath::Max(1, ndivy%100);
 //     ndivy = 100*nx2 + Int_t(Float_t(nx1)*gPad->GetAbsHNDC());
      strcat(chopt, "N");
   }
   if (gPad->GetGridy()) {
      gridl = (axmax-axmin)/(gPad->GetX2() - gPad->GetX1());
      strcat(chopt, "W");
   }
   if (Hoption.Logy) {
      strcat(chopt, "G");
      ndiv =TMath::Abs(ndivy);
      umin = TMath::Power(10,aymin);
      umax = TMath::Power(10,aymax);
   } else {
      ndiv =TMath::Abs(ndivy);
      umin = aymin;
      umax = aymax;
   }

//   Display axis as time
   if (fYaxis->GetTimeDisplay()) {
      strcat(chopt,"t");
      if (strlen(fYaxis->GetTimeFormat()) == 0) {
         axis.SetTimeFormat(fYaxis->ChooseTimeFormat(Hparam.ymax-Hparam.ymin));
      }
   }
   uminsave = umin;
   umaxsave = umax;
   ndivsave = ndiv;
   axis.SetOption(chopt);
   axis.PaintAxis(axmin, aymin,
                  axmin, aymax,
                  umin, umax,  ndiv, chopt, gridl);
   if (gPad->GetTicky()) {
      if (gPad->GetTicky() < 2) {
         strcat(chopt, "U");
         axis.SetTickSize(-fYaxis->GetTickLength());
      } else {
         strcat(chopt, "+L");
      }
      if ((cw=strstr(chopt,"W"))) *cw='z';
      axis.SetTitle("");
      axis.PaintAxis(axmax, aymin,
                     axmax, aymax,
                     uminsave, umaxsave,  ndivsave, chopt, gridl);
   }
}


//______________________________________________________________________________
void THistPainter::PaintBoxes()
{
//*-*-*-*-*-*-*-*Control function to draw a table as a box plot*-*-*-*-*-*
//*-*            ==============================================
//*-*
//*-*    For each cell (i,j) a box is drawn.
//*-*    The size of the box is proportional to the cell content.
//Begin_Html
/*
<img src="gif/PaintBox.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Style_t linesav   = fH->GetLineStyle();
   Width_t widthsav  = fH->GetLineWidth();
   Style_t fillsav   = fH->GetFillStyle();
   if (fH->GetFillColor() == 0)  fH->SetFillStyle(0);
   fH->SetLineStyle(1);
   fH->SetLineWidth(1);
   fH->TAttLine::Modify();
   fH->TAttFill::Modify();

   Double_t z, xk,xstep, yk, ystep, xcent, ycent, xlow, xup, ylow, yup;
   Double_t dz = Hparam.zmax - Hparam.zmin;
   Double_t dxmin = 0.51*(gPad->PixeltoX(1)-gPad->PixeltoX(0));
   Double_t dymin = 0.51*(gPad->PixeltoY(0)-gPad->PixeltoY(1));

   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      ycent = 0.5*ystep;
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         xcent = 0.5*xstep;
         z     = fH->GetBinContent(bin);
         if (Hoption.Logz) {
            if (z != 0) z = TMath::Log10(z);
            else        z = Hparam.zmin;
         }
         if (z <= Hparam.zmin) continue;
         xup  = xcent*(z - Hparam.zmin)/dz + xk + xcent;
         xlow = 2*(xk + xcent) - xup;
         if (xup-xlow < dxmin) xup = xlow+dxmin;

         if (Hoption.Logx) {
            if (xup > 0)  xup  = TMath::Log10(xup);
            else continue;
            if (xlow > 0) xlow = TMath::Log10(xlow);
            else continue;
         }
         yup  = ycent*(z - Hparam.zmin)/dz + yk + ycent;
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
         gVirtualX->SetFillColor(fH->GetFillColor()); //is redefined in TPad::PaintBox
         gPad->PaintBox(xlow, ylow, xup, yup);
      }
   }
   fH->SetFillStyle(fillsav);
   fH->SetLineStyle(linesav);
   fH->SetLineWidth(widthsav);
   fH->TAttLine::Modify();
}

//______________________________________________________________________________
void THistPainter::PaintColorLevels()
{
//*-*-*-*-*-*-*-*Control function to draw a table as a color plot*-*-*-*-*-*
//*-*            ================================================
//*-*
//*-*    For each cell (i,j) a box is drawn with a color proportional
//*-*    to the cell content.
//*-*    The color table used is defined in the current style (gStyle).
//*-*    The color palette in TStyle can be modified via TStyle::SetPalette.
//Begin_Html
/*
<img src="gif/PaintCol.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


   Double_t z, xk,xstep, yk, ystep, xlow, xup, ylow, yup;
   Double_t dz = Hparam.zmax - Hparam.zmin;
   if (dz <= 0) return;

   Style_t fillsav   = fH->GetFillStyle();
   Style_t colsav    = fH->GetFillColor();
//   SetFillStyle(1000);
   fH->TAttFill::Modify();

   Int_t ncolors  = gStyle->GetNumberOfColors();
   Double_t scale = ncolors/dz;
   Int_t color;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      yk    = fYaxis->GetBinLowEdge(j);
      ystep = fYaxis->GetBinWidth(j);
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         xk    = fXaxis->GetBinLowEdge(i);
         xstep = fXaxis->GetBinWidth(i);
         z     = fH->GetBinContent(bin);
         if (Hoption.Logz) {
            if (z != 0) z = TMath::Log10(z);
            else        z = Hparam.zmin;
         }
         if (z <= Hparam.zmin) continue;
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
         color = Int_t((z- Hparam.zmin)*scale);
         if (z >= Hparam.zmax) color = ncolors-1;
         fH->SetFillColor(gStyle->GetColorPalette(color));
         fH->TAttFill::Modify();
         gPad->PaintBox(xlow, ylow, xup, yup);
      }
   }

   if (Hoption.Zscale) PaintPalette();

   fH->SetFillStyle(fillsav);
   fH->SetFillColor(colsav);
   fH->TAttFill::Modify();

}

//______________________________________________________________________________
void THistPainter::PaintContour()
{
//*-*-*-*-*-*-*-*Control function to draw a table as a contour plot*-*-*-*-*-*
//*-*            =================================================
//*-*  Hoption.Contour may have the following values:
//*-*     1  The contour is drawn with filled colour levels. ("cont")
//*-*    11  Use colour to distinguish contours. ("cont1")
//*-*    12  Use line style to distinguish contours. ("cont2")
//*-*    13  Line style and colour are the same for all contours. ("cont3")
//*-*    14  same as 1 but uses the "SURF" algorithm ("cont4")
//*-*
//*-*  When option "List" is specified together with option "cont",
//*-*  the points used to draw the contours are saved in the TGraph format
//*-*  and are accessible in the following way:
//*-* TObjArray *contours = 
//*-*           gROOT->GetListOfSpecials()->FindObject("contours")
//*-* Int_t ncontours = contours->GetSize();
//*-* TList *list = (TList*)contours->At(i); //where i is a contour number   
//*-* list contains a list of TGraph objects. For one given contour, more than 
//*-* one disjoint polyline may be generated. The number of TGraphs per 
//*-* countour is given by list->GetSize(). 
//*-* Here we show only the case to access the first graph in the list.
//*-*    TGraph *gr1 = (TGraph*)list->First();
//*-* 
//*-*   
//Begin_Html
/*
<img src="gif/PaintContour1.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t i, j, count, ncontour, icol, n, lj, m, ix, jx, ljfill;
   Int_t itars, mode, ir[4];
   Double_t xsave, ysave, thesave,phisave,x[4], y[4], zc[4];

   if (Hoption.Contour == 14) {
      Hoption.Surf = 12;
      thesave = gPad->GetTheta();
      phisave = gPad->GetPhi();
      gPad->SetPhi(0.0001);
      gPad->SetTheta(89.99);
      PaintSurface();
      gPad->SetPhi(phisave);
      gPad->SetTheta(thesave);
      gPad->GetView()->SetBit(kCannotRotate); //tested in ExecuteEvent
//      if (Hoption.Zscale) PaintPalette();
      return;
   }

   Double_t *levels  = new Double_t[kMAXCONTOUR];
   Double_t *xarr    = new Double_t[kMAXCONTOUR];
   Double_t *yarr    = new Double_t[kMAXCONTOUR];
   Int_t  *itarr     = new Int_t[kMAXCONTOUR];
   
   Int_t npmax = 0;
   for (i=0;i<kMAXCONTOUR;i++) itarr[i] = 0;

   ncontour  = fH->GetContour();
   if (!ncontour) {
      ncontour = 20;
      fH->SetContour(ncontour);
   }
   if (ncontour > kMAXCONTOUR) {
      Warning("PaintContour", "maximum number of contours is %d, asked for %d",
              kMAXCONTOUR, ncontour);
      ncontour = kMAXCONTOUR-1;
      fH->SetContour(ncontour);
   }
   for (i=0;i<ncontour;i++) levels[i] = fH->GetContourLevel(i);
   Int_t linesav   = fH->GetLineStyle();
   Int_t colorsav  = fH->GetLineColor();
   Int_t fillsav  = fH->GetFillColor();
   if (Hoption.Contour == 13) {
      fH->TAttLine::Modify();
   }
   
   TPolyLine **polys = 0;
   TPolyLine *poly=0, *polynew=0;
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
      //if (y[0] < gPad->GetUymin() || y[0] > gPad->GetUymax()) continue;
      //if (y[2] < gPad->GetUymin() || y[2] > gPad->GetUymax()) continue;
      for (i=Hparam.xfirst; i<Hparam.xlast; i++) {
         zc[0] = fH->GetCellContent(i, j);
         zc[1] = fH->GetCellContent(i+1, j);
         zc[2] = fH->GetCellContent(i+1, j+1);
         zc[3] = fH->GetCellContent(i, j+1);
         for (k=0;k<4;k++) {
            ir[k] = TMath::BinarySearch(ncontour,levels,zc[k]);
            if (zc[k] > levels[ncontour-1]) ir[k] = ncontour-1;
         }
         if (ir[0] != ir[1] || ir[1] != ir[2] || ir[2] != ir[3] || ir[3] != ir[0]) {
            x[0] = fXaxis->GetBinCenter(i);
            x[3] = x[0];
            x[1] = fXaxis->GetBinCenter(i+1);
            x[2] = x[1];
            //if (x[0] < gPad->GetUxmin() || x[0] > gPad->GetUxmax()) continue;
            //if (x[2] < gPad->GetUxmin() || x[2] > gPad->GetUxmax()) continue;
            if (zc[0] <= zc[1]) n = 0; else n = 1;
            if (zc[2] <= zc[3]) m = 2; else m = 3;
            if (zc[n] > zc[m]) n = m;
            n++;
            lj=1;
            for (ix=1;ix<=4;ix++) {
               m = n%4 + 1;
               ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],zc[m-1],
                     ir[m-1],x[m-1],y[m-1],&xarr[lj-1],&yarr[lj-1],&itarr[lj-1]);
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
                     ir[m-1],x[m-1],y[m-1],&xarr[lj-1],&yarr[lj-1],&itarr[lj-1]);
               lj += 2*ljfill;
               n = m;
            }
//*-*- Re-order endpoints

            count = 0;
            for (ix=1; ix<=lj-5; ix +=2) {
               count = 0;
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
//               icol = (4*itarr[ix])/ncontour+1;
               theColor = Int_t((itarr[ix])*Float_t(ncolors)/Float_t(ndivz));
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

               ipoly = itarr[ix]-1;
               if (ipoly >=0 && ipoly <ncontour) {
                  poly = polys[ipoly];
                  if (np[ipoly] > poly->GetN()-2) { // extend polyline
                     polynew = new TPolyLine(np[ipoly] + 100);
                     for (k=0;k<np[ipoly];k++) {
                        polynew->SetPoint(k,poly->GetX()[k],poly->GetY()[k]);
                     }
                     delete poly;
                     polys[ipoly] = polynew;
                     poly = polynew;
                  }
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
   if (Hoption.Contour != 1) goto theEND;
   
   //The 2 points line generated above are now sorted/merged to generate
   //a list of consecutive points.
   // If the option "List" has been specified, the list of points is saved
   // in the form of TGraph objects in the ROOT list of special objects.
   xmin = gPad->GetUxmin();
   ymin = gPad->GetUymin();
   xp = new Double_t[2*npmax];
   yp = new Double_t[2*npmax];
   for (ipoly=0;ipoly<ncontour;ipoly++) {
      if (np[ipoly] == 0) continue;
      if (Hoption.List) list = (TList*)contours->At(ipoly);
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
         theColor = Int_t((ipoly+2)*Float_t(ncolors)/Float_t(ndivz));
         icol = gStyle->GetColorPalette(theColor);
         fH->SetFillColor(icol);
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
   if (Hoption.Zscale) PaintPalette();

   for (i=0;i<ncontour;i++) delete polys[i];
   delete [] polys;
   delete [] xp;
   delete [] yp;

theEND:
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
                            Double_t *xarr, Double_t *yarr, Int_t *itarr)
{
//*-*-*-*-*-*-*-*Fill the matrix XARR YARR for Contour Plot*-*-*-*-*-*-*-*
//*-*            ==========================================
//*-*

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
      elev = fH->GetContourLevel(n);
      diff = elev - elev1;
      pdif = diff/tdif;
      xlen = tlen*pdif;
      if (vert) {
         xarr[i] = x1;
         yarr[i] = y1 + xlen;
      } else {
         xarr[i] = x1 + xlen;
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
void THistPainter::PaintErrors()
{
//*-*-*-*-*-*-*-*-*-*-*Draw histogram error bars*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*-
//*-*-   Draws error bars for the current histogram. The current polymarker
//*-*-   is drawn at the centre of the errors according to CHOPT:
//*-*-
//*-*-   ' ' Coordinates are expressed in histogram coordinates
//*-*-       (of the last drawn histogram). Error bars are drawn.
//*-*-   '1' Small lines are drawn at the end of the error bars.
//*-*-   '2' Error rectangles are drawn.
//*-*-   '3' A filled area is drawn through the end points of the vertical
//*-*-       error bars.
//*-*-   '4' A smoothed filled area is drawn through the end points of the
//*-*-       vertical error bars.
//*-*-   '0' Turn off the symbols clipping.
//Begin_Html
/*
<img src="gif/PaintErrors.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   const Int_t BASEMARKER=8;
   Double_t xp, yp, ex1, ex2, ey1, ey2;
   Double_t delta;
   Double_t s2x, s2y, bxsize, bysize, symbolsize, xerror;
   Double_t xi1, xi2, xi3, xi4, yi1, yi2, yi3, yi4;
   Double_t xmin, xmax, ymin, ymax;
   Double_t logxmin = 0;
   Double_t logymin = 0;
   Int_t k, npoints, first, last, fixbin;
   Int_t if1 = 0;
   Int_t if2 = 0;
   Int_t drawmarker, errormarker;
   Int_t option0, option1, option2, option3, option4, optionE;
//*-*-_____________________________

   Double_t *xline = 0;
   Double_t *yline = 0;
   option0 = option1 = option2 = option3 = option4 = optionE = 0;
   if (Hoption.Error == 10) option0 = 1;
   if (Hoption.Error == 11) option1 = 1;
   if (Hoption.Error == 12) option2 = 1;
   if (Hoption.Error == 13) option3 = 1;
   if (Hoption.Error == 14) {option4 = 1; option3 = 1;}
   if (option2+option3 == 0) optionE = 1;
   if (Hoption.Error == 0) optionE = 0;
   if (fXaxis->GetXbins()->fN) fixbin = 0;
   else                       fixbin = 1;

   errormarker = fH->GetMarkerStyle();
   xerror      = gStyle->GetErrorX();
   symbolsize  = fH->GetMarkerSize();

//*-*-      set the graphics attributes

   fH->TAttLine::Modify();
   fH->TAttFill::Modify();
   fH->TAttMarker::Modify();

//*-*-      initiate the first and last bin

   first      = Hparam.xfirst;
   last       = Hparam.xlast;
   npoints    = last - first  +1;
   if (errormarker == 1) symbolsize = 0.01;
   Double_t factor = Hparam.factor;
//   xmin       = Hparam.xmin;
//   xmax       = Hparam.xmax;
   xmin       = gPad->GetUxmin();
   xmax       = gPad->GetUxmax();
   ymin       = gPad->GetUymin();
   ymax       = gPad->GetUymax();
   bxsize     = 0.01*symbolsize*(xmax-xmin);

//*-*-      initialize the filled area drawing

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

//*-*-      define the offset of the error bars due to the symbol size

   s2x    = gPad->PixeltoX(Int_t(0.5*symbolsize*BASEMARKER)) - gPad->PixeltoX(0);
   s2y    =-gPad->PixeltoY(Int_t(0.5*symbolsize*BASEMARKER)) + gPad->PixeltoY(0);
   bxsize = 0.5*s2x;
   bysize = 0.5*s2y;

//*-*-      initialize the first point

   if (fixbin) {
      if (Hoption.Logx) xp = TMath::Power(10,Hparam.xmin) + 0.5*Hparam.xbinsize;
      else              xp = Hparam.xmin + 0.5*Hparam.xbinsize;
   }
   else {
      delta = fH->GetBinWidth(first);
      xp    = fH->GetBinLowEdge(first) + 0.5*delta;
   }

//*      if errormarker = 0 or symbolsize = 0. no symbol is drawn

   if (Hoption.Logx) logxmin = TMath::Power(10,Hparam.xmin);
   if (Hoption.Logy) logymin = TMath::Power(10,Hparam.ymin);

//*-*----------------------- Loop over the points---------------------

   for (k=first; k<=last; k++) {

//*-*-      get the data

//*-*- xp      = X position of the current point
//*-*- yp      = Y position of the current point
//*-*- ex1   = Low X error
//*-*- ex2   = Up X error
//*-*- ey1   = Low Y error
//*-*- ey2   = Up Y error
//*-*- (xi,yi) = Error bars corrdinates

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
      if (ey1 <= 0 && yp <= 0) goto L30;
      if (Hoption.Logy && yp <= 0) goto L30;

      xi4 = xp;
      xi3 = xp;
      xi2 = xp + ex2;
      xi1 = xp - ex1;

      yi1 = yp;
      yi2 = yp;
      yi3 = yp - ey1;
      yi4 = yp + ey2;

//*-*-      take the LOG if necessary

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

//*-*-      test if error bars are not outside the limits
//*-*-      otherwise they are truncated

      xi1 = TMath::Max(xi1,xmin);
      xi2 = TMath::Min(xi2,xmax);
      yi3 = TMath::Max(yi3,ymin);
      yi4 = TMath::Min(yi4,ymax);

//*-*-      test if the marker is on the frame limits. If "Yes", the
//*-*-      marker will be not drawn and the error bars will be readjusted.

      drawmarker = kTRUE;
      if (!option0) {   // <=====Please check
         if (yi1 < ymin || yi1 > ymax) goto L30;
//         if (((yi1-s2y) < ymin && (yi1+s2y) > ymin)
//          || ((yi1-s2y) < ymax && (yi1+s2y) > ymax)
//          || ((xi3-s2x) < xmin && (xi3+s2x) > xmin)
//          || ((xi3-s2x) < xmax && (xi3+s2x) > xmax))
//         drawmarker = kFALSE;
          if (Hoption.Error != 0 && ey1 <= 0) drawmarker = kFALSE;
      }
      if (!symbolsize || !errormarker) drawmarker = kFALSE;

//*-*-      draw the error rectangles

      if (option2) gPad->PaintBox(xi1,yi3,xi2,yi4);

//*-*-      keep points for fill area drawing

      if (option3) {
         xline[if1-1] = xi3;
         xline[if2-1] = xi3;
         yline[if1-1] = yi4;
         yline[if2-1] = yi3;
         if1++;
         if2--;
      }

//*-*-      draw the error bars

      if (optionE && drawmarker) {
         if (yi3 < yi1 - s2y) gPad->PaintLine(xi3,yi3,xi4,yi1 - s2y);
         if (yi1 + s2y < yi4) gPad->PaintLine(xi3,yi1 + s2y,xi4,yi4);
         if (xi1 < xi3 - s2x) gPad->PaintLine(xi1,yi1,xi3 - s2x,yi2);
         if (xi3 + s2x < xi2) gPad->PaintLine(xi3 + s2x,yi1,xi2,yi2);
      }
      if (optionE && !drawmarker) {
         if (yi3 < yi4) gPad->PaintLine(xi3,yi3,xi4,yi4);
         if (yi1 < yi4) gPad->PaintLine(xi3,yi1,xi4,yi4);
         if (xi1 < xi3) gPad->PaintLine(xi1,yi1,xi3,yi2);
         if (xi3 < xi2) gPad->PaintLine(xi3,yi1,xi2,yi2);
      }

//*-*-      draw line at the end of the error bars

      if (option1) {
         if (yi3 < yi1-s2y) gPad->PaintLine(xi3 - bxsize,yi3,xi3 + bxsize,yi3);
         if (yi4 > yi1+s2y) gPad->PaintLine(xi3 - bxsize,yi4,xi3 + bxsize,yi4);
         if (xi1 < xi3-s2x) gPad->PaintLine(xi1,yi1 - bysize,xi1,yi1 + bysize);
         if (xi2 > xi3+s2x) gPad->PaintLine(xi2,yi1 - bysize,xi2,yi1 + bysize);
      }

//*-*-      draw the marker

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

//*-*-      draw the filled area

   if (option3) {
      TGraph graph;
      graph.SetFillStyle(fH->GetFillStyle());
      graph.SetFillColor(fH->GetFillColor());
      if (option4) graph.PaintGraph(2*npoints,xline,yline,"FC");
      else         graph.PaintGraph(2*npoints,xline,yline,"F");
      delete [] xline;
      delete [] yline;
   }
}

//______________________________________________________________________________
void THistPainter::PaintFrame()
{
//*-*-*-*-*-*-*-*-*-*Calculate range and clear pad (canvas)*-*-*-*-*-*-*-*-*-*
//*-*                ======================================
   if (Hoption.Same) return;

   RecalculateRange();

   if (Hoption.Lego || Hoption.Surf || Hoption.Contour == 14) {
      TObject *frame = gPad->FindObject("TFrame");
      if (frame) gPad->GetListOfPrimitives()->Remove(frame);
      return;
   }
   gPad->PaintPadFrame(Hparam.xmin,Hparam.ymin,Hparam.xmax,Hparam.ymax);
}

//______________________________________________________________________________
void THistPainter::PaintFunction()
{
//*-*-*-*-*-*-*-*-*-*-*Paint functions associated to an histogram*-*-*-*-*-*-*
//*-*                  ==========================================
//
//   An associated function is created by THistPainter::Fit. Note that more than
//   on fitted function can be associated with one histogram (see THistPainter::Fit).
//
//   A TF1 object can be added to the list of associated functions directly
//   by a user without calling THistPainter::Fit.
//   To add a new function to the list of associated functions, do
//     h->GetListOfFunctions()->Add(f1);
//   To retrieve a function by name from this list, do:
//     TF1 *f1 = (TF1*)h->GetListOfFunctions()->FindObject(name);
//   or
//     TF1 *f1 = h->GetFunction(name);
//
   TObject *f;
   TIter   next(fFunctions);
   while ((f = (TObject*) next())) {
      TVirtualPad *padsave = gPad;
      if (f->InheritsFrom(TF1::Class())) {
         if (f->TestBit(TF1::kNotDraw) == 0) f->Paint("lsame");
      } else  {
         f->Paint();
      }
      padsave->cd();
   }
}

//______________________________________________________________________________
void THistPainter::PaintHist()
{
//*-*-*-*-*-*-*-*-*-*Control routine to draw an histogram*-*-*-*-*-*-*-*-*-*-*
//*-*                ====================================

   static char chopth[17];
   static char choptg[5];
   static char chtemp[9];


   Int_t htype, oldhtype;
   Int_t i, j, first, last, nbins, fixbin;
   Double_t c1, yb, y1, y2, ym1, ym2, yadd, ycur, ypre;
   Double_t ync, ynext, ypc;
   yb = ynext = 0;

   strcpy(chopth, "                ");
   strcpy(choptg, "    ");
   strcpy(chtemp, "        ");

   Double_t ymin = Hparam.ymin;
   Double_t ymax = Hparam.ymax;
   Double_t baroffset = fH->GetBarOffset();
   Double_t barwidth  = fH->GetBarWidth();
   Double_t baroffsetsave = gStyle->GetBarOffset();
   Double_t barwidthsave  = gStyle->GetBarWidth();
   gStyle->SetBarOffset(baroffset);
   gStyle->SetBarWidth(barwidth);

//*-*-   Create "LIFE" structure to keep current histogram status

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

//*-*-  Loop on histogram bins

   for (j=first; j<=last;j++) {
      c1 = Hparam.factor*fH->GetBinContent(j);
      if (TMath::Abs(ymax-ymin) > 0) {
         if (Hoption.Logy) yb = TMath::Log10(TMath::Max(c1,.1*logymin));
         else              yb = c1;
      }
      yb = TMath::Max(yb, ymin);
      yb = TMath::Min(yb, ymax);

      if (Hoption.Plus) {
//*-*         compute y1, y2
         y1 = keepy[j-first];
         y2 = yb;
         if (Hoption.Plus == 1 || Hoption.Plus == -1) {
            if (Hoption.Logy)
               y2 = TMath::Log10(TMath::Power(10,y1) + Hoption.Plus*TMath::Power(10,y2));
            else
               y2 = y1 +y2*Hoption.Plus;
         }
//*-*         compute ym1, ym2
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

//*-*-           Compute X1 X2
         if (Hoption.Logy && c1 <= 0) continue;

//*-*- Update the current plot

         if (Hoption.Plus == 1 || Hoption.Plus == -1) keepy[j-first-1] = y2;
      }
      else {
         keepy[j-first] = yb;
      }
   } // end of for loop

   if (Hoption.Plus) return;

//*-*-          Draw histogram according to value of FillStyle and FillColor

   if (fixbin) { keepx[0] = Hparam.xmin; keepx[1] = Hparam.xmax; }
   else {
      if (Hoption.Line) {
         for (i=0; i<=nbins; i++) keepx[i] = fXaxis->GetBinCenter(i+first);
      } else {
         for (i=0; i<=nbins; i++) keepx[i] = fXaxis->GetBinLowEdge(i+first);
      }
   }

//*-*-     Prepare Fill area (systematic with option "Bar").

   oldhtype = fH->GetFillStyle();
   htype    = oldhtype;
   if (Hoption.Bar) {
      if (htype == 0 || htype == 1000) htype = 1001;
   }

   //Width_t lw = Width_t(fH->GetLineWidth()*gPad->GetAbsHNDC() + 0.5);
   Width_t lw = Width_t(fH->GetLineWidth()*gPad->GetWh()/800. + 0.5);

//*-*-     Code option for GrapHist

   if (Hoption.Line) chopth[0] = 'L';
   if (Hoption.Star) chopth[1] = '*';
   if (Hoption.Mark) chopth[2] = 'P';
   if (Hoption.Curve || Hoption.Hist || Hoption.Bar) {
      if (Hoption.Curve)      chopth[3] = 'C';
      if (Hoption.Hist > 0) { chopth[4] = 'H';  strcpy(choptg,"H"); }
      else if (Hoption.Bar) { chopth[5] = 'B';  strcpy(choptg,"B"); }
      if (fH->GetFillColor() && htype) {
         if (Hoption.Logy) {
            chopth[6] = '1';
            if (strlen(choptg)) {
               strcpy(chtemp,"1");
               strcat(chtemp, choptg);
               strncpy(choptg, chtemp,4);
            }
         }
         if (Hoption.Hist > 0 || Hoption.Curve) {
            chopth[7] = 'F';
            if (strlen(choptg)) {
               strcpy(chtemp,"F");
               strcat(chtemp, choptg);
               strncpy(choptg, chtemp,4);
            }
         }
      }
   }
   if (!fixbin && strlen(chopth)) {
      chopth[8] = 'N';
      if (strlen(choptg)) {
         strcpy(chtemp,"N");
         strcat(chtemp, choptg);
         strncpy(choptg, chtemp,4);
      }
   }

   if (!strlen(chopth)) {
      if (fH->GetFillColor() && htype) {
         if (!Hoption.Logy) {
            if (!Hoption.Bar) {
               if (fixbin) { strcpy(chopth,"F");   strcpy(choptg,"F   "); }
               else        { strcpy(chopth,"FN");  strcpy(choptg,"FN  "); }
            }
            else {
               if (fixbin) { strcpy(chopth,"B");   strcpy(choptg,"B   "); }
               else        { strcpy(chopth,"BN");  strcpy(choptg,"BN  "); }
            }
         }
         else {
            if (!Hoption.Bar) {
               if (fixbin) { strcpy(chopth,"F1");  strcpy(choptg,"F1  "); }
               else        { strcpy(chopth,"F1N"); strcpy(choptg,"F1N "); }
            }
            else {
               if (fixbin) { strcpy(chopth,"B1");  strcpy(choptg,"B1  "); }
               else        { strcpy(chopth,"B1N"); strcpy(choptg,"B1N "); }
            }
         }
      }
      else {
         if (fixbin) { strcpy(chopth,"H");   strcpy(choptg,"H   "); }
         else        { strcpy(chopth,"HN");  strcpy(choptg,"HN  "); }
      }
   }

//*-*-     Option LOGX

   if (Hoption.Logx) {
      chopth[9]  = 'G';
      chopth[10] = 'X';
      if (strlen(choptg)) {
         strcpy(chtemp,"GX");
         strcat(chtemp, choptg);
         strncpy(choptg, chtemp,4);
      }
      if (fixbin) {
         keepx[0] = TMath::Power(10,keepx[0]);
         keepx[1] = TMath::Power(10,keepx[1]);
      }
   }

//*-*-     Draw the histogram

   TGraph graph;
   graph.SetLineWidth(lw);
   graph.SetLineStyle(fH->GetLineStyle());
   graph.SetLineColor(fH->GetLineColor());
   graph.SetFillStyle(htype);
   graph.SetFillColor(fH->GetFillColor());
   if (fixbin) {
      graph.PaintGrapHist(nbins, keepx, keepy ,chopth);
   }
   else {
      graph.PaintGrapHist(nbins, keepx, keepy ,chopth);
   }
   delete [] keepx;
   delete [] keepy;
   gStyle->SetBarOffset(baroffsetsave);
   gStyle->SetBarWidth(barwidthsave);

   htype=oldhtype;
}

//______________________________________________________________________________
void THistPainter::PaintH3(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Control function to draw a 3-D histogram*-*-*-*-*-*-*-*-*
//*-*                ========================================

   char *cmd;
   if (fH->GetDrawOption() && strstr(fH->GetDrawOption(),"box")) {
      cmd = Form("TMarker3DBox::PaintH3((TH1 *)0x%lx,\"%s\");",(Long_t)fH,option);
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
}

//______________________________________________________________________________
Int_t THistPainter::PaintInit()
{
//*-*-*-*-*-*-*Compute histogram parameters used by the drawing routines*-*-*-*
//*-*          =========================================================

   if (fH->GetDimension() > 1 || Hoption.Lego || Hoption.Surf) return 1;

   static const char *where = "PaintInit";
   Double_t YMARGIN = 0.05;
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

//*-*  Compute X axis parameters

   Int_t last      = fXaxis->GetLast();
   Int_t first     = fXaxis->GetFirst();
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.xmin     = Hparam.xlowedge;
   Hparam.xmax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

//*-*-   if log scale in X, replace xmin,max by the log
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
      if (Hparam.xlast > last) Hparam.xlast = last;
   }

//*-*  Compute Y axis parameters
   Double_t BIGP = TMath::Power(10,32);
   Double_t ymax = -BIGP;
   Double_t ymin = BIGP;
   Double_t c1, e1;
   Double_t xv[1];
   Double_t fval;
   Int_t i;
   TObject *f;
   TF1 *f1;
   Stat_t allchan = 0;
   TIter   next(fFunctions);
   for (i=first; i<=last;i++) {
      c1 = fH->GetBinContent(i);
      ymax = TMath::Max(ymax,c1);
      ymin = TMath::Min(ymin,c1);
      if (Hoption.Error) {
         e1 = fH->GetBinError(i);
         ymax = TMath::Max(ymax,c1+e1);
         ymin = TMath::Min(ymin,c1-e1);
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
                  if (fval > 0.3*c1) ymin = TMath::Min(ymin,fval);
               }
            }
         }
         next.Reset();
      }
      allchan += c1;
   }

//*-*- Take into account maximum , minimum

   if (Hoption.Logy && ymin <= 0) {
      if (ymax >= 1) ymin = TMath::Max(.5,ymax*1e-10);
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
            Error(where, "log scale is requested but maximum is less or equal 0 (%f)", ymax);
            return 0;
         }
      }
      else {
         ymax = ymin + 1;
         ymin = ymin - 1;
      }
   }

//*-*- take into account normalization factor
   Hparam.allchan = allchan;
   Stat_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   ymax = factor*ymax;
   ymin = factor*ymin;

//*-*-----
//*-*-     For log scales, histogram coordinates are LOG10(ymin) and
//*-*-     LOG10(ymax). Final adjustment (if not option "Same"
//*-*-     or "+" for ymax) of ymax and ymin for logarithmic scale, if
//*-*-     Maximum and Minimum are not defined.
//*-*-----
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
//*-*-----
//*-*-     final adjustment of ymin for linear scale.
//*-*-     if minimum is not set , then ymin is set to zero if >0
//*-*-     or to ymin - margin if <0.
//*-*-----
   if (!minimum) {
      if (ymin >= 0) ymin = 0;
      else           ymin -= YMARGIN*(ymax-ymin);
   }
//*-*-----
//*-*-     final adjustment of YMAXI for linear scale (if not option "Same"):
//*-*-     decrease histogram height to MAX% of allowed height if HMAXIM
//*-*-     has not been called.
//*-*-----
   if (!maximum && !Hoption.Plus) {
      ymax += YMARGIN*(ymax-ymin);
   }
   Hparam.ymin = ymin;
   Hparam.ymax = ymax;
   return 1;
}


//______________________________________________________________________________
void THistPainter::PaintLego()
{
//*-*-*-*-*-*-*-*Control function to draw a table as a lego plot*-*-*-*-*-*
//*-*            ===============================================
//*-*
//*-*     In a lego plot, cell contents are represented as 3-d boxes.
//*-*     The height of the box is proportional to the cell content.
//*-*
//*-*    A lego plot can be represented in several coordinate systems.
//*-*    Default system is Cartesian coordinates.
//*-*    Possible systems are CYL,POL,SPH,PSR.
//*-*
//*-*   See THistPainter::Draw for the list of Lego options.
//*-*   See TLego for more examples of lego options.
//
//      See TStyle::SetPalette to change the color palette.
//      It is suggested to use palette 1 via the call
//      gStyle->SetColorPalette(1)
//
//Begin_Html
/*
<img src="gif/PaintLego1.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t i;
   Int_t raster = 1;
   TH1 *hid;
   fNIDS = 0;
   hlist[0] = fH;
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
   Double_t z1c = zmin;
   Double_t z2c = zmin + (zmax-zmin)*kHMAX;


//*-*- Compute the lego limits and instantiate a lego object
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

   fLego = new TLego(fXbuf, fYbuf, Hoption.System);

//*-*-      Create axis object

   TGaxis *axis = new TGaxis();

//*-*-              Initialize the levels on the Z axis
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = 20;
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);

//*-*- Initialize colors for the lighting model (option Lego1 only)
   if (Hoption.Lego == 1) {
         Color_t colormain = fH->GetLineColor();
         fLego->SetColorMain(colormain,0);
   }
   if (Hoption.Lego == 11) {
      for (Int_t id=0;id<=fNIDS;id++) {
         hid = hlist[id];
         Color_t colormain = hid->GetFillColor();
         if (colormain == 1) colormain = 17; //avoid drawing with black
         Color_t colordark = colormain + 100;
         fLego->SetColorMain(colormain,id);
         fLego->SetColorDark(colordark,id);
         if (id == 0)     fLego->SetColorMain(colormain,-1);  // Set Bottom color
         if (id == fNIDS) fLego->SetColorMain(colormain,99);  // Set Top color
      }
   }

//*-*- Now ready to draw the lego plot
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

//*-*- Set color/style for back box
   fLego->SetFillStyle(gStyle->GetFrameFillStyle());
   fLego->SetFillColor(gStyle->GetFrameFillColor());
   fLego->TAttFill::Modify();

   Int_t backcolor = gStyle->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

   if (raster) fLego->InitRaster(-1.1,-1.1,1.1,1.1,1000,800);
   else        fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.Lego == 11 || Hoption.Lego == 12) {
//      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TLego::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }

//*-*-             Initialize the colour levels
   if (ndivz >= 100) {
      Warning("PaintLego", "too many color levels, %d, reset to 8", ndivz);
      ndivz = 8;
   }
   Double_t *funlevel = new Double_t[ndivz+1];
   Int_t *colorlevel = new Int_t[ndivz+1];
   Int_t lowcolor = fH->GetFillColor();
   Int_t theColor;
   Int_t ncolors = gStyle->GetNumberOfColors();
   for (i = 0; i < ndivz; ++i) {
      funlevel[i]   = fH->GetContourLevel(i);
      theColor = lowcolor + Int_t(i*Float_t(ncolors)/Float_t(ndivz));
      colorlevel[i] = gStyle->GetColorPalette(theColor);
   }
   colorlevel[ndivz] = gStyle->GetColorPalette(lowcolor+ncolors-1);
   fLego->ColorFunction(ndivz, funlevel, colorlevel, irep);
   delete [] colorlevel;
   delete [] funlevel;

   fLego->SetLegoFunction(&TLego::LegoFunction);
   if (Hoption.Lego ==  1) fLego->SetDrawFace(&TLego::DrawFaceRaster2);
   if (Hoption.Lego == 11) fLego->SetDrawFace(&TLego::DrawFaceMode3);
   if (Hoption.Lego == 12) fLego->SetDrawFace(&TLego::DrawFaceMode2);
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
                              fLego->SetDrawFace(&TLego::DrawFaceMove2);
                              fLego->LegoCartesian(90,nx,ny,"FB");}
      if (Hoption.Lego == 11) fLego->LegoCartesian(90,nx,ny,"BF");
      if (Hoption.Lego == 12) fLego->LegoCartesian(90,nx,ny,"BF");
   }

   if (Hoption.Lego == 1 || Hoption.Lego == 11) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TLego::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }
   if (Hoption.System == kCARTESIAN) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TLego::DrawFaceMove2);
      if (Hoption.FrontBox) fLego->FrontBox(90);
   }
   if (!Hoption.Axis) PaintLegoAxis(axis, 90);
   if (Hoption.Lego == 12 && Hoption.Zscale) PaintPalette();  // MOD MWH
   fNIDS = 0;
   delete axis;
   delete fLego; fLego = 0;
}

//______________________________________________________________________________
void THistPainter::PaintLegoAxis(TGaxis *axis, Double_t ang)
{
//*-*-*-*-*-*-*Draw the axis for legos and surface plots*-*-*-*-*-*-*-*-*-*
//*-*          =========================================
//*-*

    static Double_t epsil = 0.001;

    Double_t cosa, sina;
    Double_t bmin, bmax;
    Double_t r[24]	/* was [3][8] */;
    Int_t ndivx, ndivy, ndivz, i;
    Double_t x1[3], x2[3], y1[3], y2[3], z1[3], z2[3], av[24]  /*  was [3][8] */;
    static char chopax[8], chopay[8], chopaz[8];
    Int_t ix1, ix2, iy1, iy2, iz1, iz2;
    Double_t rad;

//*-*-----------------------------------------------------------------------


    TView *view = gPad->GetView();
    if (!view) {
       Error("PaintLegoAxis", "no TView in current pad");
       return;
    }
    
    // in polar coordinates, draw a short line going from the external circle
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

//*-*-             Initialize the axis options

    if (x1[0] > x2[0]) strcpy(chopax, "SDH=+");
    else               strcpy(chopax, "SDH=-");
    if (y1[0] > y2[0]) strcpy(chopay, "SDH=+");
    else               strcpy(chopay, "SDH=-");
    strcpy(chopaz, "SDH+=");

//*-*-             Option LOG is required ?

    if (Hoption.Logx) strcat(chopax,"G");
    if (Hoption.Logy) strcat(chopay,"G");
    if (Hoption.Logz) strcat(chopaz,"G");

//*-*-          Initialize the number of divisions. If the
//*-*-          number of divisions is negative, option 'N' is required.

    ndivx = fXaxis->GetNdivisions();
    ndivy = fYaxis->GetNdivisions();
    ndivz = fZaxis->GetNdivisions();
    if (ndivx < 0) {
	ndivx = TMath::Abs(ndivx);
	chopax[6] = 'N';
    }
    if (ndivy < 0) {
	ndivy = TMath::Abs(ndivy);
	chopay[6] = 'N';
    }
    if (ndivz < 0) {
	ndivz = TMath::Abs(ndivz);
	chopaz[6] = 'N';
    }

//*-*-          Set Axis attributes.
//*-*-          The variable SCALE  rescales the VSIZ
//*-*-          in order to have the same label size for all angles.

    axis->SetLineWidth(1);

//*-*-          X axis drawing

    if (TMath::Abs(x1[0] - x2[0]) >= epsil || TMath::Abs(x1[1] - x2[1]) > epsil) {
        axis->SetLineColor(fXaxis->GetAxisColor());
        axis->SetTextFont(fXaxis->GetTitleFont());
	axis->SetTextColor(fXaxis->GetTitleColor());
	axis->SetTickSize(fXaxis->GetTickLength());
        axis->SetLabelColor(fXaxis->GetLabelColor());
        axis->SetLabelFont(fXaxis->GetLabelFont());
	axis->SetLabelOffset(fXaxis->GetLabelOffset()+fXaxis->GetTickLength());
        axis->SetLabelSize(fXaxis->GetLabelSize());
        axis->SetTitle(fXaxis->GetTitle());
        axis->SetTitleOffset(fXaxis->GetTitleOffset());
        axis->SetTitleSize(fXaxis->GetTitleSize());
        axis->SetBit(TGaxis::kCenterTitle, fXaxis->TestBit(TGaxis::kCenterTitle));
        axis->SetBit(TGaxis::kRotateTitle, fXaxis->TestBit(TGaxis::kRotateTitle));
	if (Hoption.Logx) {
	    bmin = TMath::Power(10, rmin[0]);
	    bmax = TMath::Power(10, rmax[0]);
	} else {
	    bmin = rmin[0];
	    bmax = rmax[0];
	}
//*-*-             Option time display is required ?
        if (fXaxis->GetTimeDisplay()) {
           strcat(chopax,"t");
           if (strlen(fXaxis->GetTimeFormat()) == 0) {
              axis->SetTimeFormat(fXaxis->ChooseTimeFormat(bmax-bmin));
           } else {
              axis->SetTimeFormat(fXaxis->GetTimeFormat());
           }
        }
   axis->SetOption(chopax);
	axis->PaintAxis(x1[0], x1[1], x2[0], x2[1], bmin, bmax, ndivx, chopax);
    }

//*-*-          Y axis drawing

    if (TMath::Abs(y1[0] - y2[0]) >= epsil || TMath::Abs(y1[1] - y2[1]) > epsil) {
        axis->SetLineColor(fYaxis->GetAxisColor());
        axis->SetTextFont(fYaxis->GetTitleFont());
	axis->SetTextColor(fYaxis->GetTitleColor());
	axis->SetTickSize(fYaxis->GetTickLength());
        axis->SetLabelColor(fYaxis->GetLabelColor());
        axis->SetLabelFont(fYaxis->GetLabelFont());
	axis->SetLabelOffset(fYaxis->GetLabelOffset()+fYaxis->GetTickLength());
        axis->SetLabelSize(fYaxis->GetLabelSize());
        axis->SetTitle(fYaxis->GetTitle());
        axis->SetTitleOffset(fYaxis->GetTitleOffset());
        axis->SetTitleSize(fYaxis->GetTitleSize());
        axis->SetBit(TGaxis::kCenterTitle, fYaxis->TestBit(TGaxis::kCenterTitle));
        axis->SetBit(TGaxis::kRotateTitle, fYaxis->TestBit(TGaxis::kRotateTitle));

	if (TMath::Abs(z1[0] - z2[0]) < epsil && TMath::Abs(z1[1] - z2[1]) < epsil) {
	    strcpy(chopay, "SDH+=N");
	}
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
//*-*-             Option time display is required ?
        if (fYaxis->GetTimeDisplay()) {
           strcat(chopay,"t");
           if (strlen(fYaxis->GetTimeFormat()) == 0) {
              axis->SetTimeFormat(fYaxis->ChooseTimeFormat(bmax-bmin));
           } else {
              axis->SetTimeFormat(fYaxis->GetTimeFormat());
           }
        }
   axis->SetOption(chopay);
	axis->PaintAxis(y1[0], y1[1], y2[0], y2[1], bmin, bmax, ndivy, chopay);
    }

//*-*-          Z axis drawing

    if (TMath::Abs(z1[0] - z2[0]) >= epsil || TMath::Abs(z1[1] - z2[1]) > epsil) {
        axis->SetLineColor(fZaxis->GetAxisColor());
        axis->SetTextFont(fZaxis->GetTitleFont());
	axis->SetTextColor(fZaxis->GetTitleColor());
	axis->SetTickSize(fZaxis->GetTickLength());
        axis->SetLabelColor(fZaxis->GetLabelColor());
        axis->SetLabelFont(fZaxis->GetLabelFont());
	axis->SetLabelOffset(fZaxis->GetLabelOffset());
        axis->SetLabelSize(fZaxis->GetLabelSize());
        axis->SetTitle(fZaxis->GetTitle());
        axis->SetTitleOffset(fZaxis->GetTitleOffset());
        axis->SetTitleSize(fZaxis->GetTitleSize());
        axis->SetBit(TGaxis::kCenterTitle, fZaxis->TestBit(TGaxis::kCenterTitle));
        axis->SetBit(TGaxis::kRotateTitle, fZaxis->TestBit(TGaxis::kRotateTitle));
	if (Hoption.Logz) {
	    bmin = TMath::Power(10, rmin[2]);
	    bmax = TMath::Power(10, rmax[2]);
	} else {
	    bmin = rmin[2];
	    bmax = rmax[2];
	}
//*-*-             Option time display is required ?
        if (fZaxis->GetTimeDisplay()) {
           strcat(chopaz,"t");
           if (strlen(fZaxis->GetTimeFormat()) == 0) {
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
//*-*-*-*-*-*-*-*Paint the color palette on the right side of the pad*-*-*-*-*
//*-*            ====================================================

   Double_t xup  = gPad->GetUxmax();
   Double_t x2   = gPad->GetX2();
   Double_t ymin = gPad->GetUymin();
   Double_t ymax = gPad->GetUymax();
   Double_t xr   = 0.05*(x2 - gPad->GetX1());
   Double_t xmin = xup +0.1*xr;
   Double_t xmax = xmin + xr;
   if (xmax > x2) xmax = x2-0.01*xr;
   Int_t ncolors = gStyle->GetNumberOfColors();
   Double_t dy = (ymax-ymin)/ncolors;
   Color_t colorsav = fH->GetFillColor();
   for (Int_t i=0;i<ncolors;i++) {
      fH->SetFillColor(gStyle->GetColorPalette(i));
      fH->TAttFill::Modify();
      gPad->PaintBox(xmin,ymin+i*dy,xmax,ymin+(i+1)*dy);
   }
   fH->SetFillColor(colorsav);
   TAxis *zaxis = fH->GetZaxis();
   //Draw the palette axis using the Z axis parameters
   TGaxis axis;
   axis.SetLineColor(zaxis->GetAxisColor());
   axis.SetTextColor(zaxis->GetTitleColor());
   axis.SetTextFont(zaxis->GetTitleFont());
   axis.SetLabelColor(zaxis->GetLabelColor());
   axis.SetLabelFont(zaxis->GetLabelFont());
   axis.SetLabelSize(zaxis->GetLabelSize());
   axis.SetLabelOffset(zaxis->GetLabelOffset());
   axis.SetTickSize(zaxis->GetTickLength());
   axis.SetTitle(zaxis->GetTitle());
   axis.SetTitleOffset(zaxis->GetTitleOffset());
   axis.SetTitleSize(zaxis->GetTitleSize());
   axis.SetBit(TGaxis::kCenterTitle, zaxis->TestBit(TGaxis::kCenterTitle));
   axis.SetBit(TGaxis::kRotateTitle, zaxis->TestBit(TGaxis::kRotateTitle));
   Int_t ndiv = zaxis->GetNdivisions();
   Double_t wmin = Hparam.zmin;
   Double_t wmax = Hparam.zmax;
   if (Hoption.Logz) {
      wmin = TMath::Power(10.,wmin);
      wmax = TMath::Power(10.,wmax);
      axis.PaintAxis(xmax,ymin,xmax,ymax,wmin,wmax,ndiv,"+LG");
   } else {
      axis.PaintAxis(xmax,ymin,xmax,ymax,wmin,wmax,ndiv,"+L");
   }
}

//______________________________________________________________________________
void THistPainter::PaintScatterPlot()
{
//*-*-*-*-*-*-*-*Control function to draw a table as a scatter plot*-*-*-*-*
//*-*            ==================================================
//*-*
//*-*    For each cell (i,j) a number of points proportional to the cell
//*-*    content is drawn.
//*-*    A maximum of 500 points per cell is drawn. If the maximum is above 500
//*-*    contents are normalized to 500.
//Begin_Html
/*
<img src="gif/PaintScatterPlot.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   fH->TAttMarker::Modify();

   Int_t k, marker;
   Double_t dz, z, xk,xstep, yk, ystep;
   Double_t scale = 1;
   Bool_t ltest  = kFALSE;
   Double_t zmax  = fH->GetMaximum();
   Double_t zmin  = fH->GetMinimum();
   if (zmin == 0 && zmax == 0) return;
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
         z     = fH->GetBinContent(bin);
         if (Hoption.Logz) {
            if (z > 0) z = TMath::Log10(z) - zmin;
         } else {
            z    -=  zmin;
         }
         if (z <= 0) continue;
         k = Int_t(z*scale);
         if (ltest) k++;
         if (k > 0) {
            if (k+marker >= kNMAX) {
               gPad->PaintPolyMarker(marker, fXbuf, fYbuf);
               marker=0;
            }
            for (Int_t loop=0; loop<k; loop++) {
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
}


//______________________________________________________________________________
void THistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw the statistics box*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =======================
// The type of information printed in the histogram statistics box
//  can be selected via gStyle->SetOptStat(mode).
//  The parameter mode can be = iourmen  (default = 0001111)
//    n = 1;  name of histogram is printed
//    e = 1;  number of entries printed
//    m = 1;  mean value printed
//    r = 1;  rms printed
//    u = 1;  number of underflows printed
//    o = 1;  number of overflows printed
//    i = 1;  integral of bins printed
//  Example: gStyle->SetOptStat(11);
//           print only name of histogram and number of entries.
//
// The type of information about fit parameters printed in the histogram
// statistics box can be selected via the parameter mode.
//  The parameter mode can be = pcev  (default = 0111)
//    v = 1;  print name/values of parameters
//    e = 1;  print errors (if e=1, v must be 1)
//    c = 1;  print Chisquare/Number of degress of freedom
//    p = 1;  print Probability
//  Example: gStyle->SetOptFit(1011);
//           print fit probability, parameter names/values and errors.
//
//
//  When option "same" is specified, the statistic box is not drawn.
//  Specify option "sames" to force painting statistics with option "same"
//  When option "sames" is given, one can use the following technique
//  to rename a previous "stats" box and/or change its position
//  Root > TPaveStats *st = (TPaveStats*)gPad->FindObject("stats")
//  Root > st->SetName(newname)
//  Root > st->SetX1NDC(newx1); //new x start position
//  Root > st->SetX2NDC(newx2); //new x end position
//  Root > newhist->Draw("sames")

   static char t[64];
   Int_t dofit;
   TPaveStats *stats  = (TPaveStats*)gPad->FindObject("stats");
   if (stats) {
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
   Int_t nlines = print_name + print_entries + print_mean + print_rms + print_under + print_over + print_integral;
   Int_t print_fval    = dofit%10;
   Int_t print_ferrors = (dofit/10)%10;
   Int_t print_fchi2   = (dofit/100)%10;
   Int_t print_fprob   = (dofit/1000)%10;
   Int_t nlinesf = print_fval + print_fchi2 + print_fprob;
   if (fit) nlinesf += fit->GetNpar();

//*-*- Pavetext with statistics
   Bool_t done = kFALSE;
   if (!dostat && !fit) {
      if (stats) delete stats;
      return;
   }
   Double_t  statw  = gStyle->GetStatW();
   if (fit) statw   = 1.8*gStyle->GetStatW();
   Double_t  stath  = 0.25*(nlines+nlinesf)*gStyle->GetStatH();
   if (stats) {
      stats->Clear();
      done = kTRUE;
   } else {
      stats  = new TPaveStats(
               gStyle->GetStatX()-statw,
               gStyle->GetStatY()-stath,
               gStyle->GetStatX(),
               gStyle->GetStatY(),"brNDC");
      stats->SetOptFit(dofit);
      stats->SetOptStat(dostat);
      stats->SetFillColor(gStyle->GetStatColor());
      stats->SetFillStyle(gStyle->GetStatStyle());
      stats->SetBorderSize(gStyle->GetStatBorderSize());
      stats->SetTextFont(gStyle->GetStatFont());
      stats->SetFitFormat(gStyle->GetFitFormat());
      stats->SetStatFormat(gStyle->GetStatFormat());
      stats->SetName("stats");

      stats->SetTextColor(gStyle->GetStatTextColor());
      stats->SetTextAlign(12);
      stats->SetBit(kCanDelete);
   }
   if (print_name)  stats->AddText(fH->GetName());
   if (print_entries) {
      if (fH->GetEntries() < 1e7) sprintf(t,"Nent = %-7d",Int_t(fH->GetEntries()));
      else                        sprintf(t,"Nent = %14.7g",Float_t(fH->GetEntries()));
      stats->AddText(t);
   }
   char fstats[50];
   if (print_mean) {
      sprintf(fstats,"Mean  = %s%s","%",stats->GetStatFormat());
      sprintf(t,fstats,fH->GetMean(1));
      stats->AddText(t);
   }
   if (print_rms) {
      sprintf(fstats,"RMS   = %s%s","%",stats->GetStatFormat());
      sprintf(t,fstats,fH->GetRMS(1));
      stats->AddText(t);
   }
   if (print_under) {
      sprintf(fstats,"Under = %s%s","%",stats->GetStatFormat());
      sprintf(t,fstats,fH->GetBinContent(0));
      stats->AddText(t);
   }
   if (print_over) {
      sprintf(fstats,"Over  = %s%s","%",stats->GetStatFormat());
      sprintf(t,fstats,fH->GetBinContent(fXaxis->GetNbins()+1));
      stats->AddText(t);
   }
   if (print_integral) {
      sprintf(fstats,"Integ = %s%s","%",stats->GetStatFormat());
      sprintf(t,fstats,fH->Integral());
      stats->AddText(t);
   }

//     Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNumberFitPoints() - fit->GetNpar();
      sprintf(fstats,"Chi2 / ndf = %s%s / %d","%",stats->GetFitFormat(),ndf);
      sprintf(t,fstats,(Float_t)fit->GetChisquare());
      if (print_fchi2) stats->AddText(t);
      if (print_fprob) {
         sprintf(fstats,"Prob  = %s%s","%",stats->GetFitFormat());
         sprintf(t,fstats,(Float_t)TMath::Prob(fit->GetChisquare(),ndf));
         stats->AddText(t);
      }
      if (print_fval || print_ferrors) {
         for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
            if (print_ferrors) {
               sprintf(fstats,"%-8s = %s%s #pm %s%s ",fit->GetParName(ipar),"%",stats->GetFitFormat(),"%",stats->GetFitFormat());
               sprintf(t,fstats,(Float_t)fit->GetParameter(ipar)
                               ,(Float_t)fit->GetParError(ipar));
            } else {
               sprintf(fstats,"%-8s = %s%s ",fit->GetParName(ipar),"%",stats->GetFitFormat());
               sprintf(t,fstats,(Float_t)fit->GetParameter(ipar));
            }
            t[63] = 0;
            stats->AddText(t);
         }
      }
   }

   if (!done) stats->Draw();
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
   //    r = 1;  rms printed
   //    u = 1;  number of underflows printed
   //    o = 1;  number of overflows printed
   //    i = 1;  integral of bins printed
   //  Example: gStyle->SetOptStat(11);
   //           print only name of histogram and number of entries.

   if (fH->GetDimension() != 2) return;
   TH2 *h2 = (TH2*)fH;

   static char t[64];
   Int_t dofit;
   TPaveStats *stats  = (TPaveStats*)gPad->FindObject("stats");
   if (stats) {
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
   Double_t  stath  = 0.25*nlines*gStyle->GetStatH();
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
   }
   if (print_name)  stats->AddText(h2->GetName());
   if (print_entries) {
      if (h2->GetEntries() < 1e7) sprintf(t,"Nent = %-7d",Int_t(h2->GetEntries()));
      else                        sprintf(t,"Nent = %14.7g",Float_t(h2->GetEntries()));
      stats->AddText(t);
   }
   if (print_mean) {
      sprintf(t,"Mean x = %6.4g",h2->GetMean(1));
      stats->AddText(t);
      sprintf(t,"Mean y = %6.4g",h2->GetMean(2));
      stats->AddText(t);
   }
   if (print_rms) {
      sprintf(t,"RMS x  = %6.4g",h2->GetRMS(1));
      stats->AddText(t);
      sprintf(t,"RMS y  = %6.4g",h2->GetRMS(2));
      stats->AddText(t);
   }
   if (print_under || print_over) {
      //get 3*3 under/overflows for 2d hist
      Stat_t unov[9];

      unov[0] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetLast()+1);
      unov[1] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetLast()+1);
      unov[2] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetLast()+1,h2->GetYaxis()->GetLast()+1,h2->GetYaxis()->GetLast()+1);
      unov[3] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[4] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[5] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetLast()+1,h2->GetYaxis()->GetFirst(),h2->GetYaxis()->GetLast());
      unov[6] = h2->Integral(0,h2->GetXaxis()->GetFirst()-1,0,h2->GetYaxis()->GetFirst()-1);
      unov[7] = h2->Integral(h2->GetXaxis()->GetFirst(),h2->GetXaxis()->GetLast(),0,h2->GetYaxis()->GetFirst()-1);
      unov[8] = h2->Integral(h2->GetXaxis()->GetLast()+1,h2->GetXaxis()->GetLast()+1,0,h2->GetYaxis()->GetFirst()-1);

      sprintf(t, " %7d %7d %7d\n", (Int_t)unov[0], (Int_t)unov[1], (Int_t)unov[2]);
      stats->AddText(t);
      if (h2->GetEntries() < 1e7)
         sprintf(t, " %7d %7d %7d\n", (Int_t)unov[3], (Int_t)unov[4], (Int_t)unov[5]);
      else
         sprintf(t, " %7d %14.7g %7d\n", (Int_t)unov[3], (Float_t)unov[4], (Int_t)unov[5]);
      stats->AddText(t);
      sprintf(t, " %7d %7d %7d\n", (Int_t)unov[6], (Int_t)unov[7], (Int_t)unov[8]);
      stats->AddText(t);
   }
   if (print_integral) {
      sprintf(t,"Integ  = %6.4g",h2->Integral());
      stats->AddText(t);
   }

   // Draw Fit parameters
   if (fit) {
      Int_t ndf = fit->GetNumberFitPoints() - fit->GetNpar();
      sprintf(t,"Chi2 / ndf = %6.4g / %d",(Float_t)fit->GetChisquare(),ndf);
      stats->AddText(t);
      for (Int_t ipar=0;ipar<fit->GetNpar();ipar++) {
         sprintf(t,"%-8s = %5.4g #pm %5.4g ",fit->GetParName(ipar)
                                   ,(Float_t)fit->GetParameter(ipar)
                                   ,(Float_t)fit->GetParError(ipar));
         t[32] = 0;
         stats->AddText(t);
      }
   }

   if (!done) stats->Draw();
   stats->Paint();
}

//______________________________________________________________________________
void THistPainter::PaintSurface()
{
//*-*-*-*-*-*-*-*Control function to draw a table as a surface plot*-*-*-*-*-*
//*-*            ==================================================
//*-*
//*-*     In a surface plot, cell contents are represented as a mesh.
//*-*     The height of the mesh is proportional to the cell content.
//*-*
//*-*    A surface plot can be represented in several coordinate systems.
//*-*    Default system is Cartesian coordinates.
//*-*    Possible systems are CYL,POL,SPH,PSR.
//*-*
//*-*   See THistPainter::Draw for a list of Surface options
//*-*  The following picture is generated with option SURF1.
//
//      See TStyle::SetPalette to change the color palette.
//      It is suggested to use palette 1 via the call
//      gStyle->SetColorPalette(1)
//
//Begin_Html
/*
<img src="gif/PaintSurface1.gif">
*/
//End_Html
//*-*  The following picture is generated with option SURF3.
//Begin_Html
/*
<img src="gif/PaintSurface3.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   const Double_t ydiff = 1;
   const Double_t yligh1 = 10;
   const Double_t qa = 0.15;
   const Double_t qd = 0.15;
   const Double_t qs = 0.8;
   Double_t fmin, fmax;
   Int_t i;
   Int_t raster = 0;
   Int_t irep   = 0;

   fNIDS        = 0;

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
   Double_t z1c = zmin;
   Double_t z2c = zmin + (zmax-zmin)*kHMAX;


//*-*- Compute the lego limits and instantiate a lego object
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

   fLego = new TLego(fXbuf, fYbuf, Hoption.System);
   fLego->SetLineColor(fH->GetLineColor());
   fLego->SetFillColor(fH->GetFillColor());

//*-*-      Create axis object

   TGaxis *axis = new TGaxis();

//*-*-              Initialize the levels on the Z axis
   Int_t ndiv   = fH->GetContour();
   if (ndiv == 0 ) {
      ndiv = 20;
      fH->SetContour(ndiv);
   }
   Int_t ndivz  = TMath::Abs(ndiv);
   if (fH->TestBit(TH1::kUserContour) == 0) fH->SetContour(ndiv);

//*-*- Initialize colors for the lighting model
   Color_t colormain = fH->GetFillColor();
//   if (colormain == 1) colormain = 17; //avoid drawing with black
   Color_t colordark = colormain + 100;
   fLego->SetColorMain(colormain,0);
   fLego->SetColorDark(colordark,0);
   fLego->SetColorMain(colormain,-1);  // Set Bottom color
   fLego->SetColorMain(colormain,99);  // Set Top color
   if (Hoption.Surf == 13) fLego->SetMesh(3);
   if (Hoption.Surf == 12 || Hoption.Surf == 14) fLego->SetMesh(0);

//*-*- Close the surface in case of non cartesian coordinates.

   if (Hoption.System != kCARTESIAN) {nx++; ny++;}

//*-*-             Initialize the colour levels
   if (ndivz >= 100) {
      Warning("PaintSurface", "too many color levels, %d, reset to 8", ndivz);
      ndivz = 8;
   }
   Double_t *funlevel = new Double_t[ndivz+1];
   Int_t *colorlevel = new Int_t[ndivz+1];
   Int_t theColor;
   Int_t ncolors = gStyle->GetNumberOfColors();
   for (i = 0; i < ndivz; ++i) {
      funlevel[i]   = fH->GetContourLevel(i);
      theColor = Int_t(i*Float_t(ncolors)/Float_t(ndivz));
      colorlevel[i] = gStyle->GetColorPalette(theColor);
   }
   colorlevel[ndivz] = gStyle->GetColorPalette(ncolors-1);
   fLego->ColorFunction(ndivz, funlevel, colorlevel, irep);
   delete [] colorlevel;
   delete [] funlevel;

//*-*- Now ready to draw the surface plot

   TView *view = gPad->GetView();
   if (!view) {
      Error("PaintSurface", "no TView in current pad");
      return;
   }

   Double_t thedeg =  90 - gPad->GetTheta();
   Double_t phideg = -90 - gPad->GetPhi();
   Double_t psideg = view->GetPsi();
   view->SetView(phideg, thedeg, psideg, irep);

//*-*- Set color/style for back box
   fLego->SetFillStyle(gStyle->GetFrameFillStyle());
   fLego->SetFillColor(gStyle->GetFrameFillColor());
   fLego->TAttFill::Modify();

   Int_t backcolor = gStyle->GetFrameFillColor();
   if (Hoption.System != kCARTESIAN) backcolor = 0;
   view->PadRange(backcolor);

   fLego->SetFillStyle(fH->GetFillStyle());
   fLego->SetFillColor(fH->GetFillColor());
   fLego->TAttFill::Modify();

//*-*- Draw the filled contour on top
   Int_t icol1 = fH->GetFillColor();

   if (Hoption.Surf == 13) {
      Hoption.Surf = 23;
      fLego->SetSurfaceFunction(&TLego::SurfaceFunction);
      fLego->SetDrawFace(&TLego::DrawFaceMode2);
      nx--; ny--;
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"BF");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"BF");
      Hoption.Surf = 13;
      fLego->SetMesh(1);
   }

   if (raster) fLego->InitRaster(-1.1,-1.1,1.1,1.1,1000,800);
   else        fLego->InitMoveScreen(-1.1,1.1);

   if (Hoption.Surf == 11 || Hoption.Surf == 12 || Hoption.Surf == 14) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TLego::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }

//*-*- Gouraud Shading surface
   if (Hoption.Surf == 14) {
//*-* Set light sources
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
      colref->RGBtoHLS(r,g,b,hue,light,satur);
      TColor *acol;
      for (Int_t col=0;col<nbcol;col++) {
         acol = gROOT->GetColor(col+icol1);
         acol->HLStoRGB(hue,.4+col*dcol,satur,r,g,b);
         acol->SetRGB(r,g,b);
      }
      fLego->Spectrum(nbcol, fmin, fmax, icol1, 1, irep);
      fLego->SetSurfaceFunction(&TLego::GouraudFunction);
      fLego->SetDrawFace(&TLego::DrawFaceMode2);
      if (Hoption.System == kPOLAR)       fLego->SurfacePolar(1,nx,ny,"BF");
      if (Hoption.System == kCYLINDRICAL) fLego->SurfaceCylindrical(1,nx,ny,"BF");
      if (Hoption.System == kSPHERICAL)   fLego->SurfaceSpherical(0,1,nx,ny,"BF");
      if (Hoption.System == kRAPIDITY )   fLego->SurfaceSpherical(1,1,nx,ny,"BF");
      if (Hoption.System == kCARTESIAN)   fLego->SurfaceCartesian(90,nx,ny,"BF");
   }
   else {
//*-*- Draw the surface
      fLego->SetSurfaceFunction(&TLego::SurfaceFunction);
      if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SetDrawFace(&TLego::DrawFaceRaster1);
      if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SetDrawFace(&TLego::DrawFaceMode2);
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
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SetDrawFace(&TLego::DrawFaceMove1);
         if (Hoption.Surf ==  1 || Hoption.Surf == 13) fLego->SurfaceCartesian(90,nx,ny,"FB");
         if (Hoption.Surf == 11 || Hoption.Surf == 12) fLego->SurfaceCartesian(90,nx,ny,"BF");
      }
   }

   if (Hoption.Surf == 1 || Hoption.Surf == 13) {
      fLego->SetLineColor(1);
      if (Hoption.System == kCARTESIAN && Hoption.BackBox) {
         fLego->SetDrawFace(&TLego::DrawFaceMove1);
         fLego->BackBox(90);
      }
   }
   if (Hoption.System == kCARTESIAN) {
      fLego->InitMoveScreen(-1.1,1.1);
      fLego->SetDrawFace(&TLego::DrawFaceMove2);
      if (Hoption.FrontBox) fLego->FrontBox(90);
   }
   if (!Hoption.Axis) PaintLegoAxis(axis, 90);

   if (Hoption.Surf >= 11 && Hoption.Zscale) PaintPalette();  // MOD MWH
   
   fNIDS = 0;
   delete axis;
   delete fLego; fLego = 0;
}


//______________________________________________________________________________
void THistPainter::PaintTable()
{
//*-*-*-*-*-*-*-*-*-*Control function to draw 2-D/3-D tables*-*-*-*-*-*-*-*
//*-*                =======================================

   if (!TableInit()) return;  //fill Hparam structure with histo parameters

   PaintFrame();

   if (fH->GetEntries() >= 0) {
      if (Hoption.Scat)    PaintScatterPlot();
      if (Hoption.Arrow)   PaintArrows();
      if (Hoption.Box)     PaintBoxes();
      if (Hoption.Color)   PaintColorLevels();
      if (Hoption.Contour) PaintContour();
      if (Hoption.Text)    PaintText();
   }
   if (Hoption.Lego)    PaintLego();
   if (Hoption.Surf && !Hoption.Contour)    PaintSurface();

   if (!Hoption.Lego && !Hoption.Surf) PaintAxis();     //*-* Draw the axes

   PaintTitle();    //*-* Draw histogram title
//   PaintFile();     //*-* Draw Current File name corresp to current directory
//   PaintDate();     //*-* Draw date/time
//   if (!Hoption.Same && !Hoption.Lego && !Hoption.Surf) {
   if (Hoption.Same != 1) {
      TF2 *fit = (TF2*)fFunctions->First();
      if (!fH->TestBit(TH1::kNoStats)) {  // bit set via TH1::SetStats
         PaintStat2(gStyle->GetOptStat(),fit);
      }
   }
}


//______________________________________________________________________________
void THistPainter::PaintText()
{
//*-*-*-*-*-*Control function to draw a table with the bin values*-*-*-*-*-*
//*-*        ====================================================
//*-*
//*-*    For each cell (i,j) the cell content is printed.
//*-*    The text attributes are:
//*_*      - text font = current TStyle font
//*-*      - text size = 0.02*padheight*markersize
//*-*      - text color= marker color
//Begin_Html
/*
<img src="gif/PaintText.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   TText text;
   text.SetTextFont(gStyle->GetTextFont());
   text.SetTextColor(fH->GetMarkerColor());
   text.SetTextAlign(22);
   text.SetTextSize(0.02*fH->GetMarkerSize());
   text.TAttText::Modify();

   Double_t x, y, z;
   char value[50];

   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      y    = fYaxis->GetBinCenter(j);
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         Int_t bin  = j*(fXaxis->GetNbins()+2) + i;
         x    = fXaxis->GetBinCenter(i);
         z     = fH->GetBinContent(bin);
         if (z <= Hparam.zmin) continue;
         sprintf(value,"%g",z);
         gPad->PaintText(x,y,value);
      }
   }
}

//______________________________________________________________________________
void THistPainter::PaintTitle()
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw the histogram title*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================
   if (Hoption.Same) return;
   Int_t nt = strlen(fH->GetTitle());
   TPaveText *title  = (TPaveText*)gPad->FindObject("title");
   if (nt == 0 || gStyle->GetOptTitle() <= 0) {
      if (title) delete title;
      return;
   }
   if (title) {
      TText *t0 = (TText*)title->GetLine(0);
      if (t0) t0->SetTitle(fH->GetTitle());
      return;
   }
   Double_t ht = gStyle->GetTitleH();
   Double_t wt = gStyle->GetTitleW();
   if (ht <= 0) ht = 0.05;
   if (wt <= 0) wt = TMath::Min(0.6, 0.05+0.015*nt);

   TPaveText *ptitle = new TPaveText(
             gStyle->GetTitleX(),
             gStyle->GetTitleY()-ht,
             gStyle->GetTitleX()+wt,
             gStyle->GetTitleY(),"blNDC");

//*-*- box with the histogram title
   ptitle->SetFillColor(gStyle->GetTitleColor());
   ptitle->SetFillStyle(gStyle->GetTitleStyle());
   ptitle->SetName("title");
   ptitle->SetBorderSize(gStyle->GetTitleBorderSize());
//   ptitle->SetTextSize(0.85*ht);
   ptitle->SetTextColor(gStyle->GetTitleTextColor());
   ptitle->SetTextFont(gStyle->GetTitleFont());
   ptitle->AddText(fH->GetTitle());
   ptitle->SetBit(kCanDelete);
   ptitle->Draw();
   ptitle->Paint();

}

//______________________________________________________________________________
void THistPainter::RecalculateRange()
{
//*-*-*-*-*-*Recompute the histogram range following graphics operations*-*-*
//*-*        ===========================================================
   if (Hoption.Same) return;

//*-*- Compute x,y range
   Double_t xmin = Hparam.xmin;
   Double_t xmax = Hparam.xmax;
   Double_t ymin = Hparam.ymin;
   Double_t ymax = Hparam.ymax;
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
//   if (h == fH) return ;  //what happens if the old fH has been deleted
                            //and h created at the same address
   fH = h;
   fXaxis = h->GetXaxis();
   fYaxis = h->GetYaxis();
   fZaxis = h->GetZaxis();
   fFunctions = fH->GetListOfFunctions();
}

//______________________________________________________________________________
Int_t THistPainter::TableInit()
{
//*-*-*-*-*-*-*-*-*Initialize various options to draw tables*-*-*-*-*-*-*-*
//*-*              =========================================

   static const char *where = "TableInit";

   Int_t first, last;
   Double_t YMARGIN= 0.05;
   Double_t zmin, zmax;
   Int_t maximum = 0;
   Int_t minimum = 0;
   if (fH->GetMaximumStored() != -1111) maximum = 1;
   if (fH->GetMinimumStored() != -1111) minimum = 1;

//*-*------------------  Compute X axis parameters
   first           = fXaxis->GetFirst();
   last            = fXaxis->GetLast();
   Hparam.xlast    = last;
   Hparam.xfirst   = first;
   Hparam.xlowedge = fXaxis->GetBinLowEdge(first);
   Hparam.xbinsize = fXaxis->GetBinWidth(first);
   Hparam.xmin     = Hparam.xlowedge;
   Hparam.xmax     = fXaxis->GetBinLowEdge(last)+fXaxis->GetBinWidth(last);

//*-*-   if log scale in X, replace xmin,max by the log
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

//*-*------------------  Compute Y axis parameters
   first           = fYaxis->GetFirst();
   last            = fYaxis->GetLast();
   Hparam.ylast    = last;
   Hparam.yfirst   = first;
   Hparam.ylowedge = fYaxis->GetBinLowEdge(first);
   Hparam.ybinsize = fYaxis->GetBinWidth(first);
   if (!Hparam.ybinsize) Hparam.ybinsize = 1;
   Hparam.ymin     = Hparam.ylowedge;
   Hparam.ymax     = fYaxis->GetBinLowEdge(last)+fYaxis->GetBinWidth(last);

//*-*-   if log scale in Y, replace ymin,max by the log
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


//*-*------------------  Compute Z axis parameters
   Double_t BIGP = TMath::Power(10,32);
   zmax = -BIGP;
   zmin = BIGP;
   Double_t c1, e1;
   Stat_t allchan = 0;
   for (Int_t j=Hparam.yfirst; j<=Hparam.ylast;j++) {
      for (Int_t i=Hparam.xfirst; i<=Hparam.xlast;i++) {
         c1 = fH->GetCellContent(i,j);
         zmax = TMath::Max(zmax,c1);
         if (Hoption.Error) {
            e1 = fH->GetCellError(i,j);
            zmax = TMath::Max(zmax,c1+e1);
         }
         if (Hoption.Func) {
//            f1 = GetCellFunction(i,j);
//            zmax = TMath::Max(zmax,f1);
         }
         zmin = TMath::Min(zmin,c1);
         allchan += c1;
      }
   }

//*-*- Take into account maximum , minimum

   if (maximum) zmax = fH->GetMaximumStored();
   if (minimum) zmin = fH->GetMinimumStored();
   if (Hoption.Logz && zmax <= 0) {
      Error(where, "log scale is requested but maximum is less or equal 0 (%f)", zmax);
      return 0;
   }
   if (zmin >= zmax && !Hoption.Plus) {
      if (Hoption.Logz) {
         if (zmax > 0) zmin = 0.001*zmax;
         else {
            Error(where, "log scale is requested but maximum is less or equal 0 (%f)", zmax);
            return 0;
         }
      }
//      else {
//         zmax = zmin + 1;
//         zmin = zmin - 1;
//      }
   }

//*-*- take into account normalization factor
   Hparam.allchan = allchan;
   Stat_t factor = allchan;
   if (fH->GetNormFactor() > 0) factor = fH->GetNormFactor();
   if (allchan) factor /= allchan;
   if (factor == 0) factor = 1;
   Hparam.factor = factor;
   zmax = factor*zmax;
   zmin = factor*zmin;
   c1 = zmax;
   if (TMath::Abs(zmin) > TMath::Abs(c1)) c1 = zmin;

//*-*-----
//*-*-     For log scales, histogram coordinates are log10(ymin) and
//*-*-     log10(ymax). Final adjustment (if not option "Same")
//*-*-     or "+" for ymax) of ymax and ymin for logarithmic scale, if
//*-*-     Maximum and Minimum are not defined.
//*-*-----
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
//*-*-----
//*-*-     final adjustment of YMAXI for linear scale (if not option "Same"):
//*-*-     decrease histogram height to MAX% of allowed height if HMAXIM
//*-*-     has not been called.
//*-*-     MAX% is the value in percent which has been set in HPLSET
//*-*-     (default is 90%).
//*-*-----
   if (!maximum && !Hoption.Plus) {
      zmax += YMARGIN*(zmax-zmin);
   }
//*-*-----
//*-*-     final adjustment of ymin for linear scale.
//*-*-     if minimum is not set , then ymin is set to zero if >0
//*-*-     or to ymin - YMARGIN if <0.
//*-*-----
   if (!minimum) {
      if (zmin >= 0) zmin = 0;
      else           zmin -= YMARGIN*(zmax-zmin);
   }

LZMIN:
   Hparam.zmin = zmin;
   Hparam.zmax = zmax;

//*-*- Set bar offset and width
   Hparam.baroffset = fH->GetBarOffset();
   Hparam.barwidth  = fH->GetBarWidth();

   return 1;
}
