// @(#)root/gpad:$Name:  $:$Id: TDrawPanelHist.cxx,v 1.4 2001/08/07 13:44:45 brun Exp $
// Author: Rene Brun   26/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TDrawPanelHist.h"
#include "TGroupButton.h"
#include "TSlider.h"
#include "TText.h"
#include "TH1.h"
#include "TF1.h"

#include <stdio.h>

ClassImp(TDrawPanelHist)

//______________________________________________________________________________
//
//  A TDrawPanelHist is a TDialogCanvas specialized to control
//  histogram drawing options.
//   With the mouse, the user can control:
//     - the drawing range in X and Y
//     - the drawing options
//   When the DRAW button is executed, the selected histogram is drawn
//   with the current parameters.
//
//   Use the slider to control the range of the histogram to be drawn.
//
//   The options are documented in TH1::Draw.
//Begin_Html
/*
<img src="gif/drawpanelhist.gif">
*/
//End_Html
//

//______________________________________________________________________________
TDrawPanelHist::TDrawPanelHist() : TDialogCanvas()
{
   // DrawPanelHist default constructor.

}

//_____________________________________________________________________________
TDrawPanelHist::TDrawPanelHist(const char *name, const char *title, UInt_t ww, UInt_t wh)
          : TDialogCanvas(name, title,ww,wh)
{
   // DrawPanelHist constructor.

   TGroupButton *b;
   fOption    = "";
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   fRefObject = this;
   fHistogram = gROOT->GetSelectedPrimitive();

   BuildStandardButtons();

   // Range Slider
   fSlider = new TSlider("slider","X slider",.05,.11,.95,.17);
   fSlider->SetObject(this);
   if (fHistogram->InheritsFrom("TH1")) {
      TH1 *h = (TH1 *)fHistogram;
      Int_t nbins   = h->GetXaxis()->GetNbins();
      Int_t first   = h->GetXaxis()->GetFirst();
      Int_t last    = h->GetXaxis()->GetLast();
      Float_t xmin  = 0;
      Float_t xmax  = 1;
      if (first > 1) xmin = Float_t(first)/Float_t(nbins);
      if (last  > 1) xmax = Float_t(last)/Float_t(nbins);
      fSlider->SetRange(xmin,xmax);
   }

   // Standard functions
   Int_t nlines = 14;
   Float_t x0 = 0.1;
   Float_t y  = 0.98;
   Float_t dy = (y-0.18)/nlines;
   Float_t yb = 0.7*dy;

   y -= dy;
   b = new TGroupButton("DRAW","hist","AddOption(\"hist\")",x0,y-yb,.27,y);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","lego1","AddOption(\"lego1\")",x0,y-yb,.27,y);
      b->Draw();
   b = new TGroupButton("DRAW","lego2","AddOption(\"lego2\")",.3,y-yb,.47,y);
      b->Draw();
   b = new TGroupButton("DRAW","lego3","AddOption(\"lego3\")",.5,y-yb,.67,y);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","surf","AddOption(\"surf1\")",x0,y-yb,.27,y);
      b->Draw();
   b = new TGroupButton("DRAW","surf/colors","AddOption(\"surf2\")",.3,y-yb,.47,y);
      b->Draw();
   b = new TGroupButton("DRAW","surf/contour","AddOption(\"surf3\")",.5,y-yb,.67,y);
      b->Draw();
   b = new TGroupButton("DRAW","Gouraud","AddOption(\"surf4\")",.7,y-yb,.87,y);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","cont0","AddOption(\"cont0\")",x0,y-yb,.27,y);
      b->Draw();
   b = new TGroupButton("DRAW","cont1","AddOption(\"cont1\")",.3,y-yb,.47,y);
      b->Draw();
   b = new TGroupButton("DRAW","cont2","AddOption(\"cont2\")",.5,y-yb,.67,y);
      b->Draw();
   b = new TGroupButton("DRAW","cont3","AddOption(\"cont3\")",.7,y-yb,.87,y);
      b->Draw();

   // Quiet/Verbose buttons
   y -= dy;
   b = new TGroupButton("SYSTEM","Cartesian","",x0,y-yb,.45,y);
      b->SetBorderMode(-1);
      b->SetFillColor(15);
      b->Draw();
   b = new TGroupButton("SYSTEM","Polar","AddOption(\"pol\")",.55,y-yb,.9,y);
      b->SetFillColor(15);
      b->Draw();
   y -= dy;
   b = new TGroupButton("SYSTEM","Spheric","AddOption(\"sph\")",x0,y-yb,.45,y);
      b->SetFillColor(15);
      b->Draw();
   b = new TGroupButton("SYSTEM","Cylindric","AddOption(\"cyl\")",.55,y-yb,.9,y);
      b->SetFillColor(15);
      b->Draw();

   // Other buttons
   y -= dy;
   b = new TGroupButton("ERRORS","E1: errors/edges","AddOption(\"E1\")",x0,y-yb,.48,y);
      b->SetFillColor(41);
      b->Draw();
   b = new TGroupButton("ERRORS","E2: errors/rectangles","AddOption(\"E2\")",.52,y-yb,.9,y);
      b->SetFillColor(41);
      b->Draw();
   y -= dy;
   b = new TGroupButton("ERRORS","E3: errors/fill","AddOption(\"E3\")",x0,y-yb,.48,y);
      b->SetFillColor(41);
      b->Draw();
   b = new TGroupButton("ERRORS","E4: errors/contour","AddOption(\"E4\")",.52,y-yb,.9,y);
      b->SetFillColor(41);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW1","L : Lines","AddOption(\"L\")",x0,y-yb,.48,y);
      b->SetFillColor(31);
      b->Draw();
   b = new TGroupButton("DRAW2","P : markers","AddOption(\"P\")",.52,y-yb,.9,y);
      b->SetFillColor(31);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","ARR : arrows","AddOption(\"ARR\")",x0,y-yb,.48,y);
      b->SetFillColor(31);
      b->Draw();
   b = new TGroupButton("DRAW","BOX : boxes","AddOption(\"BOX\")",.52,y-yb,.9,y);
      b->SetFillColor(31);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","COL: colors","AddOption(\"COL\")",x0,y-yb,.48,y);
      b->SetFillColor(31);
      b->Draw();
   b = new TGroupButton("DRAW","TEXT : values","AddOption(\"TEXT\")",.52,y-yb,.9,y);
      b->SetFillColor(31);
      b->Draw();
   y -= dy;
   b = new TGroupButton("SAME","Same Picture","SetSame()",x0,y-yb,.48,y);
      b->SetFillColor(43);
      b->Draw();

   char cmd[64];
   if (fHistogram) {
      sprintf(cmd,"drawpanel: %s",fHistogram->GetName());
      SetTitle(cmd);
   }

   Modified(kTRUE);
   Update();
   SetEditable(kFALSE);

   //add this TDrawPanelHist to the list of cleanups such that in case
   //the referenced object is deleted, its pointer be reset
   gROOT->GetListOfCleanups()->Add(this);
   
   fRefPad->cd();
}

//______________________________________________________________________________
TDrawPanelHist::~TDrawPanelHist()
{
   // DrawPanelHist destructor.
   gROOT->GetListOfCleanups()->Remove(this);

}

//______________________________________________________________________________
void TDrawPanelHist::AddOption(Option_t *option)
{
   // Add option to the current list of options.

   fOption += option;
}

//______________________________________________________________________________
void TDrawPanelHist::Apply(const char *action)
{
   // Collect all options and draw histogram.

   if (!fHistogram) return;
   if (!fRefPad) return;
   fRefPad->cd();

   SetCursor(kWatch);

   if (!strcmp(action,"Defaults")) {
      SetDefaults();
      SetCursor(kCross);
      return;
   }

   // take into account slider to set the histogram range;
   TObject *obj;
   TGroupButton *button;
   TIter next(fPrimitives);

   while ((obj = next())) {
      if (obj->InheritsFrom(TGroupButton::Class())) {
         button = (TGroupButton*)obj;
         if (button->GetBorderMode() < 0) button->ExecuteAction();
      }
   }

   TH1 *h1;
   if (fHistogram->InheritsFrom("TF1")) {
      h1 = (TH1*)((TF1*)fHistogram)->GetHistogram();
   } else if (fHistogram->InheritsFrom("TH1")) {
      h1 = (TH1*)fHistogram;
   } else {
      h1 = 0;
   }
   if (h1) {
      Int_t nbins   = h1->GetXaxis()->GetNbins();
      Int_t first   = 1 + Int_t(nbins*fSlider->GetMinimum());
      Int_t last    =     Int_t(nbins*fSlider->GetMaximum());
      h1->GetXaxis()->SetRange(first,last);
   }
   Int_t keep = fHistogram->TestBit(kCanDelete);
   fHistogram->SetBit(kCanDelete,0);
   fHistogram->Draw((char*)fOption.Data());
   fHistogram->SetBit(kCanDelete,keep);
   fOption   = "";
   fRefPad->Update();
   SetCursor(kCross);
}

//______________________________________________________________________________
void TDrawPanelHist::BuildStandardButtons()
{
   // Create Draw, Defaults and Close buttons.

   TGroupButton *b = new TGroupButton("APPLY","Draw","",.05,.01,.3,.09);
   b->SetTextSize(0.55);
   b->SetBorderSize(3);
   b->SetFillColor(44);
   b->Draw();

   b = new TGroupButton("APPLY","Defaults","",.375,.01,.625,.09);
   b->SetTextSize(0.55);
   b->SetBorderSize(3);
   b->SetFillColor(44);
   b->Draw();

   b = new TGroupButton("APPLY","Close","",.70,.01,.95,.09);
   b->SetTextSize(0.55);
   b->SetBorderSize(3);
   b->SetFillColor(44);
   b->Draw();
}

//______________________________________________________________________________
void TDrawPanelHist::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Control mousse events when slider is used in a drawpanel
   //
   // This function is called by TPad::ExecuteEvent or TSliderBox::ExecuteEvent
   // We return in the first case.
   // When called by the slider,  px = 0 and py = 0

   //if (event == kMouseLeave || event == kMouseEnter || (px && py)) {
   if (px && py) {
      SetCursor(kCross);
      return;
   }

   Float_t xpmin = fSlider->GetMinimum();
   Float_t xpmax = fSlider->GetMaximum();

   static TH1 *h1;
   static Bool_t done = kFALSE;
   static Int_t px1,py1,px2,py2,nbins;
   static Int_t pxmin,pxmax;
   static Float_t xmin,xmax,ymin,ymax,xleft,xright;
   Int_t first,last;

   if (!fRefPad) return;
   fRefPad->cd();

   switch (event) {

   case kButton1Down:
      fRefPad->GetCanvas()->FeedbackMode(kTRUE);
      gVirtualX->SetLineWidth(2);
      gVirtualX->SetLineColor(-1);
      if (done) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      done = kTRUE;
      if (fHistogram->InheritsFrom("TF1")) {
         h1 = (TH1*)((TF1*)fHistogram)->GetHistogram();
      } else if (fHistogram->InheritsFrom("TH1")) {
         h1 = (TH1*)fHistogram;
      } else {
         h1 = 0;
         break;
      }
      nbins = h1->GetXaxis()->GetNbins();
      xmin  = fRefPad->GetUxmin();
      xmax  = fRefPad->GetUxmax();
      xleft = xmin+(xmax-xmin)*xpmin;
      xright= xmin+(xmax-xmin)*xpmax;
      ymin  = fRefPad->GetUymin();
      ymax  = fRefPad->GetUymax();
      px1   = fRefPad->XtoAbsPixel(xleft);
      py1   = fRefPad->YtoAbsPixel(ymin);
      px2   = fRefPad->XtoAbsPixel(xright);
      py2   = fRefPad->YtoAbsPixel(ymax);
      pxmin = fRefPad->XtoAbsPixel(xmin);
      pxmax = fRefPad->XtoAbsPixel(xmax);
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      break;

   case kMouseMotion:
      break;

   case kButton1Motion:
      if (h1 == 0) break;
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      first   = 1 + Int_t(nbins*xpmin);
      last    =     Int_t(nbins*xpmax);
      xleft  = fRefPad->XtoPad(h1->GetXaxis()->GetBinLowEdge(first));
      xright = fRefPad->XtoPad(h1->GetXaxis()->GetBinLowEdge(last)+h1->GetXaxis()->GetBinWidth(last));
      px1 = fRefPad->XtoAbsPixel(xleft);
      px2 = fRefPad->XtoAbsPixel(xright);
      if (px1 < pxmin) px1 = pxmin;
      if (px2 > pxmax) px2 = pxmax;
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);

      break;

   case kButton1Up:
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      done = kFALSE;
      fRefPad->GetCanvas()->FeedbackMode(kFALSE);
      gVirtualX->SetLineWidth(-1);

      break;
   }
}

//______________________________________________________________________________
void TDrawPanelHist::RecursiveRemove(TObject *obj)
{
//  when obj is deleted, clear fHistogram if fHistogram=obj
   
   TDialogCanvas::RecursiveRemove(obj);
   if (obj == fHistogram) fHistogram = 0;
}

//______________________________________________________________________________
void TDrawPanelHist::SavePrimitive(ofstream &, Option_t *)
{
   // Save this drawpanel in a macro.

}

//______________________________________________________________________________
void TDrawPanelHist::SetDefaults()
{
   // Set default draw panel options.

   fOption    = "";
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   fRefObject = this;
   fHistogram = gROOT->GetSelectedPrimitive();

   TIter next(fPrimitives);
   TObject *obj;
   TGroupButton *button;
   while ((obj = next())) {
      if (obj == this) continue;
      if (obj->InheritsFrom(TGroupButton::Class())) {
         button = (TGroupButton*)obj;
         if (button->GetBorderMode() < 0) {
            button->SetBorderMode(1);
            button->Modified();
         }
      }
   }
   fSlider->SetRange(0,1);

   char cmd[64];
   if (fHistogram) {
      sprintf(cmd,"drawpanel: %s",fHistogram->GetName());
      SetTitle(cmd);
   }

   Modified();
   Update();
}

//______________________________________________________________________________
void TDrawPanelHist::SetSame()
{
   // Set graphics option "same".

   AddOption("same");
}
