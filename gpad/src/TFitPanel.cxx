// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   24/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TFitPanel.h"
#include "TGroupButton.h"
#include "TSlider.h"
#include "TText.h"
#include "TH1.h"
#include "TF1.h"

#include <stdio.h>

ClassImp(TFitPanel)

//______________________________________________________________________________
//
//   A FitPanel is a TDialogCanvas specialized to control histogram fits.
//   With the mouse, the user can control:
//     - the type of function to be fitted
//     - the various fit options
//     - the drawing options
//   When the FIT button is executed, the selected histogram is fitted
//   with the current parameters.
//
//   One can select a range of the histogram to be fitted via the slider.
//
//   The options are documented in TH1::Fit.
//Begin_Html
/*
<img src="gif/fitpanel.gif">
*/
//End_Html
//

//______________________________________________________________________________
TFitPanel::TFitPanel() : TDialogCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*-*FitPanel default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============================

}

//_____________________________________________________________________________
TFitPanel::TFitPanel(const char *name, const char *title, UInt_t ww, UInt_t wh)
          : TDialogCanvas(name, title,ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*FitPanel constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   TGroupButton *b;
   fOption    = "r";
   fFunction  = "";
   fSame      = "";
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   fRefObject = this;
   fObjectFit = gROOT->GetSelectedPrimitive();

   BuildStandardButtons();

// Range Slider
   fSlider = new TSlider("slider","X slider",.05,.11,.95,.17);
   fSlider->SetObject(this);

// Standard functions
   Int_t nlines = 10;
   Float_t x0 = 0.1;
   Float_t y  = 0.98;
   Float_t dy = (y-0.18)/nlines;
   Float_t yb = 0.7*dy;
   Float_t dpx = (0.9-x0)/5;
   Int_t i;
   char text[25];
   char tfunc[10];
   for (i=0;i<5;i++) {
      sprintf(text,"SetFunction(\"pol%d\")",i);
      sprintf(tfunc,"pol%d",i);
      b = new TGroupButton("FUNCTION",tfunc,text,x0+i*dpx,y-yb,x0+i*dpx+0.9*dpx,y);
      b->Draw();
   }

   y -= dy;
   for (i=0;i<5;i++) {
      sprintf(text,"SetFunction(\"pol%d\")",i+5);
      sprintf(tfunc,"pol%d",i+5);
      b = new TGroupButton("FUNCTION",tfunc,text,x0+i*dpx,y-yb,x0+i*dpx+0.9*dpx,y);
      b->Draw();
   }
   y  -= dy;
   dpx = 0.25;
   b = new TGroupButton("FUNCTION","gaus","SetFunction(\"gaus\")",.1,y-yb,.28,y);
      b->SetBorderMode(-1);
      b->Draw();
   b = new TGroupButton("FUNCTION","landau","SetFunction(\"landau\")",.30,y-yb,.48,y);
      b->Draw();
   b = new TGroupButton("FUNCTION","expo","SetFunction(\"expo\")",.52,y-yb,.7,y);
      b->Draw();
   b = new TGroupButton("FUNCTION","user","SetFunction(\"user\")",.72,y-yb,.9,y);
      b->Draw();

// Quiet/Verbose buttons
   y -= dy;
   b = new TGroupButton("MODE","Quiet","AddOption(\"Q\")",x0,y-yb,.32,y);
      b->SetFillColor(15);
      b->Draw();
   b = new TGroupButton("MODE","Verbose","AddOption(\"V\")",.34,y-yb,.56,y);
      b->SetFillColor(15);
      b->Draw();
   b = new TGroupButton("SAME","Same Picture","SetSame()",.58,y-yb,.9,y);
      b->SetFillColor(43);
      b->Draw();

// Other buttons
   y -= dy;
   b = new TGroupButton("WEIGHTS","W: Set all weights to 1","AddOption(\"W\")",x0,y-yb,.9,y);
      b->SetFillColor(33);
      b->Draw();
   y -= dy;
   b = new TGroupButton("ERRORS","E: Compute best errors","AddOption(\"E\")",x0,y-yb,.9,y);
      b->SetFillColor(41);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","+ : Add to list of functions","AddOption(\"+\")",x0,y-yb,.9,y);
      b->SetFillColor(31);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","N : Do not store/draw function","AddOption(\"N\")",x0,y-yb,.9,y);
      b->SetFillColor(31);
      b->Draw();
   y -= dy;
   b = new TGroupButton("DRAW","0 : Do not draw function","AddOption(\"0\")",x0,y-yb,.9,y);
      b->SetFillColor(31);
      b->Draw();
   y -= dy;
   b = new TGroupButton("FITM","L : Log  Likelihood","AddOption(\"L\")",x0,y-yb,.9,y);
      b->SetFillColor(43);
      b->Draw();

   if (!gROOT->GetFunction("gaus")) {
      Float_t xmin = 1.;
      Float_t xmax = 2.;
      new TF1("gaus","gaus",xmin,xmax);
      new TF1("landau","landau",xmin,xmax);
      new TF1("expo","expo",xmin,xmax);
      for (i=0;i<10;i++) new TF1(Form("pol%d",i),Form("pol%d",i),xmin,xmax);
   }

   char cmd[64];
   if (fObjectFit) {
      sprintf(cmd,"%s: %s",GetName(),fObjectFit->GetName());
      SetTitle(cmd);
   }

   Modified(kTRUE);
   Update();

   fRefPad->cd();
}

//______________________________________________________________________________
TFitPanel::~TFitPanel()
{
//*-*-*-*-*-*-*-*-*-*-*FitPanel default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
}

//______________________________________________________________________________
void TFitPanel::AddOption(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Add option to the current list of options*-*-*-*-*-*-*
//*-*                  =========================================

   fOption += option;
}

//______________________________________________________________________________
void TFitPanel::Apply(const char *action)
{
//*-*-*-*-*-*-*-*-*-*Collect all options and fit histogram*-*-*-*-*-*-*
//*-*                =====================================

   if (!fRefPad) return;
   fRefPad->cd();

   SetCursor(kWatch);

   if (!strcmp(action,"Defaults")) {
      SetDefaults();
      SetCursor(kCross);
      return;
   }
// take into account slider to set the function range;
   TObject *obj;
   TGroupButton *button;
   TIter next(fPrimitives);

   while ((obj = next())) {
      if (obj->InheritsFrom(TGroupButton::Class())) {
         button = (TGroupButton*)obj;
         if (button->GetBorderMode() < 0) button->ExecuteAction();
      }
   }

   TF1 *f1 = (TF1*)gROOT->GetFunction(fFunction.Data());
   if (!f1) return;

   Float_t xhmin = fRefPad->GetUxmin();
   Float_t xhmax = fRefPad->GetUxmax();
   Float_t xmin  = xhmin + (xhmax-xhmin)*fSlider->GetMinimum();
   Float_t xmax  = xhmin + (xhmax-xhmin)*fSlider->GetMaximum();
   f1->SetRange(xmin,xmax);

// Warning below. In case object is not a TH1, TH2,etc, cannot execute block.
   if (fObjectFit->InheritsFrom(TH1::Class())) {
      TH1 *h1 = (TH1*)fObjectFit;
      h1->Fit((char*)fFunction.Data(), (char*)fOption.Data(), (char*)fSame.Data());
   }
   fOption   = "r";
   fFunction = "gaus";
   fSame     = "";
}

//______________________________________________________________________________
void TFitPanel::BuildStandardButtons()
{
//*-*-*-*-*-*-*-*-*Create FIT, Defaults and CLOSE buttons*-*-*-*-*-*-*-*-*-*-*
//*-*              ======================================

   TGroupButton *b = new TGroupButton("APPLY","Fit","",.05,.01,.3,.09);
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
void TFitPanel::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*Control mouse events when slider is used in a fitpanel
//*-*                ======================================================
//
//    This function is called by TPad::ExecuteEvent or TSliderBox::ExecuteEvent
//    We return in the first case.
//    When called by the slider,  px = 0 and py = 0

   if (px && py) {
      SetCursor(kCross);
      return;
   }

   Float_t xpmin = fSlider->GetMinimum();
   Float_t xpmax = fSlider->GetMaximum();

   static Bool_t done = kFALSE;
   static Int_t px1,py1,px2,py2;
   static Float_t xmin,xmax,ymin,ymax;

   if (!fRefPad) return;
   fRefPad->cd();
   fRefPad->GetCanvas()->FeedbackMode(kTRUE);
   gVirtualX->SetLineWidth(2);

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      if (done) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      done = kTRUE;
      xmin = fRefPad->GetUxmin();
      xmax = fRefPad->GetUxmax();
      ymin = fRefPad->GetUymin();
      ymax = fRefPad->GetUymax();
      px1  = fRefPad->XtoAbsPixel(xmin+(xmax-xmin)*xpmin);
      py1  = fRefPad->YtoAbsPixel(ymin);
      px2  = fRefPad->XtoAbsPixel(xmin+(xmax-xmin)*xpmax);
      py2  = fRefPad->YtoAbsPixel(ymax);
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      break;

   case kMouseMotion:
      break;

   case kButton1Motion:
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      px1 = fRefPad->XtoAbsPixel(xmin+(xmax-xmin)*xpmin);
      px2 = fRefPad->XtoAbsPixel(xmin+(xmax-xmin)*xpmax);
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
void TFitPanel::SavePrimitive(ofstream &, Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Save this fitpanel in a macro*-*-*-*-*-*-*
//*-*                  =============================
}

//______________________________________________________________________________
void TFitPanel::SetDefaults()
{
//*-*-*-*-*-*-*-*-*-*Set default fit panel options*-*-*-*-*-*-*
//*-*                =============================

   fOption    = "r";
   fFunction  = "";
   fSame      = "";
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   fRefObject = this;
   fObjectFit = gROOT->GetSelectedPrimitive();

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
   if (fObjectFit) {
      sprintf(cmd,"%s: %s",GetName(),fObjectFit->GetName());
      SetTitle(cmd);
   }
   Modified();
   Update();
}

//______________________________________________________________________________
void TFitPanel::SetFunction(const char *function)
{
//*-*-*-*-*-*-*-*-*-*Set the function to be used in the fit*-*-*-*-*-*-*
//*-*                ======================================

   fFunction = function;
}

//______________________________________________________________________________
void TFitPanel::SetSame()
{
//*-*-*-*-*-*-*-*-*-*Set graphics option "same"*-*-*-*-*-*-*
//*-*                =========================

   fSame = "same";
}
