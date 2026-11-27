// @(#)root/gpad:$Id$
// Author: Rene Brun   03/07/96

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TDialogCanvas.h"
#include "TGroupButton.h"
#include "TStyle.h"


/** \class TDialogCanvas
\ingroup gpad

A canvas specialized to set attributes.

It contains, in general, TGroupButton objects.
When the APPLY button is executed, the actions corresponding
to the active buttons are executed via the Interpreter.

See examples in TAttLineCanvas, TAttFillCanvas, TAttTextCanvas, TAttMarkerCanvas
*/

////////////////////////////////////////////////////////////////////////////////
/// DialogCanvas default constructor

TDialogCanvas::TDialogCanvas() : TCanvas()
{
}

////////////////////////////////////////////////////////////////////////////////
/// DialogCanvas constructor

TDialogCanvas::TDialogCanvas(const char *name, const char *title, Int_t ww, Int_t wh)
             : TCanvas(name,title,-ww,wh)
{
   SetFillColor(36);
}

////////////////////////////////////////////////////////////////////////////////
/// DialogCanvas constructor

TDialogCanvas::TDialogCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, UInt_t ww, UInt_t wh)
             : TCanvas(name,title,-wtopx,wtopy,ww,wh)
{
   SetFillColor(36);
}

////////////////////////////////////////////////////////////////////////////////
/// DialogCanvas default destructor

TDialogCanvas::~TDialogCanvas()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Called when the APPLY button is executed

void TDialogCanvas::Apply(const char *action)
{
   if (!fRefPad)
      return;
   SetCursor(kWatch);

   TIter next(fPrimitives);
   TObject *refobj = fRefObject;
   if (!strcmp(action,"gStyle"))
      fRefObject = gStyle;

   while (auto obj = next()) {
      if (auto button = dynamic_cast<TGroupButton *>(obj))
         if (button->GetBorderMode() < 0)
            button->ExecuteAction();
   }
   fRefObject = refobj;
   if (gROOT->GetSelectedPad()) {
      gROOT->GetSelectedPad()->Modified();
      gROOT->GetSelectedPad()->Update();
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Create APPLY, gStyle and CLOSE buttons

void TDialogCanvas::BuildStandardButtons()
{
   auto btn1 = new TGroupButton("APPLY","Apply","",.05,.01,.3,.09);
   btn1->SetTextSize(0.55);
   btn1->SetBorderSize(3);
   btn1->SetFillColor(44);
   Add(btn1);

   auto btn2 = new TGroupButton("APPLY","gStyle","",.375,.01,.625,.09);
   btn2->SetTextSize(0.55);
   btn2->SetBorderSize(3);
   btn2->SetFillColor(44);
   Add(btn2);

   auto btn3 = new TGroupButton("APPLY","Close","",.70,.01,.95,.09);
   btn3->SetTextSize(0.55);
   btn3->SetBorderSize(3);
   btn3->SetFillColor(44);
   Add(btn3);
}


////////////////////////////////////////////////////////////////////////////////
/// Set world coordinate system for the pad

void TDialogCanvas::Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   TPad::Range(x1,y1,x2,y2);
}


////////////////////////////////////////////////////////////////////////////////
/// Recursively remove object from a pad and its sub-pads

void TDialogCanvas::RecursiveRemove(TObject *obj)
{
   TPad::RecursiveRemove(obj);
   if (fRefObject == obj) fRefObject = nullptr;
   if (fRefPad    == obj) fRefPad    = nullptr;
}
