// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   28/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TFitPanelGraph.h"
#include "TGroupButton.h"
#include "TSlider.h"
#include "TText.h"
#include "TGraph.h"
#include "TF1.h"

#include <stdio.h>

ClassImp(TFitPanelGraph)

//______________________________________________________________________________
//
//   A TFitPanelGraph is a TFitPanel specialized to control graph fits.
//   With the mouse, the user can control:
//     - the type of function to be fitted
//     - the various fit options
//     - the drawing options
//   When the FIT button is executed, the selected histogram is fitted
//   with the current parameters.
//   The options are documented in TGraph::Fit.
//Begin_Html
/*
<img src="gif/fitpanel.gif">
*/
//End_Html
//

//______________________________________________________________________________
TFitPanelGraph::TFitPanelGraph() : TFitPanel()
{
//*-*-*-*-*-*-*-*-*-*-*-*FitPanelGraph default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ============================

}

//_____________________________________________________________________________
TFitPanelGraph::TFitPanelGraph(const char *name, const char *title, UInt_t ww, UInt_t wh)
          : TFitPanel(name, title,ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*FitPanelGraph constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================
}

//______________________________________________________________________________
TFitPanelGraph::~TFitPanelGraph()
{
//*-*-*-*-*-*-*-*-*-*-*FitPanelGraph default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
}

//______________________________________________________________________________
void TFitPanelGraph::Apply(const char *action)
{
//*-*-*-*-*-*-*-*-*-*Collect all options and fit histogram*-*-*-*-*-*-*
//*-*                =====================================

   if (!fRefPad) return;
   fRefPad->cd();

   SetCursor(kWatch);

   if (!strcmp(action,"Defaults")) {
      SetDefaults();
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

   TGraph *gr    = (TGraph*)fObjectFit;
   Int_t npoints = gr->GetN();
   Float_t *gx   = gr->GetX();

   TF1 *f1 = (TF1*)gROOT->GetFunction(fFunction.Data());
   if (!f1) return;
   Float_t xgrmin = gx[0];
   Float_t xgrmax = gx[0];
   for (Int_t i=0;i<npoints;i++) {
      if (gx[i] < xgrmin) xgrmin = gx[i];
      if (gx[i] > xgrmax) xgrmax = gx[i];
   }
   Float_t smin  = fSlider->GetMinimum();
   Float_t smax  = fSlider->GetMaximum();
   Float_t xpmin = fRefPad->GetUxmin();
   Float_t xpmax = fRefPad->GetUxmax();
   Float_t xmin  = xpmin + (xpmax-xpmin)*smin;
   Float_t xmax  = xpmin + (xpmax-xpmin)*smax;
   if (smin <= 0) xmin = xgrmin;
   if (smax >= 1) xmax = xgrmax;
   f1->SetRange(xmin,xmax);

   gr->Fit((char*)fFunction.Data(), (char*)fOption.Data(), (char*)fSame.Data());
   fOption   = "r";
   fFunction = "gaus";
   fSame     = "";
   fRefPad->Modified();
   fRefPad->Update();
}

//______________________________________________________________________________
void TFitPanelGraph::SavePrimitive(ofstream &, Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Save this fitpanelgraph in a macro*-*-*-*-*-*-*
//*-*                  ==================================
}
