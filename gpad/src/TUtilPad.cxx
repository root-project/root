// @(#)root/gpad:$Name:  $:$Id: TVirtualUtilPad.cxx,v 1.1 2002/09/15 10:16:44 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// misc. pad/canvas  utilities                                          //
//                                                                      //
// The functions in this class are called via the TPluginManager.       //
// see TVirtualUtilPad.h for more information .                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TUtilPad.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TFitPanelGraph.h"
#include "TDrawPanelHist.h"
#include "TInspectCanvas.h"

ClassImp(TUtilPad)

//______________________________________________________________________________
TUtilPad::TUtilPad() : TVirtualUtilPad()
{
// note that this object is automatically added to the gROOT list of specials
// in the TVirtualUtilPad constructor.
}

//______________________________________________________________________________
TUtilPad::~TUtilPad()
{
}

//______________________________________________________________________________
void TUtilPad::DrawPanel()
{
// interface to the TDrawPanelHist
   
   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TDrawPanelHist *R__drawpanelhist = (TDrawPanelHist*)lc->FindObject("R__drawpanelhist");
   if (!R__drawpanelhist) {
      new TDrawPanelHist("R__drawpanelhist","Hist Draw Panel",330,450);
      return; 
   }
   R__drawpanelhist->SetDefaults(); 
   R__drawpanelhist->Show();
}

//______________________________________________________________________________
void TUtilPad::FitPanel()
{
// interface to the TFitPanel
   
   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TFitPanel *R__fitpanel = (TFitPanel*)lc->FindObject("R__fitpanel");
   if (!R__fitpanel) {
      new TFitPanel("R__fitpanel","Fit Panel",300,400);
      return;
   }
   R__fitpanel->SetDefaults();
   R__fitpanel->Show();
}

//______________________________________________________________________________
void TUtilPad::FitPanelGraph()
{
// interface to the TFitPanelGraph
   
   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TFitPanelGraph *R__fitpanel = (TFitPanelGraph*)lc->FindObject("R__fitpanelgraph");
   if (!R__fitpanel) { 
      new TFitPanelGraph("R__fitpanelgraph","Fit Panel",300,400);
      return;
   }
   R__fitpanel->SetDefaults(); 
   R__fitpanel->Show();
}

//______________________________________________________________________________
void TUtilPad::InspectCanvas(const TObject *obj)
{
// interface to the object inspector
   
   TInspectCanvas::Inspector((TObject*)obj);
}

//______________________________________________________________________________
void TUtilPad::MakeCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, Int_t ww, Int_t wh)
{
// to create a general canvas with position and size
   
   new TCanvas(name,title,wtopx,wtopy,ww,wh);
}

//______________________________________________________________________________
void TUtilPad::RemoveObject(TObject *parent, const TObject *obj)
{
// to remove an object (eg a TF1) from the list of functions of parent.
   
   if (!parent->InheritsFrom(TGraph::Class())) return;
   TGraph *gr = (TGraph*)parent;
   gr->GetListOfFunctions()->Remove((TObject*)obj);
}
