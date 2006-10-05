// @(#)root/gpad:$Name:  $:$Id: TUtilPad.cxx,v 1.5 2005/02/04 13:07:16 brun Exp $
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
#include "TEnv.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TFitPanelGraph.h"
#include "TDrawPanelHist.h"
#include "TInspectCanvas.h"
#include "TVirtualPadEditor.h"
#include "TPluginManager.h"

Int_t TUtilPad::fgPanelVersion = 0;

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
void TUtilPad::DrawPanel(const TVirtualPad *pad, const TObject *obj)
{
// interface to the TDrawPanelHist
   
   const char *editor = gEnv->GetValue("Plugin.TVirtualPadEditor","");

   if (fgPanelVersion == 0 && strstr(editor,"TGedEditor")) {
      //new interface by Carsten Hof
      //gROOT->ProcessLine(Form("TVirtualPadEditor::ShowEditor();"));
      TVirtualPadEditor *editor = TVirtualPadEditor::GetPadEditor();
      editor->Show();
      gROOT->ProcessLine(Form("((TCanvas*)0x%x)->Selected((TVirtualPad*)0x%x,(TObject*)0x%x,1)",pad->GetCanvas(),pad,obj));
      return;
   }
   
   // old Drawpanel (default)
   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TDrawPanelHist *R__drawpanelhist = (TDrawPanelHist*)lc->FindObject("R__drawpanelhist");
   if (!R__drawpanelhist) {
      new TDrawPanelHist("R__drawpanelhist","Hist Draw Panel",330,450,pad,obj);
      return; 
   }
   R__drawpanelhist->SetDefaults(); 
   R__drawpanelhist->Show();
}

//______________________________________________________________________________
void TUtilPad::FitPanel(const TVirtualPad *pad, const TObject *obj)
{
// interface to the TFitPanel
   
   if (fgPanelVersion == 0) {

      // new interface (default)
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TFitEditor"))) {
      if (h->LoadPlugin() == -1)
         return;
      h->ExecPlugin(2, pad, obj);
      }
   } else {
   
      // old FitPanel - use TUtilPad::SetPanelVersion(1)
      TList *lc = (TList*)gROOT->GetListOfCanvases();
      TFitPanel *R__fitpanel = (TFitPanel*)lc->FindObject("R__fitpanel");
      if (!R__fitpanel) {
         new TFitPanel("R__fitpanel","Fit Panel",300,400,pad,obj);
         return;
      }
      R__fitpanel->SetDefaults();
      R__fitpanel->Show();
   }
}

//______________________________________________________________________________
void TUtilPad::FitPanelGraph(const TVirtualPad *pad, const TObject *obj)
{
// interface to the TFitPanelGraph
   
   if (fgPanelVersion == 0) {
      TPluginHandler *h;
      h = gROOT->GetPluginManager()->FindHandler("TFitEditor");
      if (h->LoadPlugin() == -1)
         return;
      h->ExecPlugin(2, pad, obj);
   } else {

   // old FitPanel - use TUtilPad::SetPanelVersion(1)
      TList *lc = (TList*)gROOT->GetListOfCanvases();
      TFitPanelGraph *R__fitpanel = (TFitPanelGraph*)lc->FindObject("R__fitpanelgraph");
      if (!R__fitpanel) { 
         new TFitPanelGraph("R__fitpanelgraph","Fit Panel",300,400,pad,obj);
         return;
      }
      R__fitpanel->SetDefaults(); 
      R__fitpanel->Show();
   }
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

//______________________________________________________________________________
void TUtilPad::SetPanelVersion(Int_t version)
{
// static function to set teh DrawPanel version
//   version = 0  (default) old DrawPanel
//   version = 1  new prototype from Marek Biskup
   
   fgPanelVersion = version;
}
