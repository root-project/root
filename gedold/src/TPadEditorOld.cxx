// @(#)root/gpad:$Name:  $:$Id: TPadEditorOld.cxx,v 1.0 2003/11/26
// Author: Ilka Antcheva   26/11/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class handle the old pad editor interface                       //
//                                                                      //
// Its functions are called via the TPluginManager.                     //
// see TVirtualPadEditor.h for more information.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>

#include "TPadEditorOld.h"
#include "TROOT.h"
#include "TControlBar.h"
#include "TGroupButton.h"
#include "TAttFillCanvas.h"
#include "TAttLineCanvas.h"
#include "TAttMarkerCanvas.h"
#include "TAttTextCanvas.h"
#include "TEnv.h"


ClassImp(TPadEditorOld)

//______________________________________________________________________________
TPadEditorOld::TPadEditorOld(TCanvas*) : TVirtualPadEditor() 
{
   // Create the old Editor 

   fControlBar = 0;
   TString show = gEnv->GetValue("Canvas.ShowEditor","false");
   if (show == "true") Build();
}

//______________________________________________________________________________
TPadEditorOld::~TPadEditorOld()
{
   // Delete the control bar implementation.

   delete fControlBar;
}

//______________________________________________________________________________
void TPadEditorOld::Build() 
{
   // Create the Editor control bar
   fControlBar = new TControlBar("vertical");
   fControlBar->AddButton("Arc",       "gROOT->SetEditorMode(\"Arc\")",       "Create an arc of circle");
   fControlBar->AddButton("Line",      "gROOT->SetEditorMode(\"Line\")",      "Create a line segment");
   fControlBar->AddButton("Arrow",     "gROOT->SetEditorMode(\"Arrow\")",     "Create an Arrow");
   fControlBar->AddButton("Button",    "gROOT->SetEditorMode(\"Button\")",    "Create a user interface Button");
   fControlBar->AddButton("Diamond",   "gROOT->SetEditorMode(\"Diamond\")",   "Create a diamond");
   fControlBar->AddButton("Ellipse",   "gROOT->SetEditorMode(\"Ellipse\")",   "Create an Ellipse");
   fControlBar->AddButton("Pad",       "gROOT->SetEditorMode(\"Pad\")",       "Create a pad");
   fControlBar->AddButton("Pave",      "gROOT->SetEditorMode(\"Pave\")",      "Create a Pave");
   fControlBar->AddButton("PaveLabel", "gROOT->SetEditorMode(\"PaveLabel\")", "Create a PaveLabel (prompt for label)");
   fControlBar->AddButton("PaveText",  "gROOT->SetEditorMode(\"PaveText\")",  "Create a PaveText");
   fControlBar->AddButton("PavesText", "gROOT->SetEditorMode(\"PavesText\")", "Create a PavesText");
   fControlBar->AddButton("PolyLine",  "gROOT->SetEditorMode(\"PolyLine\")",  "Create a PolyLine (TGraph)");
   fControlBar->AddButton("CurlyLine", "gROOT->SetEditorMode(\"CurlyLine\")", "Create a Curly/WavyLine");
   fControlBar->AddButton("CurlyArc",  "gROOT->SetEditorMode(\"CurlyArc\")",  "Create a Curly/WavyArc");
   fControlBar->AddButton("Text/Latex","gROOT->SetEditorMode(\"Text\")",      "Create a Text/Latex string");
   fControlBar->AddButton("Marker",    "gROOT->SetEditorMode(\"Marker\")",    "Create a marker");
   fControlBar->AddButton("<...Graphical Cut...>", "gROOT->SetEditorMode(\"CutG\")","Create a Graphical Cut");
   fControlBar->Show();
}

//______________________________________________________________________________
void TPadEditorOld::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{

   // Draw line in CurrentPad World coordinates

   Int_t px1 = gPad->XtoPixel(x1);
   Int_t px2 = gPad->XtoPixel(x2);
   Int_t py1 = gPad->YtoPixel(y1);
   Int_t py2 = gPad->YtoPixel(y2);

   gVirtualX->DrawLine(px1, py1, px2, py2);
}

//______________________________________________________________________________
void TPadEditorOld::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{

   // Draw line in CurrentPad NDC coordinates

   Int_t px1 = gPad->UtoPixel(u1);
   Int_t px2 = gPad->UtoPixel(u2);
   Int_t py1 = gPad->VtoPixel(v1);
   Int_t py2 = gPad->VtoPixel(v2);

   gVirtualX->DrawLine(px1, py1, px2, py2);
}

//______________________________________________________________________________
void TPadEditorOld::DrawText(Double_t x, Double_t y, const char *text)
{

   // Draw text in CurrentPad World coordinates

   Int_t px = gPad->XtoPixel(x);
   Int_t py = gPad->YtoPixel(y);

   Float_t angle = gVirtualX->GetTextAngle();
   Float_t mgn   = gVirtualX->GetTextSize();
   gVirtualX->DrawText(px, py, angle, mgn, text, TVirtualX::kClear);
}

//______________________________________________________________________________
void TPadEditorOld::DrawTextNDC(Double_t u, Double_t v, const char *text)
{

   // Draw text in CurrentPad NDC coordinates

   Int_t px = gPad->UtoPixel(u);
   Int_t py = gPad->VtoPixel(v);

   Float_t angle = gVirtualX->GetTextAngle();
   Float_t mgn   = gVirtualX->GetTextSize();
   gVirtualX->DrawText(px, py, angle, mgn, text, TVirtualX::kClear);
}

//______________________________________________________________________________
void TPadEditorOld::FillAttributes(Int_t col, Int_t sty)
{

   // Update fill area attributes via the dialog canvas

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TAttFillCanvas *R__attfill = (TAttFillCanvas*)lc->FindObject("R__attfill");
   if (!R__attfill) {
      R__attfill = new TAttFillCanvas("R__attfill","Fill Attributes",250,400);
   }
   R__attfill->UpdateFillAttributes(col,sty);
   R__attfill->Show();
}


//______________________________________________________________________________
void TPadEditorOld::LineAttributes(Int_t col, Int_t sty, Int_t width)
{

   // Update line attributes via the dialog canvas

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TAttLineCanvas *R__attline = (TAttLineCanvas*)lc->FindObject("R__attline");
   if (!R__attline) {
      R__attline = new TAttLineCanvas("R__attline","Line Attributes",250,400);
   }
   R__attline->UpdateLineAttributes(col,sty,width);
   R__attline->Show();
}


//______________________________________________________________________________
void TPadEditorOld::MarkerAttributes(Int_t col, Int_t sty, Float_t msiz)
{

   // Update marker attributes via the dialog canvas

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TAttMarkerCanvas *R__attmarker = (TAttMarkerCanvas*)lc->FindObject("R__attmarker");
   if (!R__attmarker) {
      R__attmarker = new TAttMarkerCanvas("R__attmarker","Marker Attributes",250,400);
   }
   R__attmarker->UpdateMarkerAttributes(col,sty,msiz);
   R__attmarker->Show();
}


//______________________________________________________________________________
void TPadEditorOld::TextAttributes(Int_t align,Float_t angle,Int_t col,Int_t font,Float_t tsize)
{

   // Update text attributes via the dialog canvas

   TList *lc = (TList*)gROOT->GetListOfCanvases();
   TAttTextCanvas *R__atttext = (TAttTextCanvas*)lc->FindObject("R__atttext");
   if (!R__atttext) {
      R__atttext = new TAttTextCanvas("R__atttext","Text Attributes",400,600);
   }
   R__atttext->UpdateTextAttributes(align,angle,col,font,tsize);
   R__atttext->Show();
}
