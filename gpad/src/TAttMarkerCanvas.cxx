// @(#)root/gpad:$Name:  $:$Id: TAttMarkerCanvas.cxx,v 1.1.1.1 2000/05/16 17:00:41 rdm Exp $
// Author: Rene Brun   04/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TAttMarkerCanvas.h"
#include "TGroupButton.h"
#include "TMarker.h"
#include "TText.h"
#include "TVirtualPad.h"

ClassImp(TAttMarkerCanvas)

//______________________________________________________________________________
//
//   An AttMarkerCanvas is a TDialogCanvas specialized to set line attributes.
//Begin_Html
/*
<img src="gif/attmarkercanvas.gif">
*/
//End_Html
//

//______________________________________________________________________________
TAttMarkerCanvas::TAttMarkerCanvas() : TDialogCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*-*AttMarkerCanvas default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

}

//_____________________________________________________________________________
TAttMarkerCanvas::TAttMarkerCanvas(const char *name, const char *title, UInt_t ww, UInt_t wh)
                : TDialogCanvas(name,title,ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*AttMarkerCanvas constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   TVirtualPad *padsav = gPad;

   BuildStandardButtons();

//*-*- Marker styles choice buttons
   TGroupButton *test1 = 0;
   TMarker *mark;
   Float_t xlow, ylow, wpad, hpad;
   Int_t i,j;
   xlow = 0.05;
   wpad = 0.19;
   hpad = 0.062;
   char command[64];
   static Int_t markers[15] = { 1, 2, 3, 8, 5,
                               21,22,23,24,25,
                               26,27,28,29,30};
   Int_t number = 0;
   for (j=0;j<3;j++) {
      ylow = 0.34 + j*hpad;
      for (i=0;i<5;i++) {
         number++;
         xlow = 0.05 + i*wpad;
         sprintf(command,"SetMarkerStyle(%d)",markers[number-1]);
         test1 = new TGroupButton("Style","",command,xlow, ylow, xlow+0.9*wpad, ylow+0.9*hpad);
         test1->SetEditable(kTRUE);
         if (number == 1) test1->SetBorderMode(-1);
         test1->SetFillColor(18);
         test1->SetBorderSize(2);
         test1->Draw();
         test1->cd();
         mark = new TMarker(0.5, 0.5,markers[number-1]);
         mark->SetMarkerSize(2);
         mark->Draw();
         test1->SetEditable(kFALSE);
         cd();
      }
   }

//*-*-  Marker Size choice buttons
   wpad = 0.19;
   hpad = 0.065;
   Float_t sizem = 0.1;
   for (j=0;j<3;j++) {
      ylow = 0.11 + j*hpad;
      for (i=0;i<5;i++) {
         sizem += 0.2;
         xlow = 0.05 + i*wpad;
         if (!i && !j) sizem = 1;
         sprintf(command,"SetMarkerSize(%f)",sizem);
         test1 = new TGroupButton("Size","",command,xlow, ylow, xlow+0.9*wpad, ylow+0.9*hpad);
         test1->SetEditable(kTRUE);
         if (!i && !j) test1->SetBorderMode(-1);
         test1->SetFillColor(18);
         test1->SetBorderSize(2);
         test1->Draw();
         test1->cd();
         mark = new TMarker(0.5, 0.5, 24);
         mark->SetMarkerSize(sizem);
         mark->Draw();
         if (!i && !j) sizem = 0.1;
         test1->SetEditable(kFALSE);
         cd();
      }
   }

//*-* draw colortable pads
   test1->DisplayColorTable("SetMarkerColor",0.05, 0.60, 0.90, 0.38);
   Update();
   SetEditable(kFALSE);

   padsav->cd();
}

//______________________________________________________________________________
TAttMarkerCanvas::~TAttMarkerCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*AttMarkerCanvas default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
}

//______________________________________________________________________________
void TAttMarkerCanvas::UpdateMarkerAttributes(Int_t col,Int_t sty,Float_t msiz)
{
//*-*-*-*-*-*-*-*-*-*-*Update marker attributes*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TIter next(GetListOfPrimitives());
   TGroupButton *button;
   char cmd[64];
   fRefObject = gROOT->GetSelectedPrimitive();
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   if (fRefObject) {
      sprintf(cmd,"attmarker: %s",fRefObject->GetName());
      SetTitle(cmd);
   }
   TObject *obj;
   while ((obj = next())) {
      if (!obj->InheritsFrom(TGroupButton::Class())) continue;
      button = (TGroupButton*)obj;
      if (button->GetBorderMode() < 0) {
         button->SetBorderMode(1);
         button->Modified();
      }
      sprintf(cmd,"SetMarkerColor(%d)",col);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      sprintf(cmd,"SetMarkerStyle(%d)",sty);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      sprintf(cmd,"SetMarkerSize(%f)",msiz);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      if (button->GetBorderMode() < 0) {
         button->Modified();
      }
   }
   Update();
}
