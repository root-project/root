// @(#)root/gpad:$Name$:$Id$
// Author: Rene Brun   04/07/96
// ---------------------------------- AttFillCanvas.C

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TROOT.h"
#include "TAttFillCanvas.h"
#include "TGroupButton.h"
#include "TLine.h"
#include "TText.h"

ClassImp(TAttFillCanvas)

//______________________________________________________________________________
//
//   An AttFillCanvas is a TDialogCanvas specialized to set fill attributes.
//Begin_Html
/*
<img src="gif/attfillcanvas.gif">
*/
//End_Html
//

//______________________________________________________________________________
TAttFillCanvas::TAttFillCanvas() : TDialogCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*-*AttFillCanvas default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

}

//_____________________________________________________________________________
TAttFillCanvas::TAttFillCanvas(const char *name, const char *title, UInt_t ww, UInt_t wh)
             : TDialogCanvas(name,title,ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*AttFillCanvas constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   static Int_t lsty[10] = {1001,3004,3005,3006,3007,0,3013,3010,3014,3012};
   TVirtualPad *padsav = gPad;

   BuildStandardButtons();

//*-*- Fill styles choice buttons
   TGroupButton *test1 = 0;
   Float_t xlow, ylow, wpad, hpad;
   Int_t i,j;
   wpad = 0.19;
   hpad = 0.20;
   char command[64];
   Int_t number = 0;
   for (j=0;j<2;j++) {
      ylow = 0.12 + j*hpad;
      for (i=0;i<5;i++) {
         xlow = 0.05 + i*wpad;
         sprintf(command,"SetFillStyle(%d)",lsty[number]);
         test1 = new TGroupButton("Style","",command,xlow, ylow, xlow+0.9*wpad, ylow+0.9*hpad);
         if (number == 0) {
            test1->SetBorderMode(-1);
            test1->SetFillColor(1);
         } else {
            test1->SetFillColor(10);
         }
         if (number == 5) {
            test1->SetFillStyle(1001);
         } else {
            test1->SetFillStyle(lsty[number]);
         }
         test1->SetBorderSize(2);
         test1->Draw();
         number++;
      }
   }

//*-* draw colortable pads
   test1->DisplayColorTable("SetFillColor",0.05, 0.60, 0.90, 0.38);
   Modified(kTRUE);
   Update();

   padsav->cd();
}

//______________________________________________________________________________
TAttFillCanvas::~TAttFillCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*AttFillCanvas default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
}

//______________________________________________________________________________
void TAttFillCanvas::UpdateFillAttributes(Int_t col, Int_t sty)
{
//*-*-*-*-*-*-*-*-*-*-*Update object attributes*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TIter next(GetListOfPrimitives());
   TGroupButton *button;
   char cmd[64];
   fRefObject = gROOT->GetSelectedPrimitive();
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   if (fRefObject) {
      sprintf(cmd,"attfill: %s",fRefObject->GetName());
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
      sprintf(cmd,"SetFillColor(%d)",col);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      sprintf(cmd,"SetFillStyle(%d)",sty);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      if (button->GetBorderMode() < 0) {
         button->Modified();
      }
   }
   Update();
}
