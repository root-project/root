// @(#)root/gpad:$Name:  $:$Id: TAttFillCanvas.cxx,v 1.2 2001/05/28 06:20:52 brun Exp $
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

   TVirtualPad *padsav = gPad;

   BuildStandardButtons();

//*-*- Fill styles choice buttons
   TGroupButton *test1 = 0;
   Float_t xlow, ylow;
   Float_t xmin = 0.03;
   Float_t xmax = 1-xmin;
   Float_t ymin = 0.12;
   Float_t ymax = 0.58;
   Float_t wpad = (xmax-xmin)/7;
   Float_t hpad = (ymax-ymin)/4;
   Int_t i,j;
   char command[64];
   Int_t number = 0;
   for (j=0;j<4;j++) {
      ylow = ymin + j*hpad;
      for (i=0;i<7;i++) {
         if (number == 25) {number++; continue;}
         xlow = xmin + i*wpad;
         sprintf(command,"SetFillStyle(%d)",3001+number);
         test1 = new TGroupButton("Style","",command,xlow, ylow, xlow+0.9*wpad, ylow+0.9*hpad);
         if (number == 26) { //fill
            test1->SetBorderMode(-1);
            test1->SetFillColor(1);
            test1->SetMethod("SetFillStyle(1001)");
         } else if (number == 27) { //hollow
            test1->SetMethod("SetFillStyle(0)");
            test1->SetFillStyle(1001);
            test1->SetFillColor(10);
         } else { //pattern
            test1->SetFillColor(10);
            test1->SetFillStyle(3001+number);
         }
         test1->SetBorderSize(2);
         test1->Draw();
         number++;
      }
   }

//*-* draw colortable pads
   test1->DisplayColorTable("SetFillColor",0.05, 0.60, 0.90, 0.38);
   Update();
   SetEditable(kFALSE);

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
