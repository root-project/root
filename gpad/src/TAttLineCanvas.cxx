// @(#)root/gpad:$Name:  $:$Id: TAttLineCanvas.cxx,v 1.1.1.1 2000/05/16 17:00:41 rdm Exp $
// Author: Rene Brun   03/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TAttLineCanvas.h"
#include "TGroupButton.h"
#include "TLine.h"
#include "TText.h"

ClassImp(TAttLineCanvas)

//______________________________________________________________________________
//
//   An AttLineCanvas is a TDialogCanvas specialized to set line attributes.
//Begin_Html
/*
<img src="gif/attlinecanvas.gif">
*/
//End_Html
//

//______________________________________________________________________________
TAttLineCanvas::TAttLineCanvas() : TDialogCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*-*AttLineCanvas default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

}

//_____________________________________________________________________________
TAttLineCanvas::TAttLineCanvas(const char *name, const char *title, UInt_t ww, UInt_t wh)
             : TDialogCanvas(name,title,ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*AttLineCanvas constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   TVirtualPad *padsav = gPad;

   BuildStandardButtons();

//*-*- Line styles choice buttons
   TGroupButton *test1 = 0;
   TLine *line;
   Float_t xlow, ylow, wpad, hpad;
   Int_t i,j;
   xlow = 0.05;
   wpad = 0.9;
   hpad = 0.042;
   char command[64];
   for (i=0;i<4;i++) {
      ylow = 0.13 + i*hpad;
      sprintf(command,"SetLineStyle(%d)",i+1);
      test1 = new TGroupButton("Style","",command,xlow, ylow, xlow+wpad, ylow+0.8*hpad);
      test1->SetEditable(kTRUE);
      if (i == 0) test1->SetBorderMode(-1);
      test1->SetBorderSize(1);
      test1->Draw();
      test1->cd();
      line = new TLine(0.05, 0.5, 0.95,0.5);
      line->SetLineColor(1);
      line->SetLineStyle(i+1);
      line->Draw();
      test1->SetEditable(kFALSE);
      cd();
   }

//*-*-  Line Width choice buttons
   wpad = 0.19;
   hpad = 0.085;
   Int_t number = 0;
   for (j=0;j<3;j++) {
      ylow = 0.32 + j*hpad;
      for (i=0;i<5;i++) {
         number++;
         xlow = 0.05 + i*wpad;
         sprintf(command,"SetLineWidth(%d)",number);
         test1 = new TGroupButton("Width","",command,xlow, ylow, xlow+0.9*wpad, ylow+0.9*hpad);
         test1->SetEditable(kTRUE);
         if (number == 1) test1->SetBorderMode(-1);
         test1->SetFillColor(18);
         test1->SetBorderSize(2);
         test1->Draw();
         test1->cd();
         line = new TLine(0.5, 0.1, 0.5,0.9);
         line->SetLineColor(1);
         line->SetLineWidth(number);
         line->Draw();
         test1->SetEditable(kFALSE);
         cd();
      }
   }

//*-* draw colortable pads
   test1->DisplayColorTable("SetLineColor",0.05, 0.60, 0.90, 0.38);
   Update();
   SetEditable(kFALSE);

   padsav->cd();
}

//______________________________________________________________________________
TAttLineCanvas::~TAttLineCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*AttLineCanvas default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
}

//______________________________________________________________________________
void TAttLineCanvas::UpdateLineAttributes(Int_t col, Int_t sty, Int_t width)
{
//*-*-*-*-*-*-*-*-*-*-*Update object attributes*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TIter next(GetListOfPrimitives());
   TGroupButton *button;
   char cmd[64];
   fRefObject = gROOT->GetSelectedPrimitive();
   fRefPad    = (TPad*)gROOT->GetSelectedPad();
   if (fRefObject) {
      sprintf(cmd,"attline: %s",fRefObject->GetName());
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
      sprintf(cmd,"SetLineColor(%d)",col);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      sprintf(cmd,"SetLineStyle(%d)",sty);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      sprintf(cmd,"SetLineWidth(%d)",width);
      if (!strcmp(button->GetTitle(),cmd)) button->SetBorderMode(-1);
      if (button->GetBorderMode() < 0) {
         button->Modified();
      }
   }
   Update();
}
