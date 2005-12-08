// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldHintsEditor.cxx,v 1.2 2004/09/14 09:57:58 brun Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldHintsEditor                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiBldHintsEditor.h"
#include "TGuiBldHintsButton.h"
#include "TGNumberEntry.h"
#include "TGuiBldEditor.h"
#include "TGLabel.h"
#include "TG3DLine.h"

ClassImp(TGuiBldHintsEditor)


///////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldHintsEditor::TGuiBldHintsEditor(const TGWindow *p, TGuiBldEditor *e) :
                     TGVerticalFrame(p, 1, 1), fEditor(e)
{
   //

   fEditDisabled = kTRUE;

   // horizontal frame
   TGHorizontalFrame *frame3 = new TGHorizontalFrame(this,262,18,kHorizontalFrame);
   TGLabel *frame4 = new TGLabel(frame3,"Layout");
   frame3->AddFrame(frame4, new TGLayoutHints(kLHintsLeft | kLHintsTop));
   TGHorizontal3DLine *frame5 = new TGHorizontal3DLine(frame3,211,5);
   frame3->AddFrame(frame5, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX,5,5));
   AddFrame(frame3, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX));
   frame3->MoveResize(5,1,262,18);

   // horizontal frame
   TGHorizontalFrame *frame6 = new TGHorizontalFrame(this,259,90,kHorizontalFrame );

   // vertical frame
   TGVerticalFrame *frame14 = new TGVerticalFrame(frame6,162,82,kVerticalFrame );

   // horizontal frame
   TGHorizontalFrame *frame15 = new TGHorizontalFrame(frame14,88,22,kHorizontalFrame);

   // composite frame
   TGCompositeFrame *frame16 = new TGCompositeFrame(frame15,88,22,kHorizontalFrame);
   fHintsTop = new TGTextButton(frame16,"  Top  ");
   fHintsTop->AllowStayDown(kTRUE);
   fHintsTop->Resize(45,22);
   frame16->AddFrame(fHintsTop);
   fPadTop = new TGNumberEntry(frame16, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame16->AddFrame(fPadTop);
   frame15->AddFrame(frame16);
   frame14->AddFrame(frame15, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));

   // horizontal frame
   TGHorizontalFrame *frame22 = new TGHorizontalFrame(frame14,154,22,kHorizontalFrame);

   // composite frame
   TGCompositeFrame *frame23 = new TGCompositeFrame(frame22,75,22,kHorizontalFrame);
   fHintsLeft = new TGTextButton(frame23," Left ");
   fHintsLeft->AllowStayDown(kTRUE);
   fHintsLeft->Resize(32,22);
   frame23->AddFrame(fHintsLeft);
   fPadLeft = new TGNumberEntry(frame23, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame23->AddFrame(fPadLeft);
   frame22->AddFrame(frame23, new TGLayoutHints(kLHintsNormal));

   // composite frame
   TGCompositeFrame *frame29 = new TGCompositeFrame(frame22,79,22,kHorizontalFrame);
   fHintsRight = new TGTextButton(frame29,"Right");
   fHintsRight->AllowStayDown(kTRUE);
   fHintsRight->Resize(36,22);
   frame29->AddFrame(fHintsRight);
   fPadRight = new TGNumberEntry(frame29, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame29->AddFrame(fPadRight);
   frame22->AddFrame(frame29, new TGLayoutHints(kLHintsNormal,4,0));
   frame14->AddFrame(frame22, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));

   // horizontal frame
   TGHorizontalFrame *frame35 = new TGHorizontalFrame(frame14,88,22,kHorizontalFrame);
   fHintsBottom = new TGTextButton(frame35,"Bottom");
   fHintsBottom->AllowStayDown(kTRUE);
   fHintsBottom->Resize(45,22);
   frame35->AddFrame(fHintsBottom);
   fPadBottom = new TGNumberEntry(frame35, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame35->AddFrame(fPadBottom);
   frame14->AddFrame(frame35, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));
   frame6->AddFrame(frame14, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));

   // horizontal frame
   TGVerticalFrame *frame7 = new TGVerticalFrame(frame6,89,48);

   // horizontal frame
   TGHorizontalFrame *frame8 = new TGHorizontalFrame(frame7,22,22,kHorizontalFrame);
   fCenterY = new TGuiBldHintsButton(frame8, kLHintsCenterY);
   frame8->AddFrame(fCenterY, new TGLayoutHints(kLHintsNormal,1,1,1,1));
   fExpandY = new TGuiBldHintsButton(frame8, kLHintsExpandY);
   frame8->AddFrame(fExpandY, new TGLayoutHints(kLHintsNormal,1,1,1,1));
   frame7->AddFrame(frame8, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterX,2,2,2,2));

   // vertical frame
   TGVerticalFrame *frame11 = new TGVerticalFrame(frame7,59,44,kVerticalFrame);
   fCenterX = new TGuiBldHintsButton(frame11, kLHintsCenterX);
   frame11->AddFrame(fCenterX, new TGLayoutHints(kLHintsNormal,1,1,1,1));
   fExpandX = new TGuiBldHintsButton(frame11, kLHintsExpandX);
   frame11->AddFrame(fExpandX, new TGLayoutHints(kLHintsNormal,1,1,1,1));
   frame7->AddFrame(frame11, new TGLayoutHints(kLHintsLeft | kLHintsTop,2,2,2,2));
   frame6->AddFrame(frame7);
   AddFrame(frame6);
   frame6->MoveResize(4,23,259,90);

   Resize();
   MapSubwindows();
   MapWindow();

   fExpandX->Connect("Pressed()", "TGButton", fCenterX, "SetDown(=kFALSE)");
   fCenterX->Connect("Pressed()", "TGButton", fExpandX, "SetDown(=kFALSE)");
   fExpandY->Connect("Pressed()", "TGButton", fCenterY, "SetDown(=kFALSE)");
   fCenterY->Connect("Pressed()", "TGButton", fExpandY, "SetDown(=kFALSE)");

   fHintsTop->Connect("Pressed()", "TGButton", fHintsBottom, "SetDown(=kFALSE)");
   fHintsBottom->Connect("Pressed()", "TGButton", fHintsTop, "SetDown(=kFALSE))");
   fHintsRight->Connect("Pressed()", "TGButton", fHintsLeft, "SetDown(=kFALSE))");
   fHintsLeft->Connect("Pressed()", "TGButton", fHintsRight, "SetDown(=kFALSE))");

   fExpandX->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCenterX->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fExpandY->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCenterY->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fHintsTop->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fHintsBottom->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fHintsRight->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fHintsLeft->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadTop->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadLeft->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadRight->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadBottom->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
}

//______________________________________________________________________________
void  TGuiBldHintsEditor::ChangeSelected(TGFrame *frame)
{
   //

   if (!frame) return;

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) return;

   ULong_t lh = fe->fLayout->GetLayoutHints();

   fCenterX->SetDown(lh & kLHintsCenterX);
   fCenterY->SetDown(lh & kLHintsCenterY);
   fExpandX->SetDown(lh & kLHintsExpandX);
   fExpandY->SetDown(lh & kLHintsExpandY);

   fHintsTop->SetDown(lh & kLHintsTop);
   fHintsRight->SetDown(lh & kLHintsRight);
   fHintsLeft->SetDown(lh & kLHintsLeft);
   fHintsBottom->SetDown(lh & kLHintsBottom);

   fPadTop->SetIntNumber(fe->fLayout->GetPadTop());
   fPadLeft->SetIntNumber(fe->fLayout->GetPadLeft());
   fPadRight->SetIntNumber(fe->fLayout->GetPadRight());
   fPadBottom->SetIntNumber(fe->fLayout->GetPadBottom());
}

 //______________________________________________________________________________
void  TGuiBldHintsEditor::UpdateState()
{
   //

   TGFrame *frame = fEditor->GetSelected();

   if (!frame) return;

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) return;

   ULong_t lh = fe->fLayout->GetLayoutHints();

   if (fCenterX->IsDown()) {
      lh |= kLHintsCenterX;
      lh &= ~kLHintsExpandX;
   } else {
      lh &= ~kLHintsCenterX;
   }

   if (fCenterY->IsDown()) {
      lh |= kLHintsCenterY;
      lh &= ~kLHintsExpandY;
   } else {
      lh &= ~kLHintsCenterY;
   }

   if (fExpandX->IsDown()) {
      lh |= kLHintsExpandX;
      lh &= ~kLHintsCenterX;
   } else {
      lh &= ~kLHintsExpandX;
   }

   if (fExpandY->IsDown()) {
      lh |= kLHintsExpandY;
      lh &= ~kLHintsCenterY;
   } else {
      lh &= ~kLHintsExpandY;
   }

   if (fHintsTop->IsDown()) {
      lh |= kLHintsTop;
      lh &= ~kLHintsBottom;
   } else {
      lh &= ~kLHintsTop;
   }

   if (fHintsBottom->IsDown()) {
      lh |= kLHintsBottom;
      //lh &= ~kLHintsTop;
   } else {
      lh &= ~kLHintsBottom;
   }

   if (fHintsRight->IsDown()) {
      lh |= kLHintsRight;
      lh &= ~kLHintsLeft;
   } else {
      lh &= ~kLHintsRight;
   }

   if (fHintsLeft->IsDown()) {
      lh |= kLHintsLeft;
      lh &= ~kLHintsRight;
   } else {
      lh &= ~kLHintsLeft;
   }

   fe->fLayout->SetPadLeft(fPadLeft->GetIntNumber());
   fe->fLayout->SetPadRight(fPadRight->GetIntNumber());
   fe->fLayout->SetPadTop(fPadTop->GetIntNumber());
   fe->fLayout->SetPadBottom(fPadBottom->GetIntNumber());

   if (fe->fLayout->References() > 1) {
      TGLayoutHints *lh = new TGLayoutHints(*fe->fLayout);
      fe->fLayout->RemoveReference();
      lh->AddReference();
      fe->fLayout = lh;
   } else {
      fe->fLayout->SetLayoutHints(lh);
   }

   fEditor->UpdateSelected(frame);
}
