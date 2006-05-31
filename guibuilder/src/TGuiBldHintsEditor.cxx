// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldHintsEditor.cxx,v 1.6 2006/05/28 20:15:09 brun Exp $
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
#include "TGuiBldNameFrame.h"

ClassImp(TGuiBldHintsEditor)

////////////////////////////////////////////////////////////////////////////////
class TGuiBldHintsManager : public TGVerticalFrame {

public:
   TGuiBldEditor  *fEditor;
   TGNumberEntry  *fColumns;
   TGNumberEntry  *fRows;
   TGCheckButton  *fButton; 
   TGuiBldHintsEditor *fHints;
   TGMatrixLayout *fMatrix;

   UInt_t  fPadTop;      // save values
   UInt_t  fPadBottom;   //   
   UInt_t  fPadLeft;     //
   UInt_t  fPadRight;    //

public:
   TGuiBldHintsManager(const TGWindow *p, TGuiBldEditor *editor, TGuiBldHintsEditor *hints);
   virtual ~TGuiBldHintsManager() { }
   void ChangeSelected(TGFrame *frame);
   Bool_t IsLayoutSubframes() const { return fButton->IsDown(); }
};

//______________________________________________________________________________
TGuiBldHintsManager::TGuiBldHintsManager(const TGWindow *p, TGuiBldEditor *e, 
                                          TGuiBldHintsEditor *hints) : 
                     TGVerticalFrame(p, 1, 1), fEditor(e), fHints(hints)
{
   // Constructor.

   fEditDisabled = kEditDisable;
   SetCleanup(kDeepCleanup);
   fRows = 0;
   fColumns = 0;
   fButton = 0;

   TGHorizontal3DLine *frame394 = new TGHorizontal3DLine(this,125,4);
   AddFrame(frame394, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX,0,0,2,2));

   // horizontal frame
   TGHorizontalFrame *frame399 = new TGHorizontalFrame(this,123,21,kHorizontalFrame);
   fButton = new TGCheckButton(frame399,"");
   frame399->AddFrame(fButton, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,1,2,2));
   TGLabel *frame401 = new TGLabel(frame399,"Layout subframes");
   frame399->AddFrame(frame401, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,1,1,2,2));

   AddFrame(frame399, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,1,1,1,1));

   // horizontal frame
   TGHorizontalFrame *frame416 = new TGHorizontalFrame(this,115,56,kHorizontalFrame);

   // vertical frame
   TGVerticalFrame *frame417 = new TGVerticalFrame(frame416,53,52,kVerticalFrame);
   TGLabel *frame419 = new TGLabel(frame417,"columns");
   frame417->AddFrame(frame419, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,1,1));
   fColumns = new TGNumberEntry(frame417, (Double_t) 1,3,-1,(TGNumberFormat::EStyle) 5);
   frame417->AddFrame(fColumns, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,1,1,5,5));

   frame416->AddFrame(frame417, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX | kLHintsExpandY,2,2,2,2));

   // vertical frame
   TGVerticalFrame *frame418 = new TGVerticalFrame(frame416,54,52,kVerticalFrame);
   TGLabel *frame420 = new TGLabel(frame418,"rows");
   frame418->AddFrame(frame420, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,1,1));
   fRows = new TGNumberEntry(frame418, (Double_t) 1,3,-1,(TGNumberFormat::EStyle) 5);
   frame418->AddFrame(fRows, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,1,1,5,5));

   frame416->AddFrame(frame418, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX | kLHintsExpandY,2,2,2,2));

   AddFrame(frame416, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsExpandX,5,5,0,0));

   fButton->Connect("Toggled(Bool_t)", "TGuiBldHintsEditor", fHints, "LayoutSubframes(Bool_t)");
   fRows->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
   fColumns->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
   fRows->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
   fColumns->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
   fRows->SetLimits(TGNumberFormat::kNELLimitMin, 1);
   fColumns->SetLimits(TGNumberFormat::kNELLimitMin, 1);

   fPadTop = 2;
   fPadBottom = 2;
   fPadLeft = 2;
   fPadRight = 2;

   MapSubwindows();
   Resize();
   MapWindow();
}

//______________________________________________________________________________
void TGuiBldHintsManager::ChangeSelected(TGFrame *frame)
{
   // action whne selcted/grabbed frame was changed

   fMatrix = 0;

   if (!frame) {
      UnmapWindow();
      fButton->SetEnabled(kFALSE);
      fButton->SetDown(kFALSE);
      fRows->SetNumber(0);
      fColumns->SetNumber(0);
      return;
   }

   Bool_t enable = frame->InheritsFrom(TGCompositeFrame::Class()) &&
                   !(frame->GetEditDisabled() & kEditDisableLayout);

   if (!enable) {
      UnmapWindow();
      fButton->SetEnabled(kFALSE);
      fButton->SetDown(kFALSE);
      fRows->SetNumber(0);
      fColumns->SetNumber(0);
   } else {
      TGCompositeFrame *comp = (TGCompositeFrame*)frame;
      TGLayoutManager *lm = comp->GetLayoutManager();

      if (!lm) {
         return;
      }
      Int_t n = comp->GetList()->GetEntries();

      MapWindow();
      fButton->SetEnabled(kTRUE);
      fButton->SetDown(kFALSE);

      if (lm->IsA() == TGVerticalLayout::Class()) {
         fRows->SetNumber(n);
         fColumns->SetNumber(1);
      } else if (lm->IsA() == TGHorizontalLayout::Class()) {
         fColumns->SetNumber(n);
         fRows->SetNumber(1);
      } else if (lm->IsA() == TGMatrixLayout::Class()) {
         fMatrix = (TGMatrixLayout*)lm;

         fColumns->SetNumber(fMatrix->fColumns);
         fRows->SetNumber(fMatrix->fRows);
      }
   }
   DoRedraw();
}

///////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldHintsEditor::TGuiBldHintsEditor(const TGWindow *p, TGuiBldEditor *e) :
                     TGVerticalFrame(p, 1, 1), fEditor(e)
{
   // ctor.

   SetCleanup(kDeepCleanup);

   TGVerticalFrame *frame3 = new TGVerticalFrame(this,262,18,kVerticalFrame);
   fNameFrame = new TGuiBldNameFrame(frame3, e);
   frame3->AddFrame(fNameFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,5,5,2,2));

   // vertical frame
   TGVerticalFrame *frame14 = new TGVerticalFrame(frame3,162,82,kVerticalFrame );

   // horizontal frame
   TGHorizontalFrame *frame15 = new TGHorizontalFrame(frame14,88,22,kHorizontalFrame);

   // composite frame
   TGCompositeFrame *frame16 = new TGCompositeFrame(frame15,88,22,kHorizontalFrame);
   fHintsTop = new TGTextButton(frame16,"T");
   fHintsTop->SetToolTipText("Set amount of top padding", 350);
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
   fHintsLeft = new TGTextButton(frame23," L");
   fHintsLeft->SetToolTipText("Set amount of left padding", 350);
   fHintsLeft->AllowStayDown(kTRUE);
   fHintsLeft->Resize(32,22);
   frame23->AddFrame(fHintsLeft);
   fPadLeft = new TGNumberEntry(frame23, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame23->AddFrame(fPadLeft);
   frame22->AddFrame(frame23, new TGLayoutHints(kLHintsNormal));

   // composite frame
   TGCompositeFrame *frame29 = new TGCompositeFrame(frame22,79,22,kHorizontalFrame);
   fHintsRight = new TGTextButton(frame29,"R");
   fHintsRight->SetToolTipText("Set amount of right padding", 350);
   fHintsRight->AllowStayDown(kTRUE);
   fHintsRight->Resize(36,22);
   frame29->AddFrame(fHintsRight);
   fPadRight = new TGNumberEntry(frame29, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame29->AddFrame(fPadRight);
   frame22->AddFrame(frame29, new TGLayoutHints(kLHintsNormal,4,0));
   frame14->AddFrame(frame22, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));

   // horizontal frame
   TGHorizontalFrame *frame35 = new TGHorizontalFrame(frame14,88,22,kHorizontalFrame);
   fHintsBottom = new TGTextButton(frame35,"B");
   fHintsBottom->SetToolTipText("Set amount of bottom padding", 350);
   fHintsBottom->AllowStayDown(kTRUE);
   fHintsBottom->Resize(45,22);
   frame35->AddFrame(fHintsBottom);
   fPadBottom = new TGNumberEntry(frame35, (Double_t) 0,3,-1,(TGNumberFormat::EStyle) 0);
   frame35->AddFrame(fPadBottom);
   frame14->AddFrame(frame35, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));
   frame3->AddFrame(frame14, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,2,2,2,2));

   TGHorizontalFrame *frame7 = new TGHorizontalFrame(frame3,89,48);

   // horizontal frame
   TGHorizontalFrame *frame8 = new TGHorizontalFrame(frame7,22,22,kHorizontalFrame);
   fCenterY = new TGuiBldHintsButton(frame8, kLHintsCenterY);
   fCenterY->SetToolTipText("Center frame in Y", 350);
   frame8->AddFrame(fCenterY, new TGLayoutHints(kLHintsNormal,1,1,1,1));
   fExpandY = new TGuiBldHintsButton(frame8, kLHintsExpandY);
   fExpandY->SetToolTipText("Expand frame in Y", 350);
   frame8->AddFrame(fExpandY, new TGLayoutHints(kLHintsNormal,1,1,1,1));
   frame7->AddFrame(frame8, new TGLayoutHints(kLHintsCenterY | kLHintsCenterX,2,2,2,2));

   // vertical frame
   TGVerticalFrame *frame11 = new TGVerticalFrame(frame7,59,44,kVerticalFrame);
   fCenterX = new TGuiBldHintsButton(frame11, kLHintsCenterX);
   fCenterX->SetToolTipText("Center frame in X", 350);
   frame11->AddFrame(fCenterX, new TGLayoutHints(kLHintsCenterY,1,1,1,1));
   fExpandX = new TGuiBldHintsButton(frame11, kLHintsExpandX);
   fExpandX->SetToolTipText("Expand frame in X", 350);
   frame11->AddFrame(fExpandX, new TGLayoutHints(kLHintsCenterY,1,1,1,1));
   frame7->AddFrame(frame11, new TGLayoutHints(kLHintsLeft | kLHintsTop,2,2,2,2));
   frame3->AddFrame(frame7, new TGLayoutHints(kLHintsCenterY | kLHintsCenterX ,2,2,2,2));

   fHintsManager = new TGuiBldHintsManager(frame3, e, this);
   frame3->AddFrame(fHintsManager, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,5,5,2,2));
   fHintsManager->UnmapWindow();
   AddFrame(frame3);

   SetEditDisabled(1);
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

   fPadTop->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadLeft->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadRight->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadBottom->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
}

//______________________________________________________________________________
void  TGuiBldHintsEditor::ChangeSelected(TGFrame *frame)
{
   // Change selected

   if (!frame) {
      fNameFrame->Reset();
      return;
   }
   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      fNameFrame->Reset();
      return;
   }
      
   fNameFrame->ChangeSelected(frame);
   fHintsManager->ChangeSelected(frame);

   ULong_t lh = fe->fLayout->GetLayoutHints();

   fCenterX->SetEnabled(kTRUE);
   fCenterY->SetEnabled(kTRUE);
   fExpandX->SetEnabled(!(frame->GetEditDisabled() & kEditDisableWidth));
   fExpandY->SetEnabled(!(frame->GetEditDisabled() & kEditDisableHeight));
   fClient->NeedRedraw(fExpandX);
   fClient->NeedRedraw(fExpandY);

   fHintsTop->SetEnabled(kTRUE);
   fHintsRight->SetEnabled(kTRUE);
   fHintsLeft->SetEnabled(kTRUE);
   fHintsBottom->SetEnabled(kTRUE);

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
void TGuiBldHintsEditor::UpdateState()
{
   // Update state

   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      fNameFrame->Reset();
      return;
   }

   if (fHintsManager->IsLayoutSubframes() && 
       ((gTQSender == fPadTop) || (gTQSender == fPadBottom) ||
       (gTQSender == fPadLeft) || (gTQSender == fPadRight))) {
      SetMatrixSep();
      return;
   }

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

//______________________________________________________________________________
void TGuiBldHintsEditor::LayoutSubframes(Bool_t on)
{
   // Layout subframes.

   if (!fEditor) {
      return;
   }
   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      fNameFrame->Reset();
      return;
   }

   Bool_t enable = frame->InheritsFrom(TGCompositeFrame::Class()) &&
                   !(frame->GetEditDisabled() & kEditDisableLayout);

   if (!on) {
      fPadTop->SetIntNumber(fHintsManager->fPadTop);
      fPadBottom->SetIntNumber(fHintsManager->fPadBottom); 
      fPadLeft->SetIntNumber(fHintsManager->fPadLeft);
      fPadRight->SetIntNumber(fHintsManager->fPadRight);

      ChangeSelected(frame);
      return;
   }
   if (!enable) {
      return;
   }

   TGCompositeFrame *comp = (TGCompositeFrame*)frame;
   fHintsManager->fRows->SetState(kTRUE);
   fHintsManager->fColumns->SetState(kTRUE);
   comp->SetLayoutBroken(kFALSE);

   if (!fHintsManager->fMatrix) {
      if (!(frame->GetParent()->GetEditDisabled() & kEditDisableLayout)) {
         comp->Resize();
      } else {
         if (comp->GetLayoutManager()) {
            comp->GetLayoutManager()->Layout();
         } else {
            comp->Layout();
         }
      }
      return;
   }

   MatrixLayout();
}

//______________________________________________________________________________
void TGuiBldHintsEditor::SetMatrixSep()
{
   // Set matrix layout separator.

   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      fNameFrame->Reset();
      return;
   }

   Bool_t enable = frame->InheritsFrom(TGCompositeFrame::Class()) &&
                   !(frame->GetEditDisabled() & kEditDisableLayout) && 
                    ((TGCompositeFrame*)frame)->GetLayoutManager() &&
                    ((TGCompositeFrame*)frame)->GetLayoutManager()->InheritsFrom(TGMatrixLayout::Class());

   if (!enable) {
      return;
   }

   TGNumberEntry *ne = (TGNumberEntry*)gTQSender;
   UInt_t sep = ne->GetIntNumber();

   fPadTop->SetIntNumber(sep);
   fPadLeft->SetIntNumber(sep);
   fPadRight->SetIntNumber(sep);
   fPadBottom->SetIntNumber(sep);
   fHintsManager->fButton->SetDown(kTRUE);

   fHintsManager->fMatrix->fSep = sep;
   frame->SetLayoutBroken(kFALSE);

   if (!(frame->GetParent()->GetEditDisabled() & kEditDisableLayout)) {
      frame->Resize();
   } else {
      fHintsManager->fMatrix->Layout();
   }
   fClient->NeedRedraw(frame, kTRUE);
}

//______________________________________________________________________________
void TGuiBldHintsEditor::MatrixLayout()
{
   // Apply matrix layout.

   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      fNameFrame->Reset();
      return;
   }

   Bool_t enable = frame->InheritsFrom(TGCompositeFrame::Class()) &&
                   !(frame->GetEditDisabled() & kEditDisableLayout);

   if (!enable) {
      return;
   }

   TGCompositeFrame *comp = (TGCompositeFrame*)frame;

   UInt_t rows = fHintsManager->fRows->GetIntNumber();
   UInt_t cols = fHintsManager->fColumns->GetIntNumber();
   UInt_t sep = fPadTop->GetIntNumber();
/*
   fCenterX->SetEnabled(kFALSE);
   fCenterY->SetEnabled(kFALSE);
   fExpandX->SetEnabled(kFALSE);
   fExpandY->SetEnabled(kFALSE);

   fHintsTop->SetEnabled(kFALSE);
   fHintsRight->SetEnabled(kFALSE);
   fHintsLeft->SetEnabled(kFALSE);
   fHintsBottom->SetEnabled(kFALSE);
*/
   fHintsManager->fPadTop = fPadTop->GetIntNumber();      // save
   fHintsManager->fPadBottom = fPadBottom->GetIntNumber();   //   
   fHintsManager->fPadLeft = fPadLeft->GetIntNumber();     //
   fHintsManager->fPadRight = fPadRight->GetIntNumber(); 

   fPadTop->SetIntNumber(sep);
   fPadLeft->SetIntNumber(sep);
   fPadRight->SetIntNumber(sep);
   fPadBottom->SetIntNumber(sep);

   fHintsManager->fRows->SetState(kTRUE);
   fHintsManager->fColumns->SetState(kTRUE);

   comp->SetLayoutBroken(kFALSE);

   fHintsManager->fMatrix = new TGMatrixLayout(comp, rows, cols, sep, 0);
   comp->SetLayoutManager(fHintsManager->fMatrix);

   if (!(comp->GetParent()->GetEditDisabled() & kEditDisableLayout)) {
      comp->Resize();
   } else {
      fHintsManager->fMatrix->Layout();
   }
   fClient->NeedRedraw(comp, kTRUE);
}
