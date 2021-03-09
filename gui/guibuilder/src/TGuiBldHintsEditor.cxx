// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGuiBldHintsEditor.h"
#include "TGuiBldHintsButton.h"
#include "TGNumberEntry.h"
#include "TGuiBldEditor.h"
#include "TGLabel.h"
#include "TG3DLine.h"
#include "TGuiBldNameFrame.h"
#include "TGuiBldGeometryFrame.h"
#include "TRootGuiBuilder.h"
#include "TGTableLayout.h"


/** \class TGuiBldHintsEditor
    \ingroup guibuilder

Editor of widget's layout hints used by the ROOT GUI builder.

*/


ClassImp(TGuiBldHintsEditor);

////////////////////////////////////////////////////////////////////////////////
class TGuiBldHintsManager : public TGVerticalFrame {

public:
   TGuiBldEditor  *fEditor;
   TGNumberEntry  *fColumns;
   TGNumberEntry  *fRows;
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
};

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGuiBldHintsManager::TGuiBldHintsManager(const TGWindow *p, TGuiBldEditor *e,
                                          TGuiBldHintsEditor *hints) :
                     TGVerticalFrame(p, 1, 1), fEditor(e), fHints(hints)
{
   fEditDisabled = kEditDisable;
   SetCleanup(kDeepCleanup);
   fRows = 0;
   fColumns = 0;

   //-----check button to layout subframes was moved to HintsEditor to be generalized ------

   // "Matrix layout" group frame
   TGGroupFrame *fGroupFrame4066 = new TGGroupFrame(this, "Matrix layout");
   TGHorizontalFrame *f = new TGHorizontalFrame(fGroupFrame4066);

   f->AddFrame(new TGLabel(f," Cols "), new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));
   fColumns = new TGNumberEntry(f,0.0,4,-1,(TGNumberFormat::EStyle)5);
   f->AddFrame(fColumns, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));

   f->AddFrame(new TGLabel(f," Rows "), new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));
   fRows = new TGNumberEntry(f,0.0,4,-1,(TGNumberFormat::EStyle)5);
   f->AddFrame(fRows, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));

   fGroupFrame4066->AddFrame(f, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 2, 2, 2, 2));

   TGTextButton *fAppButton = new TGTextButton(fGroupFrame4066, " Apply ");
   fGroupFrame4066->AddFrame(fAppButton, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 5, 5, 2, 2));

   AddFrame(fGroupFrame4066, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   fAppButton->Connect("Clicked()", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
   //fRows->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
   //fColumns->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", fHints, "MatrixLayout()");
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

////////////////////////////////////////////////////////////////////////////////
/// action when selected/grabbed frame was changed

void TGuiBldHintsManager::ChangeSelected(TGFrame *frame)
{
   fMatrix = 0;

   if (!frame) {
      UnmapWindow();
      fHints->fLayButton->SetEnabled(kFALSE);
      fHints->fLayButton->SetDown(kFALSE);
      fRows->SetNumber(0);
      fColumns->SetNumber(0);
      return;
   }

   Bool_t enable = frame->InheritsFrom(TGCompositeFrame::Class()) &&
                   !(frame->GetEditDisabled() & kEditDisableLayout);

   if (!enable) {
      UnmapWindow();
      fHints->fLayButton->SetEnabled(kFALSE);
      fHints->fLayButton->SetDown(kFALSE);
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
      fHints->fLayButton->SetEnabled(kTRUE);
      fHints->fLayButton->SetDown(kFALSE);

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
////////////////////////////////////////////////////////////////////////////////
/// ctor.

TGuiBldHintsEditor::TGuiBldHintsEditor(const TGWindow *p, TGuiBldEditor *e) :
                     TGVerticalFrame(p, 1, 1), fEditor(e)
{
   SetCleanup(kDeepCleanup);

   fBuilder = (TRootGuiBuilder*)TRootGuiBuilder::Instance();

   TGVerticalFrame *frame3 = new TGVerticalFrame(this,262,18,kVerticalFrame);

   // horizontal frame - layout subframes (token from matrix layout)
   TGHorizontalFrame *framez399 = new TGHorizontalFrame(frame3,123,40,kHorizontalFrame);
   fLayButton = new TGCheckButton(framez399,"");
   framez399->AddFrame(fLayButton, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,1,2,2));
   TGLabel *framez401 = new TGLabel(framez399,"Layout subframes");
   framez399->AddFrame(framez401, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,1,1,2,2));

   frame3->AddFrame(framez399, new TGLayoutHints(kLHintsLeft | kLHintsCenterX | kLHintsTop,1,1,1,1));

   fLayButton->Connect("Toggled(Bool_t)", "TGuiBldHintsEditor", this, "LayoutSubframes(Bool_t)");

   //--------layout hints in new layout---------------------------------------

   // "Padding" group frame
   fPaddingFrame = new TGGroupFrame(frame3, "Padding");
   fPaddingFrame->SetLayoutManager(new TGTableLayout(fPaddingFrame, 2, 4));
   
   fPaddingFrame->AddFrame(new TGLabel(fPaddingFrame,"Top "), 
                           new TGTableLayoutHints(0, 1, 0, 1, 
                           kLHintsRight | kLHintsCenterY, 0, 2, 2, 2));
   fPadTop = new TGNumberEntry(fPaddingFrame,0.0,4,-1,(TGNumberFormat::EStyle) 5);
   fPaddingFrame->AddFrame(fPadTop, new TGTableLayoutHints(1, 2, 0, 1,
                           kLHintsLeft | kLHintsCenterY, 0, 0, 2, 2));

   fPaddingFrame->AddFrame(new TGLabel(fPaddingFrame," Left "),
                           new TGTableLayoutHints(2, 3, 0, 1, 
                           kLHintsRight | kLHintsCenterY, 2, 2, 2, 2));
   fPadLeft = new TGNumberEntry(fPaddingFrame,0.0,4,-1,(TGNumberFormat::EStyle) 5);
   fPaddingFrame->AddFrame(fPadLeft, new TGTableLayoutHints(3, 4, 0, 1,
                           kLHintsLeft | kLHintsCenterY, 0, 0, 2, 2));

   fPaddingFrame->AddFrame(new TGLabel(fPaddingFrame,"Bottom "),
                           new TGTableLayoutHints(0, 1, 1, 2, 
                           kLHintsRight | kLHintsCenterY, 0, 2, 2, 2));
   fPadBottom = new TGNumberEntry(fPaddingFrame,0.0,4,-1,(TGNumberFormat::EStyle) 5);
   fPaddingFrame->AddFrame(fPadBottom, new TGTableLayoutHints(1, 2, 1, 2,
                           kLHintsLeft | kLHintsCenterY, 0, 0, 2, 2));

   fPaddingFrame->AddFrame(new TGLabel(fPaddingFrame," Right "),
                           new TGTableLayoutHints(2, 3, 1, 2, 
                           kLHintsRight | kLHintsCenterY, 2, 2, 2, 2));
   fPadRight = new TGNumberEntry(fPaddingFrame,0.0,4,-1,(TGNumberFormat::EStyle) 5);
   fPaddingFrame->AddFrame(fPadRight, new TGTableLayoutHints(3, 4, 1, 2,
                           kLHintsLeft | kLHintsCenterY, 0, 0, 2, 2));

   frame3->AddFrame(fPaddingFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 2, 2));

   // "Layout" group frame
   fHintsFrame = new TGGroupFrame(frame3,"Layout");

   fHintsFrame->SetLayoutManager(new TGTableLayout(fHintsFrame, 4, 2));

   fCbTop = new TGCheckButton(fHintsFrame, "Top");
   fHintsFrame->AddFrame(fCbTop, new TGTableLayoutHints(0, 1, 0, 1,
                         kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));

   fCbBottom = new TGCheckButton(fHintsFrame, "Bottom");
   fHintsFrame->AddFrame(fCbBottom, new TGTableLayoutHints(0, 1, 1, 2,
                         kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));

   fCbLeft = new TGCheckButton(fHintsFrame, "Left");
   fHintsFrame->AddFrame(fCbLeft, new TGTableLayoutHints(0, 1, 2, 3,
                         kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));

   fCbRight = new TGCheckButton(fHintsFrame, "Right");
   fHintsFrame->AddFrame(fCbRight, new TGTableLayoutHints(0, 1, 3, 4,
                         kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));

   fCbCenterX = new TGCheckButton(fHintsFrame, "Center in X");
   fHintsFrame->AddFrame(fCbCenterX, new TGTableLayoutHints(1, 2, 0, 1,
                         kLHintsLeft | kLHintsCenterY, 9, 2, 2, 2));

   fCbCenterY = new TGCheckButton(fHintsFrame, "Center in Y");
   fHintsFrame->AddFrame(fCbCenterY, new TGTableLayoutHints(1, 2, 1, 2,
                         kLHintsLeft | kLHintsCenterY, 9, 2, 2, 2));

   fCbExpandX = new TGCheckButton(fHintsFrame, "Expand in X");
   fHintsFrame->AddFrame(fCbExpandX, new TGTableLayoutHints(1, 2, 2, 3,
                         kLHintsLeft | kLHintsCenterY, 9, 2, 2, 2));

   fCbExpandY = new TGCheckButton(fHintsFrame, "Expand in Y");
   fHintsFrame->AddFrame(fCbExpandY, new TGTableLayoutHints(1, 2, 3, 4,
                         kLHintsLeft | kLHintsCenterY, 9, 2, 2, 2));

   frame3->AddFrame(fHintsFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 2, 2));

   fHintsManager = new TGuiBldHintsManager(frame3, e, this);
   frame3->AddFrame(fHintsManager, new TGLayoutHints(kLHintsBottom | kLHintsExpandX,2,2,2,2));
   fHintsManager->UnmapWindow();
   AddFrame(frame3);

   SetEditDisabled(1);
   Resize();
   MapSubwindows();
   MapWindow();

   fCbTop->Connect("Clicked()", "TGButton", fCbBottom, "SetDown(=kFALSE)");
   fCbBottom->Connect("Clicked()",  "TGButton", fCbTop, "SetDown(=kFALSE)");
   fCbRight->Connect("Clicked()",  "TGButton", fCbLeft, "SetDown(=kFALSE)");
   fCbLeft->Connect("Clicked()",  "TGButton", fCbRight, "SetDown(=kFALSE)");

   fCbTop->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbBottom->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbRight->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbLeft->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbExpandX->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbCenterX->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbExpandY->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");
   fCbCenterY->Connect("Clicked()", "TGuiBldHintsEditor", this, "UpdateState()");

   fPadTop->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadLeft->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadRight->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadBottom->Connect("ValueSet(Long_t)", "TGuiBldHintsEditor", this, "UpdateState()");

   fPadTop->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadLeft->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadRight->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
   fPadBottom->GetNumberEntry()->Connect("ReturnPressed()", "TGuiBldHintsEditor", this, "UpdateState()");
}


////////////////////////////////////////////////////////////////////////////////
/// Change selected

void  TGuiBldHintsEditor::ChangeSelected(TGFrame *frame)
{
   if (!frame) {
      return;
   }
   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      return;
   }

   fHintsManager->ChangeSelected(frame);

   ULong_t lh = fe->fLayout->GetLayoutHints();

   fCbCenterX->SetEnabled(kTRUE);
   fCbCenterY->SetEnabled(kTRUE);
   fCbExpandX->SetEnabled(!(frame->GetEditDisabled() & kEditDisableWidth));
   fCbExpandY->SetEnabled(!(frame->GetEditDisabled() & kEditDisableHeight));
   fClient->NeedRedraw(fCbExpandX);
   fClient->NeedRedraw(fCbExpandY);

   fCbTop->SetEnabled(kTRUE);
   fCbRight->SetEnabled(kTRUE);
   fCbLeft->SetEnabled(kTRUE);
   fCbBottom->SetEnabled(kTRUE);

   fCbCenterX->SetDown(lh & kLHintsCenterX);
   fCbCenterY->SetDown(lh & kLHintsCenterY);
   fCbExpandX->SetDown(lh & kLHintsExpandX);
   fCbExpandY->SetDown(lh & kLHintsExpandY);

   fCbTop->SetDown(lh & kLHintsTop);
   fCbRight->SetDown(lh & kLHintsRight);
   fCbLeft->SetDown(lh & kLHintsLeft);
   fCbBottom->SetDown(lh & kLHintsBottom);

   fPadTop->SetIntNumber(fe->fLayout->GetPadTop());
   fPadLeft->SetIntNumber(fe->fLayout->GetPadLeft());
   fPadRight->SetIntNumber(fe->fLayout->GetPadRight());
   fPadBottom->SetIntNumber(fe->fLayout->GetPadBottom());
}

////////////////////////////////////////////////////////////////////////////////
/// Update state

void TGuiBldHintsEditor::UpdateState()
{
   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      return;
   }

   if (fLayButton->IsDown() &&
       ((gTQSender == fPadTop) || (gTQSender == fPadBottom) ||
       (gTQSender == fPadLeft) || (gTQSender == fPadRight))) {
      SetMatrixSep();
      return;
   }

   ULong_t lh = fe->fLayout->GetLayoutHints();

   if (fCbCenterX->IsDown()) {
      lh |= kLHintsCenterX;
   } else {
      lh &= ~kLHintsCenterX;
   }

   if (fCbCenterY->IsDown()) {
      lh |= kLHintsCenterY;
   } else {
      lh &= ~kLHintsCenterY;
   }

   if (fCbExpandX->IsDown()) {
      lh |= kLHintsExpandX;
   } else {
      lh &= ~kLHintsExpandX;
   }

   if (fCbExpandY->IsDown()) {
      lh |= kLHintsExpandY;
   } else {
      lh &= ~kLHintsExpandY;
   }

   if (fCbTop->IsDown()) {
      lh |= kLHintsTop;
      lh &= ~kLHintsBottom;
   } else {
      lh &= ~kLHintsTop;
   }

   if (fCbBottom->IsDown()) {
      lh |= kLHintsBottom;
      lh &= ~kLHintsTop;
   } else {
      lh &= ~kLHintsBottom;
   }

   if (fCbRight->IsDown()) {
      lh |= kLHintsRight;
      lh &= ~kLHintsLeft;
   } else {
      lh &= ~kLHintsRight;
   }

   if (fCbLeft->IsDown()) {
      lh |= kLHintsLeft;
      lh &= ~kLHintsRight;
   } else {
      lh &= ~kLHintsLeft;
   }

   if (fPadLeft->GetIntNumber() >=0) {
     fe->fLayout->SetPadLeft(fPadLeft->GetIntNumber());
   }
   if (fPadRight->GetIntNumber() >=0) {
     fe->fLayout->SetPadRight(fPadRight->GetIntNumber());
   }
   if (fPadTop->GetIntNumber() >=0) {
     fe->fLayout->SetPadTop(fPadTop->GetIntNumber());
   }
   if (fPadBottom->GetIntNumber() >=0) {
     fe->fLayout->SetPadBottom(fPadBottom->GetIntNumber());
   }

   if (fe->fLayout->References() > 1) {
      TGLayoutHints *lh2 = new TGLayoutHints(*fe->fLayout);
      fe->fLayout->RemoveReference();
      lh2->AddReference();
      fe->fLayout = lh2;
   } else {
      fe->fLayout->SetLayoutHints(lh);
   }

   fEditor->UpdateSelected(frame);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the position of selected frame when adjusted by the right panel input.

void TGuiBldHintsEditor::SetPosition()
{
   if (!fEditor) {
      return;
   }
   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      //fNameFrame->Reset();
      return;
   }

   if ((fEditor->GetXPos() >= 0) && (fEditor->GetYPos() >= 0)) {
      frame->MoveResize(fEditor->GetXPos(), fEditor->GetYPos(),
                        frame->GetWidth(), frame->GetHeight());
      fClient->NeedRedraw(frame, kTRUE);
      TGWindow *root = (TGWindow*)fClient->GetRoot();
      fClient->NeedRedraw(root, kTRUE);
      if (fBuilder) {
         fClient->NeedRedraw(fBuilder, kTRUE);
      }
   } else {
      fEditor->SetYPos(frame->GetY());
      fEditor->SetXPos(frame->GetX());
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Layout subframes.

void TGuiBldHintsEditor::LayoutSubframes(Bool_t on)
{
   if (!fEditor) {
      return;
   }
   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      //fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      //fNameFrame->Reset();
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
         //comp->Resize();
         comp->Layout();
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

////////////////////////////////////////////////////////////////////////////////
/// Set matrix layout separator.

void TGuiBldHintsEditor::SetMatrixSep()
{
   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      //fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      //fNameFrame->Reset();
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
   fLayButton->SetDown(kTRUE);

   fHintsManager->fMatrix->fSep = sep;
   frame->SetLayoutBroken(kFALSE);

   if (!(frame->GetParent()->GetEditDisabled() & kEditDisableLayout)) {
      frame->Resize();
   } else {
      fHintsManager->fMatrix->Layout();
   }
   fClient->NeedRedraw(frame, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Apply matrix layout.

void TGuiBldHintsEditor::MatrixLayout()
{
   TGFrame *frame = fEditor->GetSelected();

   if (!frame) {
      //fNameFrame->Reset();
      return;
   }

   TGFrameElement *fe = frame->GetFrameElement();

   if (!fe) {
      //fNameFrame->Reset();
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

   fCbCenterX->SetEnabled(kFALSE);
   fCbCenterY->SetEnabled(kFALSE);
   fCbExpandX->SetEnabled(kFALSE);
   fCbExpandY->SetEnabled(kFALSE);

   fCbTop->SetEnabled(kFALSE);
   fCbRight->SetEnabled(kFALSE);
   fCbLeft->SetEnabled(kFALSE);
   fCbBottom->SetEnabled(kFALSE);

   fHintsManager->fPadTop = fPadTop->GetIntNumber();
   fHintsManager->fPadBottom = fPadBottom->GetIntNumber();
   fHintsManager->fPadLeft = fPadLeft->GetIntNumber();
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
      comp->Layout(); //resize?
   } else {
      fHintsManager->fMatrix->Layout();
   }
   fClient->NeedRedraw(comp, kTRUE);
}
