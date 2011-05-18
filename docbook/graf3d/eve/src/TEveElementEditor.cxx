// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveElementEditor.h"
#include "TEveElement.h"
#include "TEveTransEditor.h"

#include "TVirtualPad.h"
#include "TColor.h"

#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

//==============================================================================
//==============================================================================
// TEveElementEditor
//==============================================================================

//______________________________________________________________________________
//
// Editor for TEveElement class.

ClassImp(TEveElementEditor);

//______________________________________________________________________________
TEveElementEditor::TEveElementEditor(const TGWindow *p,
                                     Int_t width, Int_t height,
                                     UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),

   fRE           (0),
   fHFrame       (0),
   fPreLabel     (0),
   fRnrSelf      (0),
   fRnrChildren  (0),
   fRnrState     (0),
   fMainColor    (0),
   fTransparency (0),
   fTrans        (0)
{
   // Constructor.

   MakeTitle("TEveElement");
   fPriority = 0;

   fHFrame = new TGHorizontalFrame(this);

   fPreLabel = new TGLabel(fHFrame, "Show:");
   fHFrame->AddFrame(fPreLabel, new TGLayoutHints(kLHintsLeft, 1, 6, 2, 0));
   fRnrSelf = new TGCheckButton(fHFrame, "Self");
   fHFrame->AddFrame(fRnrSelf, new TGLayoutHints(kLHintsLeft, 0, 2, 1, 1));
   fRnrSelf->Connect
      ("Toggled(Bool_t)",
       "TEveElementEditor", this, "DoRnrSelf()");

   fRnrChildren = new TGCheckButton(fHFrame, "Children");
   fHFrame->AddFrame(fRnrChildren, new TGLayoutHints(kLHintsLeft, 0, 1, 1, 1));
   fRnrChildren->Connect
      ("Toggled(Bool_t)",
       "TEveElementEditor", this, "DoRnrChildren()");

   fRnrState = new TGCheckButton(fHFrame, "");
   fHFrame->AddFrame(fRnrState, new TGLayoutHints(kLHintsLeft, 1, 2, 1, 1));
   fRnrState->Connect
      ("Toggled(Bool_t)",
       "TEveElementEditor", this, "DoRnrState()");

   fMainColor = new TGColorSelect(fHFrame, 0, -1);
   fHFrame->AddFrame(fMainColor, new TGLayoutHints(kLHintsLeft, 2, 0, -2, 0));
   fMainColor->Connect
      ("ColorSelected(Pixel_t)",
       "TEveElementEditor", this, "DoMainColor(Pixel_t)");

   fTransparency = new TGNumberEntry(fHFrame, 0., 2, -1,
                                     TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                     TGNumberFormat::kNELLimitMinMax, 0, 100);
   fTransparency->SetHeight(18);
   fTransparency->GetNumberEntry()->SetToolTipText("Transparency: 0 is opaque, 100 fully transparent.");
   fHFrame->AddFrame(fTransparency, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   fTransparency->Connect
      ("ValueSet(Long_t)",
       "TEveElementEditor", this, "DoTransparency()");

   AddFrame(fHFrame, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));

   fTrans = new TEveTransSubEditor(this);
   fTrans->Connect("UseTrans()",     "TEveElementEditor", this, "Update()");
   fTrans->Connect("TransChanged()", "TEveElementEditor", this, "Update()");
   AddFrame(fTrans, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 0, 0));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElementEditor::SetModel(TObject* obj)
{
   // Set model object.

   fRE = dynamic_cast<TEveElement*>(obj);

   fPreLabel   ->UnmapWindow();
   fRnrSelf    ->UnmapWindow();
   fRnrChildren->UnmapWindow();
   fRnrState   ->UnmapWindow();
   if (fRE->CanEditElement()) {
      fPreLabel->MapWindow();
      if (fRE->SingleRnrState()) {
         fRnrState->SetState(fRE->GetRnrState() ? kButtonDown : kButtonUp);
         fRnrState->MapWindow();
      } else {
         fRnrSelf->SetState(fRE->GetRnrSelf() ? kButtonDown : kButtonUp);
         fRnrChildren->SetState(fRE->GetRnrChildren() ? kButtonDown : kButtonUp);
         fRnrSelf->MapWindow();
         fRnrChildren->MapWindow();
      }
   }

   if (fRE->CanEditMainColor()) {
      fMainColor->SetColor(TColor::Number2Pixel(fRE->GetMainColor()), kFALSE);
      fMainColor->MapWindow();
   } else {
      fMainColor->UnmapWindow();
   }
   if (fRE->CanEditMainTransparency()) {
      fTransparency->SetNumber(fRE->GetMainTransparency());
      fTransparency->MapWindow();
   } else {
      fTransparency->UnmapWindow();
   }
   if (fRE->CanEditMainTrans()) {
      fTrans->SetModel(fRE->PtrMainTrans());
      fTrans->MapWindow();
   } else {
      fTrans->UnmapWindow();
   }

   fHFrame->Layout();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElementEditor::DoRnrSelf()
{
   // Slot for RnrSelf.

   fRE->SetRnrSelf(fRnrSelf->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveElementEditor::DoRnrChildren()
{
   // Slot for RnrChildren.

   fRE->SetRnrChildren(fRnrChildren->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveElementEditor::DoRnrState()
{
   // Slot for RnrState.

   fRE->SetRnrState(fRnrState->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveElementEditor::DoMainColor(Pixel_t color)
{
   // Slot for MainColor.

   fRE->SetMainColorPixel(color);
   Update();
}

//______________________________________________________________________________
void TEveElementEditor::DoTransparency()
{
   // Slot for Transparency.

   fRE->SetMainTransparency((Char_t)(fTransparency->GetNumber()));
   Update();
}
