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

//______________________________________________________________________________
// TEveElementEditor
//
// Editor for TEveElement class.

ClassImp(TEveElementEditor)

//______________________________________________________________________________
TEveElementEditor::TEveElementEditor(const TGWindow *p,
                                     Int_t width, Int_t height,
                                     UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),

   fRE           (0),
   fHFrame       (0),
   fRnrSelf      (0),
   fRnrChildren  (0),
   fMainColor    (0),
   fTransparency (0),
   fHMTrans      (0)
{
   // Constructor.

   MakeTitle("TEveElement");
   fPriority = 0;

   fHFrame = new TGHorizontalFrame(this);

   fRnrSelf = new TGCheckButton(fHFrame, "RnrSelf");
   fHFrame->AddFrame(fRnrSelf, new TGLayoutHints(kLHintsLeft, 1, 2, 1, 1));
   fRnrSelf->Connect
      ("Toggled(Bool_t)",
       "TEveElementEditor", this, "DoRnrSelf()");

   fRnrChildren = new TGCheckButton(fHFrame, "RnrChildren");
   fHFrame->AddFrame(fRnrChildren, new TGLayoutHints(kLHintsLeft, 2, 1, 1, 1));
   fRnrChildren->Connect
      ("Toggled(Bool_t)",
       "TEveElementEditor", this, "DoRnrChildren()");

   fMainColor = new TGColorSelect(fHFrame, 0, -1);
   fHFrame->AddFrame(fMainColor, new TGLayoutHints(kLHintsLeft, 2, 0, 1, 1));
   fMainColor->Connect
      ("ColorSelected(Pixel_t)",
       "TEveElementEditor", this, "DoMainColor(Pixel_t)");

   fTransparency = new TGNumberEntry(fHFrame, 0., 2, -1,
                                     TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                     TGNumberFormat::kNELLimitMinMax, 0, 100);
   fTransparency->SetHeight(18);
   fTransparency->GetNumberEntry()->SetToolTipText("Transparency: 0 is opaque, 100 fully transparent.");
   fHFrame->AddFrame(fTransparency, new TGLayoutHints(kLHintsLeft, 0, 0, 2, 0));
   fTransparency->Connect
      ("ValueSet(Long_t)",
       "TEveElementEditor", this, "DoTransparency()");

   AddFrame(fHFrame, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));

   fHMTrans = new TEveTransSubEditor(this);
   fHMTrans->Connect("UseTrans()",     "TEveElementEditor", this, "Update()");
   fHMTrans->Connect("TransChanged()", "TEveElementEditor", this, "Update()");
   AddFrame(fHMTrans, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 0, 0));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElementEditor::SetModel(TObject* obj)
{
   // Set model object.

   fRE = dynamic_cast<TEveElement*>(obj);

   if (fRE->CanEditRnrElement()) {
      fRnrSelf->SetState(fRE->GetRnrSelf() ? kButtonDown : kButtonUp);
      fRnrChildren->SetState(fRE->GetRnrChildren() ? kButtonDown : kButtonUp);
      fRnrSelf->MapWindow();
      fRnrChildren->MapWindow();
   } else {
      fRnrSelf->UnmapWindow();
      fRnrChildren->UnmapWindow();
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
   if (fRE->CanEditMainHMTrans()) {
      fHMTrans->SetModel(fRE->PtrMainHMTrans());
      fHMTrans->MapWindow();
   } else {
      fHMTrans->UnmapWindow();
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
void TEveElementEditor::DoMainColor(Pixel_t color)
{
   // Slot for MainColor.

   fRE->SetMainColor(color);
   Update();
}

//______________________________________________________________________________
void TEveElementEditor::DoTransparency()
{
   // Slot for Transparency.

   fRE->SetMainTransparency((UChar_t)(fTransparency->GetNumber()));
   Update();
}
