// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionAxesEditor.h"
#include "TEveProjectionAxes.h"

#include "TGNumberEntry.h"
#include "TGComboBox.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TG3DLine.h"
#include "TGNumberEntry.h"

//______________________________________________________________________________
// GUI editor for TEveProjectionAxes.
//

ClassImp(TEveProjectionAxesEditor);

//______________________________________________________________________________
TEveProjectionAxesEditor::TEveProjectionAxesEditor(const TGWindow *p, Int_t width, Int_t height,
                                                   UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fSplitMode(0),
   fSplitLevel(0),

   fCenterFrame(0),
   fDrawCenter(0),
   fDrawOrigin(0)
{
   // Constructor.

   MakeTitle("TEveProjectionAxis");
   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);
      TGLabel* lab = new TGLabel(f, "StepMode");
      f->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 6, 1, 2));
      fSplitMode = new TGComboBox(f, "Position");
      fSplitMode->AddEntry("Value", 1);
      fSplitMode->AddEntry("Position", 0);
      fSplitMode->GetTextEntry()->SetToolTipText("Set tick-marks on equidistant values/screen position.");
      TGListBox* lb = fSplitMode->GetListBox();
      lb->Resize(lb->GetWidth(), 2*18);
      fSplitMode->Resize(80, 20);
      fSplitMode->Connect("Selected(Int_t)", "TEveProjectionAxesEditor",
                       this, "DoSplitMode(Int_t)");
      f->AddFrame(fSplitMode, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
      AddFrame(f);
   }
   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);
      TGLabel* lab = new TGLabel(f, "SplitLevel");
      f->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 8, 1, 2));

      fSplitLevel = new TGNumberEntry(f, 0, 3, -1,TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                   TGNumberFormat::kNELLimitMinMax, 0, 7);
      fSplitLevel->GetNumberEntry()->SetToolTipText("Number of tick-marks TMath::Power(2, level).");
      fSplitLevel->Connect("ValueSet(Long_t)", "TEveProjectionAxesEditor", this, "DoSplitLevel()");
      f->AddFrame(fSplitLevel, new TGLayoutHints(kLHintsTop, 1, 1, 1, 2));
      AddFrame(f, new TGLayoutHints(kLHintsTop, 0, 0, 0, 3) );
   }

   /**************************************************************************/
   // center tab
   fCenterFrame = CreateEditorTabSubFrame("Center");

   TGCompositeFrame *title1 = new TGCompositeFrame(fCenterFrame, 180, 10,
                                                   kHorizontalFrame |
                                                   kLHintsExpandX   |
                                                   kFixedWidth      |
                                                   kOwnBackground);
   title1->AddFrame(new TGLabel(title1, "Distortion Center"),
                    new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title1->AddFrame(new TGHorizontal3DLine(title1),
                    new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fCenterFrame->AddFrame(title1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));


   {

      TGHorizontalFrame* hf1 = new TGHorizontalFrame(fCenterFrame);

      fDrawOrigin = new TGCheckButton(hf1, "DrawOrigin");
      hf1->AddFrame(fDrawOrigin, new TGLayoutHints(kLHintsLeft, 2,1,0,4));
      fDrawOrigin->Connect("Toggled(Bool_t)"," TEveProjectionAxesEditor", this, "DoDrawOrigin()");


      fDrawCenter = new TGCheckButton(hf1, "DrawCenter");
      hf1->AddFrame(fDrawCenter, new TGLayoutHints(kLHintsLeft, 2,1,0,4));
      fDrawCenter->Connect("Toggled(Bool_t)"," TEveProjectionAxesEditor", this, "DoDrawCenter()");

      fCenterFrame->AddFrame(hf1, new TGLayoutHints(kLHintsTop, 0,0,0,0));

   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveProjectionAxesEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveProjectionAxes*>(obj);

   fSplitMode->Select(fM->GetSplitMode(), kFALSE);
   fSplitLevel->SetNumber(fM->GetSplitLevel());

   fDrawCenter->SetState(fM->GetDrawCenter()  ? kButtonDown : kButtonUp);
   fDrawOrigin->SetState(fM->GetDrawOrigin()  ? kButtonDown : kButtonUp);

}

//______________________________________________________________________________
void TEveProjectionAxesEditor::DoDrawOrigin()
{
   // Slot for setting draw of origin.

   fM->SetDrawOrigin(fDrawOrigin->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveProjectionAxesEditor::DoDrawCenter()
{
   // Slot for setting draw of center.

   fM->SetDrawCenter(fDrawCenter->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveProjectionAxesEditor::DoSplitMode(Int_t mode)
{
   // Slot for setting split info mode.

   TEveProjectionAxes::EMode em = (TEveProjectionAxes::EMode ) mode;
   fM->SetSplitMode(em);
   Update();
}

//______________________________________________________________________________
void TEveProjectionAxesEditor::DoSplitLevel()
{
   // Slot for setting tick-mark density.

   fM->SetSplitLevel((Int_t)fSplitLevel->GetNumber());
   Update();
}

