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
#include "TEveGValuators.h"

#include "TGComboBox.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TG3DLine.h"

//______________________________________________________________________________
// GUI editor for TEveProjectionAxes.
//

ClassImp(TEveProjectionAxesEditor);

//______________________________________________________________________________
TEveProjectionAxesEditor::TEveProjectionAxesEditor(const TGWindow *p, Int_t width, Int_t height,
                                                   UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fStepMode(0),
   fNumTickMarks(0),

   fCenterFrame(0),
   fDrawCenter(0),
   fDrawOrigin(0)
{
   // Constructor.

   MakeTitle("TEveProjectionAxis");

   TGHorizontalFrame* f = new TGHorizontalFrame(this);
   TGLabel* lab = new TGLabel(f, "TickMark Step:");
   f->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 3, 1, 2));
   fStepMode = new TGComboBox(f, "Position");
   fStepMode->AddEntry("Value", 1);
   fStepMode->AddEntry("Position", 0);
   fStepMode->GetTextEntry()->SetToolTipText("Set tick-marks on equidistant values/screen position.");
   TGListBox* lb = fStepMode->GetListBox();
   lb->Resize(lb->GetWidth(), 2*18);
   fStepMode->Resize(80, 20);
   fStepMode->Connect("Selected(Int_t)", "TEveProjectionAxesEditor", this, "DoStepMode(Int_t)");
   f->AddFrame(fStepMode, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
   AddFrame(f);

   fNumTickMarks = new TEveGValuator(this, "TickMark Num:", 90, 0);
   fNumTickMarks->SetLabelWidth(87);
   fNumTickMarks->SetNELength(4);
   fNumTickMarks->Build();
   fNumTickMarks->SetLimits(3, 50);
   fNumTickMarks->Connect("ValueSet(Double_t)", "TEveProjectionAxesEditor", this, "DoNumTickMarks()");
   AddFrame(fNumTickMarks, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

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

   fStepMode->Select(fM->GetStepMode(), kFALSE);
   fNumTickMarks->SetValue(fM->GetNumTickMarks());

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
void TEveProjectionAxesEditor::DoStepMode(Int_t mode)
{
   // Slot for setting tick-mark step mode.

   TEveProjectionAxes::EMode em = (TEveProjectionAxes::EMode ) mode;
   fM->SetStepMode(em);
   Update();
}

//______________________________________________________________________________
void TEveProjectionAxesEditor::DoNumTickMarks()
{
   // Slot for setting number of tick-marks.
   fM->SetNumTickMarks((Int_t)fNumTickMarks->GetValue());
   Update();
}

