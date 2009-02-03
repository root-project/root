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

   fLabMode(0),
   fAxesMode(0),

   fCenterFrame(0),
   fDrawCenter(0),
   fDrawOrigin(0)
{
   // Constructor.

   MakeTitle("TEveProjectionAxis");
   Int_t labw=52;

   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);

      TGCompositeFrame *labfr = new TGHorizontalFrame(f, labw, 20, kFixedSize);
      labfr->AddFrame(new TGLabel(labfr, "Labels:") , new TGLayoutHints(kLHintsLeft  | kLHintsBottom));
      f->AddFrame(labfr, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 0, 1, 1, 1));

      fLabMode = new TGComboBox(f, "Position");
      fLabMode->AddEntry("Value", 1);
      fLabMode->AddEntry("Position", 0);
      fLabMode->GetTextEntry()->SetToolTipText("Set tick-marks on equidistant values/screen position.");
      TGListBox* lb = fLabMode->GetListBox();
      lb->Resize(lb->GetWidth(), 2*18);
      fLabMode->Resize(80, 20);
      fLabMode->Connect("Selected(Int_t)", "TEveProjectionAxesEditor", this, "DoLabMode(Int_t)");
      f->AddFrame(fLabMode, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
      AddFrame(f);
   }

   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);
      TGCompositeFrame *labfr = new TGHorizontalFrame(f, labw, 20, kFixedSize);
      labfr->AddFrame(new TGLabel(labfr, "Axes:") , new TGLayoutHints(kLHintsLeft  | kLHintsBottom));
      f->AddFrame(labfr, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 0, 1, 1, 1));


      fAxesMode = new TGComboBox(f, "All");
      fAxesMode->AddEntry("Horizontal", TEveProjectionAxes::kHorizontal);
      fAxesMode->AddEntry("Vertical",TEveProjectionAxes::kVertical);
      fAxesMode->AddEntry("All", TEveProjectionAxes::kAll);
      TGListBox* lb = fAxesMode->GetListBox();
      lb->Resize(lb->GetWidth(), 2*18);
      fAxesMode->Resize(80, 20);
      fAxesMode->Connect("Selected(Int_t)", "TEveProjectionAxesEditor", this, "DoAxesMode(Int_t)");
      f->AddFrame(fAxesMode, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
      AddFrame(f);
   }

   //______________________________________________________________________________

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

   fLabMode->Select(fM->GetLabMode(), kFALSE);
   fAxesMode->Select(fM->GetAxesMode(), kFALSE);
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
void TEveProjectionAxesEditor::DoLabMode(Int_t mode)
{
   // Slot for setting tick-mark step mode.

   TEveProjectionAxes::ELabMode em = (TEveProjectionAxes::ELabMode ) mode;
   fM->SetLabMode(em);
   Update();
}

//______________________________________________________________________________
void TEveProjectionAxesEditor::DoAxesMode(Int_t mode)
{
   // Slot for setting number of axes.

   TEveProjectionAxes::EAxesMode em = (TEveProjectionAxes::EAxesMode ) mode;
   fM->SetAxesMode(em);
   Update();
}
