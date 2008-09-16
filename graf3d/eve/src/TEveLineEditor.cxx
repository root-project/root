// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveLineEditor.h"
#include "TEveLine.h"

#include "TVirtualPad.h"
#include "TColor.h"

#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

//==============================================================================
//==============================================================================
// TEveLineEditor
//==============================================================================

//______________________________________________________________________________
//
// Editor for TEveLine class.

ClassImp(TEveLineEditor);

//______________________________________________________________________________
TEveLineEditor::TEveLineEditor(const TGWindow *p, Int_t width, Int_t height,
                               UInt_t options, Pixel_t back) :
   TGedFrame  (p, width, height, options | kVerticalFrame, back),
   fM         (0),
   fRnrLine   (0),
   fRnrPoints (0),
   fSmooth    (0)
{
   // Constructor.

   fPriority = 20;
   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);

      fRnrPoints = new TGCheckButton(f, "Draw Marker");
      f->AddFrame(fRnrPoints, new TGLayoutHints(kLHintsLeft, 2,1,0,0));
      fRnrPoints->Connect("Toggled(Bool_t)"," TEveLineEditor", this, "DoRnrPoints()");

      fRnrLine  = new TGCheckButton(f, "Draw Line");
      f->AddFrame(fRnrLine, new TGLayoutHints(kLHintsLeft, 1,2,0,0));
      fRnrLine->Connect("Toggled(Bool_t)", "TEveLineEditor", this, "DoRnrLine()");

      AddFrame(f, new TGLayoutHints(kLHintsTop, 0,0,2,1));
   }
   fSmooth  = new TGCheckButton(this, "Smooth line");
   AddFrame(fSmooth, new TGLayoutHints(kLHintsNormal, 2,1,2,1));
   fSmooth->Connect("Toggled(Bool_t)", "TEveLineEditor", this, "DoSmooth()");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveLineEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveLine*>(obj);

   fRnrLine  ->SetState(fM->fRnrLine  ? kButtonDown : kButtonUp);
   fRnrPoints->SetState(fM->fRnrPoints ? kButtonDown : kButtonUp);
   fSmooth->SetState(fM->fSmooth ? kButtonDown : kButtonUp);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveLineEditor::DoRnrLine()
{
   // Slot for RnrLine.

   fM->SetRnrLine(fRnrLine->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveLineEditor::DoRnrPoints()
{
   // Slot for RnrPoints.

   fM->SetRnrPoints(fRnrPoints->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveLineEditor::DoSmooth()
{
   // Slot for anti-alias.

   fM->SetSmooth(fSmooth->IsOn());
   Update();
}
