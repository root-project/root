// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveStraightLineSetEditor.h"
#include "TEveStraightLineSet.h"

#include "TVirtualPad.h"
#include "TColor.h"

#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

//______________________________________________________________________________
// TEveStraightLineSetEditor
//
// Editor for TEveStraightLineSet class.

ClassImp(TEveStraightLineSetEditor)

//______________________________________________________________________________
TEveStraightLineSetEditor::TEveStraightLineSetEditor(const TGWindow *p, Int_t width, Int_t height,
                                                     UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0)
                                                    // Initialize widget pointers to 0
{
   // Constructor.

   MakeTitle("TEveStraightLineSet");

   TGHorizontalFrame* frame = new TGHorizontalFrame(this);

   fRnrMarkers = new TGCheckButton(frame, "RnrMarkers");
   frame->AddFrame(fRnrMarkers, new TGLayoutHints(kLHintsLeft, 1, 2, 1, 1));
   fRnrMarkers->Connect
      ("Toggled(Bool_t)",
       "TEveStraightLineSetEditor", this, "DoRnrMarkers()");

   fRnrLines = new TGCheckButton(frame, "RnrLines");
   frame->AddFrame(fRnrLines, new TGLayoutHints(kLHintsLeft, 2, 1, 1, 1));
   fRnrLines->Connect
      ("Toggled(Bool_t)",
       "TEveStraightLineSetEditor", this, "DoRnrLines()");

   AddFrame(frame, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveStraightLineSet*>(obj);

   // Set values of widgets
   fRnrMarkers->SetState(fM->GetRnrMarkers() ? kButtonDown : kButtonUp);
   fRnrLines->SetState(fM->GetRnrLines() ? kButtonDown : kButtonUp);
}

/******************************************************************************/

// Implements callback/slot methods

//______________________________________________________________________________
void TEveStraightLineSetEditor::DoRnrMarkers()
{
   // Slot for RnrMarkers.

   fM->SetRnrMarkers(fRnrMarkers->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveStraightLineSetEditor::DoRnrLines()
{
   // Slot for RnrLines.

   fM->SetRnrLines(fRnrLines->IsOn());
   Update();
}
