// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTrackEditor.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "TEveTrackPropagatorEditor.h"
#include "TEveManager.h"

#include "TEveGValuators.h"

#include "TVirtualPad.h"
#include "TColor.h"

#include "TGedEditor.h"
#include "TAttMarkerEditor.h"
#include "TGLabel.h"
#include "TG3DLine.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

//______________________________________________________________________________
// TEveTrackEditor
//
// Editor for TEveTrack class.

ClassImp(TEveTrackEditor);

//______________________________________________________________________________
TEveTrackEditor::TEveTrackEditor(const TGWindow *p, Int_t width, Int_t height,
                                 UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),
   fRSEditor(0)
{
   // Constructor.

   MakeTitle("TEveTrack");

   TGHorizontalFrame* f = new TGHorizontalFrame(this);

   fRSEditor =  new TGTextButton(f, "Edit Propagator");
   fRSEditor->Connect("Clicked()", "TEveTrackEditor", this, "DoEditPropagator()");
   f->AddFrame(fRSEditor, new TGLayoutHints(kLHintsLeft, 2, 1, 4, 4));

   AddFrame(f, new TGLayoutHints(kLHintsTop, 0,0,2,1));
}

//______________________________________________________________________________
void TEveTrackEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveTrack*>(obj);
}

//______________________________________________________________________________
void TEveTrackEditor::DoEditPropagator()
{
   // Slot for EditPropagator.

   fGedEditor->SetModel(fGedEditor->GetPad(), fM->GetPropagator(), kButton1Down);
}


//______________________________________________________________________________
// TEveTrackListEditor
//
// Editor for TEveTrackList class.

ClassImp(TEveTrackListEditor);

//______________________________________________________________________________
TEveTrackListEditor::TEveTrackListEditor(const TGWindow *p,
                                         Int_t width, Int_t height,
                                         UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),

   fTC         (0),
   fPtRange    (0),
   fPRange     (0),
   fRSSubEditor(0)
{
   // Constructor.

   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);

      fRnrPoints = new TGCheckButton(f, "Draw Marker");
      f->AddFrame(fRnrPoints, new TGLayoutHints(kLHintsLeft, 2,1,0,0));
      fRnrPoints->Connect("Toggled(Bool_t)"," TEveTrackListEditor", this, "DoRnrPoints()");

      fRnrLine  = new TGCheckButton(f, "Draw Line");
      f->AddFrame(fRnrLine, new TGLayoutHints(kLHintsLeft, 1,2,0,0));
      fRnrLine->Connect("Toggled(Bool_t)", "TEveTrackListEditor", this, "DoRnrLine()");

      AddFrame(f, new TGLayoutHints(kLHintsTop, 0,0,2,1));
   }
   {  // --- Selectors
      Int_t labelW = 51;
      Int_t dbW    = 210;

      fPtRange = new TEveGDoubleValuator(this,"Pt rng:", 40, 0);
      fPtRange->SetNELength(6);
      fPtRange->SetLabelWidth(labelW);
      fPtRange->Build();
      fPtRange->GetSlider()->SetWidth(dbW);
      fPtRange->SetLimits(0, 10, TGNumberFormat::kNESRealTwo);
      fPtRange->Connect("ValueSet()",
                        "TEveTrackListEditor", this, "DoPtRange()");
      AddFrame(fPtRange, new TGLayoutHints(kLHintsTop, 1, 1, 4, 1));

      fPRange = new TEveGDoubleValuator(this,"P rng:", 40, 0);
      fPRange->SetNELength(6);
      fPRange->SetLabelWidth(labelW);
      fPRange->Build();
      fPRange->GetSlider()->SetWidth(dbW);
      fPRange->SetLimits(0, 100, TGNumberFormat::kNESRealTwo);
      fPRange->Connect("ValueSet()",
                       "TEveTrackListEditor", this, "DoPRange()");
      AddFrame(fPRange, new TGLayoutHints(kLHintsTop, 1, 1, 4, 1));
   }

   MakeTitle("RenderStyle");
   fRSSubEditor = new TEveTrackPropagatorSubEditor(this);
   fRSSubEditor->Connect("Changed()", "TEveTrackListEditor", this, "Update()");
   AddFrame(fRSSubEditor, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,0,0));
   CreateRefsTab();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackListEditor::CreateRefsTab()
{
   // Create tab for control of path-mark display.

   fRefs = CreateEditorTabSubFrame("Refs");

   TGCompositeFrame *title1 = new TGCompositeFrame(fRefs, 145, 10,
                                                   kHorizontalFrame |
                                                   kLHintsExpandX   |
                                                   kFixedWidth      |
                                                   kOwnBackground);
   title1->AddFrame(new TGLabel(title1, "PathMarks"),
                    new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title1->AddFrame(new TGHorizontal3DLine(title1),
                    new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fRefs->AddFrame(title1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   // path marks
   fRSSubEditor->CreateRefsContainer(fRefs);
   fRSSubEditor->fPMAtt->SetGedEditor((TGedEditor*)gEve->GetEditor());
   fRSSubEditor->fFVAtt->SetGedEditor((TGedEditor*)gEve->GetEditor());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackListEditor::SetModel(TObject* obj)
{
   // Set model object.

   fTC = dynamic_cast<TEveTrackList*>(obj);
   fRnrLine  ->SetState(fTC->GetRnrLine()   ? kButtonDown : kButtonUp);
   fRnrPoints->SetState(fTC->GetRnrPoints() ? kButtonDown : kButtonUp);

   Float_t llim;
   fPtRange->SetValues(fTC->fMinPt, fTC->fMaxPt);
   llim = fTC->fLimPt > 1 ? TMath::Log10(fTC->fLimPt) : 0;
   fPtRange->SetLimits(0, fTC->fLimPt, llim < 2 ? TGNumberFormat::kNESRealTwo : (llim < 3 ? TGNumberFormat::kNESRealOne : TGNumberFormat::kNESInteger));
   fPRange ->SetValues(fTC->fMinP, fTC->fMaxP);
   llim = fTC->fLimP > 1 ? TMath::Log10(fTC->fLimP) : 0;
   fPRange ->SetLimits(0, fTC->fLimP, llim < 2 ? TGNumberFormat::kNESRealTwo : (llim < 3 ? TGNumberFormat::kNESRealOne : TGNumberFormat::kNESInteger));

   fRSSubEditor->SetModel(fTC->GetPropagator());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackListEditor::DoRnrLine()
{
   // Slot for RnrLine.

   fTC->SetRnrLine(fRnrLine->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveTrackListEditor::DoRnrPoints()
{
   // Slot for RnrPoints.

   fTC->SetRnrPoints(fRnrPoints->IsOn());
   Update();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrackListEditor::DoPtRange()
{
   // Slot for PtRange.

   fTC->SelectByPt(fPtRange->GetMin(), fPtRange->GetMax());
   Update();
}

//______________________________________________________________________________
void TEveTrackListEditor::DoPRange()
{
   // Slot for PRange.

   fTC->SelectByP(fPRange->GetMin(), fPRange->GetMax());
   Update();
}

