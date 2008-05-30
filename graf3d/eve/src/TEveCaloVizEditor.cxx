// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCaloVizEditor.h"
#include "TEveCalo.h"
#include "TEveGValuators.h"
#include "TEveRGBAPaletteEditor.h"

#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGDoubleSlider.h"
#include "TGNumberEntry.h"
#include "TG3DLine.h"

#include "TMathBase.h"
#include "TMath.h"

//______________________________________________________________________________
// GUI editor for TEveCaloEditor.
//

ClassImp(TEveCaloVizEditor);

//______________________________________________________________________________
TEveCaloVizEditor::TEveCaloVizEditor(const TGWindow *p, Int_t width, Int_t height,
                                     UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fEtaRng(0),
   fPhi(0),
   fPhiOffset(0),
   fTower(0),
   fPalette(0),
   fCellZScale(0)
{
   // Constructor.

   fTower = CreateEditorTabSubFrame("Towers");
   Int_t  labelW = 45;

   // eta
   fEtaRng = new TEveGDoubleValuator(fTower,"Eta rng:", 40, 0);
   fEtaRng->SetNELength(6);
   fEtaRng->SetLabelWidth(labelW);
   fEtaRng->Build();
   fEtaRng->GetSlider()->SetWidth(195);
   fEtaRng->SetLimits(-5.5, 5.5, TGNumberFormat::kNESRealTwo);
   fEtaRng->Connect("ValueSet()", "TEveCaloVizEditor", this, "DoEtaRange()");
   fTower->AddFrame(fEtaRng, new TGLayoutHints(kLHintsTop, 1, 1, 4, 5));

   // phi
   fPhi = new TEveGValuator(fTower, "Phi:", 90, 0);
   fPhi->SetLabelWidth(labelW);
   fPhi->SetNELength(6);
   fPhi->Build();
   fPhi->SetLimits(-180, 180);
   fPhi->Connect("ValueSet(Double_t)", "TEveCaloVizEditor", this, "DoPhi()");
   fTower->AddFrame(fPhi, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fPhiOffset = new TEveGValuator(fTower, "PhiOff:", 90, 0);
   fPhiOffset->SetLabelWidth(labelW);
   fPhiOffset->SetNELength(6);
   fPhiOffset->Build();
   fPhiOffset->SetLimits(0, 180);
   fPhiOffset->Connect("ValueSet(Double_t)", "TEveCaloVizEditor", this, "DoPhi()");
   fTower->AddFrame(fPhiOffset, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fCellZScale = new TEveGValuator(fTower, "ZScale:", 90, 0);
   fCellZScale->SetLabelWidth(labelW);
   fCellZScale->SetNELength(6);
   fCellZScale->Build();
   fCellZScale->SetLimits(0, 5, 100, TGNumberFormat::kNESRealTwo);
   fCellZScale->Connect("ValueSet(Double_t)", "TEveCaloVizEditor", this, "DoCellZScale()");
   fTower->AddFrame(fCellZScale, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   TGHorizontalFrame *title2 = new TGHorizontalFrame(fTower, 145, 10, kLHintsExpandX| kFixedWidth);
   title2->AddFrame(new TGLabel(title2, "Palette Controls"),
                    new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   title2->AddFrame(new TGHorizontal3DLine(title2),
                    new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   fTower->AddFrame(title2, new TGLayoutHints(kLHintsTop, 0, 0, 5, 0));

   fPalette = new TEveRGBAPaletteSubEditor(fTower);
   fTower->AddFrame(fPalette, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 0, 0, 0));
   fPalette->Connect("Changed()", "TEveCaloVizEditor", this, "DoPalette()");
}

//______________________________________________________________________________
void TEveCaloVizEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveCaloViz*>(obj);

   Double_t min, max;
   fM->GetData()->GetEtaLimits(min, max);
   fEtaRng->SetLimits((Float_t)min, (Float_t)max);
   fEtaRng->SetValues(fM->fEtaMin, fM->fEtaMax);

   fPhi->SetValue(fM->fPhi*TMath::RadToDeg());
   fPhiOffset->SetValue(fM->fPhiOffset*TMath::RadToDeg());

   fPalette->SetModel(fM->fPalette);

   fCellZScale->SetValue(fM->fCellZScale);
}

//______________________________________________________________________________
void TEveCaloVizEditor::DoEtaRange()
{
   // Slot for setting eta range.

   fM->SetEta(fEtaRng->GetMin(), fEtaRng->GetMax());
   Update();
}

//______________________________________________________________________________
void TEveCaloVizEditor::DoPhi()
{
  // Slot for setting phi range.

   fM->SetPhiWithRng(fPhi->GetValue()*TMath::DegToRad(), fPhiOffset->GetValue()*TMath::DegToRad());
   Update();
}

//______________________________________________________________________________
void TEveCaloVizEditor::DoCellZScale()
{
  // Slot for setting tower height.

   fM->SetCellZScale(fCellZScale->GetValue());
   Update();
}

//______________________________________________________________________________
void TEveCaloVizEditor::DoPalette()
{
   // Slot for palette changed.

   fM->InvalidateCache();
   Update();
}
