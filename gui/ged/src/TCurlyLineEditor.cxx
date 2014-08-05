// @(#)root/ged:$Id$
// Author: Ilka Antcheva, Otto Schaile 15/12/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TCurlyLineEditor                                                    //
//                                                                      //
//  Implements GUI for editing CurlyLine attributes: shape, size, angle.//                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TCurlyLineEditor.gif">
*/
//End_Html


#include "TCurlyLineEditor.h"
#include "TGedEditor.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGButton.h"
#include "TCurlyLine.h"
#include <iostream>

ClassImp(TCurlyLineEditor)

enum ECurlyLineWid {
   kCRLL_AMPL,
   kCRLL_WAVE,
   kCRLL_ISW,
   kCRLL_STRX,
   kCRLL_STRY,
   kCRLL_ENDX,
   kCRLL_ENDY
};

//______________________________________________________________________________
TCurlyLineEditor::TCurlyLineEditor(const TGWindow *p, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of CurlyLine GUI.

   fCurlyLine = 0;

   MakeTitle("Curly Line");

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGCompositeFrame *f3a = new TGCompositeFrame(f3, 80, 20);
   f3->AddFrame(f3a, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGLabel *fAmplitudeLabel = new TGLabel(f3a, "Amplitude:");
   f3a->AddFrame(fAmplitudeLabel, new TGLayoutHints(kLHintsNormal, 3, 0, 5, 5));

   TGLabel *fWaveLengthLabel = new TGLabel(f3a, "Wavelgth:");
   f3a->AddFrame(fWaveLengthLabel, new TGLayoutHints(kLHintsNormal, 3, 0, 5, 5));

   TGCompositeFrame *f3b = new TGCompositeFrame(f3, 80, 20);
   f3->AddFrame(f3b, new TGLayoutHints(kLHintsNormal, 0, 0, 0, 0));

   fAmplitudeEntry = new TGNumberEntry(f3b, 0.005, 7, kCRLL_AMPL,
                                       TGNumberFormat::kNESRealThree,
                                       TGNumberFormat::kNEANonNegative,
                                       TGNumberFormat::kNELLimitMinMax, 0.005, 0.3);
   fAmplitudeEntry->GetNumberEntry()->SetToolTipText("Set amplitude in percent of the pad height.");
   f3b->AddFrame(fAmplitudeEntry, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));

   fWaveLengthEntry = new TGNumberEntry(f3b, 0.005, 7, kCRLL_WAVE,
                                        TGNumberFormat::kNESRealThree,
                                        TGNumberFormat::kNEANonNegative,
                                        TGNumberFormat::kNELLimitMinMax, 0.005, 0.3);
   fWaveLengthEntry->GetNumberEntry()->SetToolTipText("Set wavelength in percent of the pad height.");
   fWaveLengthEntry->Associate(this);
   f3b->AddFrame(fWaveLengthEntry, new TGLayoutHints(kLHintsLeft, 4, 1, 3, 1));

   fIsWavy = new TGCheckButton(this, "Gluon (Gamma)", kCRLL_ISW);
   fIsWavy->SetToolTipText("Toggle between wavy line (Gluon) if selected; curly line (Gamma) otherwise.");
   AddFrame(fIsWavy, new TGLayoutHints(kLHintsLeft, 5, 1, 5, 8));

   fStartXFrame = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(fStartXFrame, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGCompositeFrame *f4a = new TGCompositeFrame(fStartXFrame, 80, 20);
   fStartXFrame->AddFrame(f4a, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGLabel *fStartXLabel = new TGLabel(f4a, "Start X:");
   f4a->AddFrame(fStartXLabel, new TGLayoutHints(kLHintsNormal, 21, 0, 5, 5));

   TGLabel *fStartYLabel = new TGLabel(f4a, "Y:");
   f4a->AddFrame(fStartYLabel, new TGLayoutHints(kLHintsNormal, 50, 0, 5, 5));

   TGLabel *fEndXLabel = new TGLabel(f4a, "End X:");
   f4a->AddFrame(fEndXLabel, new TGLayoutHints(kLHintsNormal, 24, 0, 5, 5));

   TGLabel *fEndYLabel = new TGLabel(f4a, "Y:");
   f4a->AddFrame(fEndYLabel, new TGLayoutHints(kLHintsNormal, 51, 0, 5, 1));

   TGCompositeFrame *f4b = new TGCompositeFrame(fStartXFrame, 80, 20);
   fStartXFrame->AddFrame(f4b, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   fStartXEntry = new TGNumberEntry(f4b, 0.0, 7, kCRLL_STRX,
                                    TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELNoLimits);
   fStartXEntry->GetNumberEntry()->SetToolTipText("Set start point X ccordinate of curly line.");
   f4b->AddFrame(fStartXEntry, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   fStartYEntry = new TGNumberEntry(f4b, 0.0, 7, kCRLL_STRY,
                                    TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELNoLimits);
   fStartYEntry->GetNumberEntry()->SetToolTipText("Set start point Y coordinate of curly line.");
   f4b->AddFrame(fStartYEntry, new TGLayoutHints(kLHintsLeft, 1, 1, 3, 1));

   fEndXEntry = new TGNumberEntry(f4b, 0.0, 7, kCRLL_ENDX,
                                  TGNumberFormat::kNESRealThree,
                                  TGNumberFormat::kNEANonNegative,
                                  TGNumberFormat::kNELNoLimits);
   fEndXEntry->GetNumberEntry()->SetToolTipText("Set end point X coordinate of curly line.");
   f4b->AddFrame(fEndXEntry, new TGLayoutHints(kLHintsLeft, 1, 1, 3, 1));

   fEndYEntry = new TGNumberEntry(f4b, 0.0, 7, kCRLL_ENDY,
                                  TGNumberFormat::kNESRealThree,
                                  TGNumberFormat::kNEANonNegative,
                                  TGNumberFormat::kNELNoLimits);
   fEndYEntry->GetNumberEntry()->SetToolTipText("Set end point Y coordinate of curly line.");
   f4b->AddFrame(fEndYEntry, new TGLayoutHints(kLHintsLeft, 1, 1, 3, 1));

}

//______________________________________________________________________________
TCurlyLineEditor::~TCurlyLineEditor()
{
   // Destructor of CurlyLine editor.
}

//______________________________________________________________________________
void TCurlyLineEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fStartXEntry->Connect("ValueSet(Long_t)", "TCurlyLineEditor", this, "DoStartXY()");
   (fStartXEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyLineEditor", this, "DoStartXY()");
   fStartYEntry->Connect("ValueSet(Long_t)", "TCurlyLineEditor", this, "DoStartXY()");
   (fStartYEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyLineEditor", this, "DoStartXY()");
   fEndXEntry->Connect("ValueSet(Long_t)", "TCurlyLineEditor", this, "DoEndXY()");
   (fEndXEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyLineEditor", this, "DoEndXY()");
   fEndYEntry->Connect("ValueSet(Long_t)", "TCurlyLineEditor", this, "DoEndXY()");
   (fEndYEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyLineEditor", this, "DoEndXY()");
   fAmplitudeEntry->Connect("ValueSet(Long_t)", "TCurlyLineEditor", this, "DoAmplitude()");
   (fAmplitudeEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyLineEditor", this, "DoAmplitude()");
   fWaveLengthEntry->Connect("ValueSet(Long_t)", "TCurlyLineEditor", this, "DoWaveLength()");
   (fWaveLengthEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyLineEditor", this, "DoWaveLength()");
   fIsWavy->Connect("Clicked()", "TCurlyLineEditor", this, "DoWavy()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TCurlyLineEditor::SetModel(TObject* obj)
{
   // Pick up the used curly line attributes.

   if (obj->InheritsFrom("TCurlyArc")) {
      HideFrame(fStartXFrame);
      fStartXEntry->Disconnect("ValueSet(Long_t)");
      (fStartXEntry->GetNumberEntry())->Disconnect("ReturnPressed()");
      fStartYEntry->Disconnect("ValueSet(Long_t)");
      (fStartYEntry->GetNumberEntry())->Disconnect("ReturnPressed()");
      fEndXEntry->Disconnect("ValueSet(Long_t)");
      (fEndXEntry->GetNumberEntry())->Disconnect("ReturnPressed()");
      fEndYEntry->Disconnect("ValueSet(Long_t)");
      (fEndYEntry->GetNumberEntry())->Disconnect("ReturnPressed()");
   }

   fCurlyLine = (TCurlyLine *)obj;
   fAvoidSignal = kTRUE;

   Double_t val = fCurlyLine->GetAmplitude();
   fAmplitudeEntry->SetNumber(val);

   val = fCurlyLine->GetWaveLength();
   fWaveLengthEntry->SetNumber(val);

   val = fCurlyLine->GetStartX();
   fStartXEntry->SetNumber(val);

   val = fCurlyLine->GetEndX();
   fEndXEntry->SetNumber(val);

   val = fCurlyLine->GetStartY();
   fStartYEntry->SetNumber(val);

   val = fCurlyLine->GetEndY();
   fEndYEntry->SetNumber(val);

   if (fCurlyLine->GetCurly())
      fIsWavy->SetState(kButtonDown);
   else
      fIsWavy->SetState(kButtonUp);

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
void TCurlyLineEditor::ActivateBaseClassEditors(TClass* cl)
{
   // Exclude TAttFillEditor.

   fGedEditor->ExcludeClassEditor(TAttFill::Class());
   TGedFrame::ActivateBaseClassEditors(cl);
}

//______________________________________________________________________________
void TCurlyLineEditor::DoStartXY()
{
   // Slot connected to the CurlyLine StartPoint.

   if (fAvoidSignal) return;
   fCurlyLine->SetStartPoint(fStartXEntry->GetNumber(), fStartYEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}
//______________________________________________________________________________
void TCurlyLineEditor::DoEndXY()
{
   // Slot connected to the CurlyLine End.

   if (fAvoidSignal) return;
   fCurlyLine->SetEndPoint(fEndXEntry->GetNumber(), fEndYEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoAmplitude()
{
   // Slot connected to the amplitude setting.

   if (fAvoidSignal) return;
   fCurlyLine->SetAmplitude((Double_t)fAmplitudeEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoWaveLength()
{
   // Slot connected to the wavelength setting.

   if (fAvoidSignal) return;
   Double_t num = fWaveLengthEntry->GetNumber();
   fCurlyLine->SetWaveLength(num);
   fCurlyLine->Paint(GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoWavy()
{
   // Slot connected to the wavy / curly setting.

   if (fAvoidSignal) return;
   if (fIsWavy->GetState() == kButtonDown)
      fCurlyLine->SetCurly();
   else
      fCurlyLine->SetWavy();
   fCurlyLine->Paint(GetDrawOption());
   Update();
}
