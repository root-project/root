// @(#)root/ged:$Name:  $:$Id: TCurlyLineEditor.cxx
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
//  TCurlyLineEditor                                                        //
//                                                                      //
//  Implements GUI for editing CurlyLine attributes: shape, size, angle.    //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TCurlyLineEditor.gif">
*/
//End_Html


#include "TCurlyLineEditor.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGButton.h"
#include "TCurlyLine.h"
#include "TVirtualPad.h"
#include "iostream"
ClassImp(TGedFrame)
ClassImp(TCurlyLineEditor)

//______________________________________________________________________________
TCurlyLineEditor::TCurlyLineEditor(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of CurlyLine GUI.

   fCurlyLine = 0;
   
   MakeTitle("CurlyLine");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGLabel *fShapeLabel = new TGLabel(f2, "Shape:");
   f2->AddFrame(fShapeLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fAmplitudeLabel = new TGLabel(f3, "Amplitude:");
   f3->AddFrame(fAmplitudeLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fAmplitudeEntry = new TGNumberEntry(f3, 30, 8, 0, 
                             TGNumberFormat::kNESRealThree,
                             TGNumberFormat::kNEANonNegative,
                             TGNumberFormat::kNELLimitMinMax,0.005,0.3);
   fAmplitudeEntry->GetNumberEntry()->SetToolTipText("Set Amplitude");
   f3->AddFrame(fAmplitudeEntry, new TGLayoutHints(kLHintsLeft, 16, 1, 1, 1));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fWaveLengthLabel = new TGLabel(f4, "Wavelgth:");
   f4->AddFrame(fWaveLengthLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fWaveLengthEntry = new TGNumberEntry(f4, 30, 8, 0, 
                                  TGNumberFormat::kNESRealThree,
                                  TGNumberFormat::kNEANonNegative, 
                                  TGNumberFormat::kNELLimitMinMax, 0.005, 0.3);
   fWaveLengthEntry->GetNumberEntry()->SetToolTipText("Set Wavelength");
   f4->AddFrame(fWaveLengthEntry, new TGLayoutHints(kLHintsLeft, 16, 1, 1, 1));

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));
   fIsWavy = new TGCheckButton(f5, "Is wavy (Gamma)", 0);
   fIsWavy->SetToolTipText("Toggle wavy / curly i.e gamma / gluon"); 
   f5->AddFrame(fIsWavy, new TGLayoutHints(kLHintsLeft, 16, 1, 1, 1));

//
   fStartXFrame = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(fStartXFrame, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fStartXLabel = new TGLabel(fStartXFrame, "Start X:");
   fStartXFrame->AddFrame(fStartXLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fStartXEntry = new TGNumberEntry(fStartXFrame, 0.0, 8, 0);
   fStartXEntry->GetNumberEntry()->SetToolTipText("Set start point of curly line.");
   fStartXFrame->AddFrame(fStartXEntry, new TGLayoutHints(kLHintsLeft, 20, 1, 1, 1));
//
   fStartYFrame = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(fStartYFrame, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fStartYLabel = new TGLabel(fStartYFrame, "Start Y:");
   fStartYFrame->AddFrame(fStartYLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fStartYEntry = new TGNumberEntry(fStartYFrame, 0.0, 8, 0);
   fStartYEntry->GetNumberEntry()->SetToolTipText("Set start point of curly line.");
   fStartYFrame->AddFrame(fStartYEntry, new TGLayoutHints(kLHintsLeft, 20, 1, 1, 1));
//
   fEndXFrame = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(fEndXFrame, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fEndXLabel = new TGLabel(fEndXFrame, "End X:");
   fEndXFrame->AddFrame(fEndXLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fEndXEntry = new TGNumberEntry(fEndXFrame, 0.0, 8, 0);
   fEndXEntry->GetNumberEntry()->SetToolTipText("Set end point of curly line.");
   fEndXFrame->AddFrame(fEndXEntry, new TGLayoutHints(kLHintsLeft, 20, 1, 1, 1));
//
   fEndYFrame = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(fEndYFrame, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fEndYLabel = new TGLabel(fEndYFrame, "End Y:");
   fEndYFrame->AddFrame(fEndYLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fEndYEntry = new TGNumberEntry(fEndYFrame, 0.0, 8, 0);
   fEndYEntry->GetNumberEntry()->SetToolTipText("Set end point of curly line.");
   fEndYFrame->AddFrame(fEndYEntry, new TGLayoutHints(kLHintsLeft, 20, 1, 1, 1));

   TClass *cl = TAttLine::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TCurlyLineEditor::~TCurlyLineEditor()
{
   // Destructor of CurlyLine editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
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
   fIsWavy->Connect("Pressed()", "TCurlyLineEditor", this, "DoWavy()");
   fIsWavy->Connect("Released()", "TCurlyLineEditor", this, "DoWavy()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TCurlyLineEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used curly line attributes.

   fModel = 0;
   fPad = 0;
   if (obj == 0 || !obj->InheritsFrom("TCurlyLine")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;
   if (obj->InheritsFrom("TCurlyArc")) {
//      std::cout << "InheritsFrom(TCurlyArc)" << std::endl;
      HideFrame(fStartXFrame);
      HideFrame(fStartYFrame);
      HideFrame(fEndXFrame);
      HideFrame(fEndYFrame);
   }

   fCurlyLine = (TCurlyLine *)fModel;

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
   if (fCurlyLine->GetCurly()) fIsWavy->SetState(kButtonUp);
   else                        fIsWavy->SetState(kButtonDown);
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoStartXY()
{
   // Slot connected to the CurlyLine StartPoint.

   fCurlyLine->SetStartPoint(fStartXEntry->GetNumber(), fStartYEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}
//______________________________________________________________________________
void TCurlyLineEditor::DoEndXY()
{
   // Slot connected to the CurlyLine End.

   fCurlyLine->SetEndPoint(fEndXEntry->GetNumber(), fEndYEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoAmplitude()
{
   // Slot connected to the amplitude setting.

   fCurlyLine->SetAmplitude((Double_t)fAmplitudeEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoWaveLength()
{
   // Slot connected to the wavelength setting.

   fCurlyLine->SetWaveLength((Double_t)fWaveLengthEntry->GetNumber());
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   HideFrame(fWaveLengthEntry);
   Update();
}

//______________________________________________________________________________
void TCurlyLineEditor::DoWavy()
{
   // Slot connected to the wavy / curly setting.
   if (fIsWavy->GetState() == kButtonUp) fCurlyLine->SetCurly();
   else                                  fCurlyLine->SetWavy();
   fCurlyLine->Paint(fCurlyLine->GetDrawOption());
   Update();
}

//______________________________________________________________________________
