// @(#)root/ged:$Name:  $:$Id: TCurlyArcEditor.cxx
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
//  TCurlyArcEditor                                                     //
//                                                                      //
//  Implements GUI for editing CurlyArc attributes: radius, phi1, phi2. //                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TCurlyArcEditor.gif">
*/
//End_Html


#include "TCurlyArcEditor.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TCurlyArc.h"
#include "TVirtualPad.h"
#include "iostream"

ClassImp(TGedFrame)
ClassImp(TCurlyArcEditor)

enum {
   kCRLA_RAD,
   kCRLA_FMIN,
   kCRLA_FMAX,
   kCRLA_CX,
   kCRLA_CY
};

//______________________________________________________________________________
TCurlyArcEditor::TCurlyArcEditor(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of CurlyArc GUI.

   fCurlyArc = 0;
   
   MakeTitle("Curly Arc");

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fRadiusLabel = new TGLabel(f3, "Radius:");
   f3->AddFrame(fRadiusLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fRadiusEntry = new TGNumberEntry(f3, 0.02, 7, kCRLA_RAD, 
                                    TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEANonNegative, 
                                    TGNumberFormat::kNELNoLimits);
   fRadiusEntry->GetNumberEntry()->SetToolTipText("Set radius of arc.");
   f3->AddFrame(fRadiusEntry, new TGLayoutHints(kLHintsLeft, 18, 1, 1, 1));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fPhiminLabel = new TGLabel(f4, "Phimin:");
   f4->AddFrame(fPhiminLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fPhiminEntry = new TGNumberEntry(f4, 0, 7, kCRLA_FMIN, 
                                    TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELLimitMinMax, 0, 360);
   fPhiminEntry->GetNumberEntry()->SetToolTipText("Set Phimin in degrees.");
   f4->AddFrame(fPhiminEntry, new TGLayoutHints(kLHintsLeft, 19, 1, 1, 1));

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fPhimaxLabel = new TGLabel(f5, "Phimax:");
   f5->AddFrame(fPhimaxLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fPhimaxEntry = new TGNumberEntry(f5, 0, 7, kCRLA_FMAX, 
                                    TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELLimitMinMax, 0, 360);
   fPhimaxEntry->GetNumberEntry()->SetToolTipText("Set Phimax in degrees.");
   f5->AddFrame(fPhimaxEntry, new TGLayoutHints(kLHintsLeft, 16, 1, 1, 1));

   TGCompositeFrame *f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fCenterXLabel = new TGLabel(f6, "Center X:");
   f6->AddFrame(fCenterXLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 8, 0, 1, 1));
   fCenterXEntry = new TGNumberEntry(f6, 0.0, 7, kCRLA_CX,
                                     TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEANonNegative, 
                                     TGNumberFormat::kNELNoLimits);
   fCenterXEntry->GetNumberEntry()->SetToolTipText("Set center X coordinate.");
   f6->AddFrame(fCenterXEntry, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 1));
//
   TGCompositeFrame *f7 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f7, new TGLayoutHints(kLHintsTop, 1, 1, 3, 0));

   TGLabel *fCenterYLabel = new TGLabel(f7, "Y:");
   f7->AddFrame(fCenterYLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 48, 0, 1, 1));
   fCenterYEntry = new TGNumberEntry(f7, 0.0, 7, kCRLA_CY,
                                     TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEANonNegative, 
                                     TGNumberFormat::kNELNoLimits);
   fCenterYEntry->GetNumberEntry()->SetToolTipText("Set center Y coordinate.");
   f7->AddFrame(fCenterYEntry, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));

   TClass *cl = TCurlyArc::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TCurlyArcEditor::~TCurlyArcEditor()
{
   // Destructor of CurlyArc editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________
void TCurlyArcEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fCenterXEntry->Connect("ValueSet(Long_t)", "TCurlyArcEditor", this, "DoCenterXY()");
   (fCenterXEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyArcEditor", this, "DoCenterXY()");
   fCenterYEntry->Connect("ValueSet(Long_t)", "TCurlyArcEditor", this, "DoCenterXY()");
   (fCenterYEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyArcEditor", this, "DoCenterXY()");
   fRadiusEntry->Connect("ValueSet(Long_t)", "TCurlyArcEditor", this, "DoRadius()");
   (fRadiusEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyArcEditor", this, "DoRadius()");
   fPhiminEntry->Connect("ValueSet(Long_t)", "TCurlyArcEditor", this, "DoPhimin()");
   (fPhiminEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyArcEditor", this, "DoPhimin()");
   fPhimaxEntry->Connect("ValueSet(Long_t)", "TCurlyArcEditor", this, "DoPhimax()");
   (fPhimaxEntry->GetNumberEntry())->Connect("ReturnPressed()", "TCurlyArcEditor", this, "DoPhimax()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TCurlyArcEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used curly arc attributes.

   fModel = 0;
   fPad = 0;
   if (obj == 0 || !obj->InheritsFrom("TCurlyArc")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;
   
   fCurlyArc = (TCurlyArc *)fModel;

   Double_t val = fCurlyArc->GetRadius();
   fRadiusEntry->SetNumber(val);

   val = fCurlyArc->GetPhimin();
   fPhiminEntry->SetNumber(val);

   val = fCurlyArc->GetPhimax();
   fPhimaxEntry->SetNumber(val);

   val = fCurlyArc->GetStartX();
   fCenterXEntry->SetNumber(val);

   val = fCurlyArc->GetStartY();
   fCenterYEntry->SetNumber(val);

   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoCenterXY()
{
   // Slot connected to set center .

   fCurlyArc->SetCenter((Double_t)fCenterXEntry->GetNumber(), (Double_t)fCenterYEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoRadius()
{
   // Slot connected to the radius setting.

   fCurlyArc->SetRadius((Double_t)fRadiusEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoPhimin()
{
   // Slot connected to the phimin setting.

   fCurlyArc->SetPhimin((Double_t)fPhiminEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoPhimax()
{
   // Slot connected to the phimax setting.
   fCurlyArc->SetPhimax((Double_t)fPhimaxEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}


//______________________________________________________________________________
