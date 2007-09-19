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
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TCurlyArc.h"
#include "iostream"

ClassImp(TCurlyArcEditor)

enum ECurlyArcWid {
   kCRLA_RAD,
   kCRLA_FMIN,
   kCRLA_FMAX,
   kCRLA_CX,
   kCRLA_CY
};

//______________________________________________________________________________
TCurlyArcEditor::TCurlyArcEditor(const TGWindow *p, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of CurlyArc GUI.

   fCurlyArc = 0;

   MakeTitle("Curly Arc");

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 2, 0));

   TGCompositeFrame *f3a = new TGCompositeFrame(f3, 80, 20);
   f3->AddFrame(f3a, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGLabel *fRadiusLabel = new TGLabel(f3a, "Radius:");
   f3a->AddFrame(fRadiusLabel, new TGLayoutHints(kLHintsNormal, 8, 0, 5, 5));

   TGLabel *fPhiminLabel = new TGLabel(f3a, "Phimin:");
   f3a->AddFrame(fPhiminLabel, new TGLayoutHints(kLHintsNormal, 8, 0, 5, 5));

   TGLabel *fPhimaxLabel = new TGLabel(f3a, "Phimax:");
   f3a->AddFrame(fPhimaxLabel, new TGLayoutHints(kLHintsNormal, 8, 0, 5, 5));

   TGLabel *fCenterXLabel = new TGLabel(f3a, "Center X:");
   f3a->AddFrame(fCenterXLabel, new TGLayoutHints(kLHintsNormal, 8, 0, 6, 5));

   TGLabel *fCenterYLabel = new TGLabel(f3a, "Y:");
   f3a->AddFrame(fCenterYLabel, new TGLayoutHints(kLHintsNormal, 49, 0, 6, 0));

   TGCompositeFrame *f3b = new TGCompositeFrame(f3, 80, 20);
   f3->AddFrame(f3b, new TGLayoutHints(kLHintsNormal, 0, 0, 0, 0));

   fRadiusEntry = new TGNumberEntry(f3b, 0.02, 7, kCRLA_RAD,
                                    TGNumberFormat::kNESRealThree,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELNoLimits);
   fRadiusEntry->GetNumberEntry()->SetToolTipText("Set radius of arc.");
   f3b->AddFrame(fRadiusEntry, new TGLayoutHints(kLHintsLeft, 6, 1, 3, 1));

   fPhiminEntry = new TGNumberEntry(f3b, 0, 7, kCRLA_FMIN,
                                    TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELLimitMinMax, 0, 360);
   fPhiminEntry->GetNumberEntry()->SetToolTipText("Set Phimin in degrees.");
   f3b->AddFrame(fPhiminEntry, new TGLayoutHints(kLHintsLeft, 6, 1, 3, 1));

   fPhimaxEntry = new TGNumberEntry(f3b, 0, 7, kCRLA_FMAX,
                                    TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative,
                                    TGNumberFormat::kNELLimitMinMax, 0, 360);
   fPhimaxEntry->GetNumberEntry()->SetToolTipText("Set Phimax in degrees.");
   f3b->AddFrame(fPhimaxEntry, new TGLayoutHints(kLHintsLeft, 6, 1, 3, 1));

   fCenterXEntry = new TGNumberEntry(f3b, 0.0, 7, kCRLA_CX,
                                     TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEANonNegative,
                                     TGNumberFormat::kNELNoLimits);
   fCenterXEntry->GetNumberEntry()->SetToolTipText("Set center X coordinate.");
   f3b->AddFrame(fCenterXEntry, new TGLayoutHints(kLHintsLeft, 6, 1, 3, 1));

   fCenterYEntry = new TGNumberEntry(f3b, 0.0, 7, kCRLA_CY,
                                     TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEANonNegative,
                                     TGNumberFormat::kNELNoLimits);
   fCenterYEntry->GetNumberEntry()->SetToolTipText("Set center Y coordinate.");
   f3b->AddFrame(fCenterYEntry, new TGLayoutHints(kLHintsLeft, 6, 1, 3, 1));

}

//______________________________________________________________________________
TCurlyArcEditor::~TCurlyArcEditor()
{
   // Destructor of CurlyArc editor.
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
void TCurlyArcEditor::SetModel(TObject* obj)
{
   // Pick up the used curly arc attributes.

   fCurlyArc = (TCurlyArc *)obj;
   fAvoidSignal = kTRUE;

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

   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
void TCurlyArcEditor::DoCenterXY()
{
   // Slot connected to set center .

   if (fAvoidSignal) return;
   fCurlyArc->SetCenter((Double_t)fCenterXEntry->GetNumber(), (Double_t)fCenterYEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoRadius()
{
   // Slot connected to the radius setting.

   if (fAvoidSignal) return;
   fCurlyArc->SetRadius((Double_t)fRadiusEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoPhimin()
{
   // Slot connected to the phimin setting.

   if (fAvoidSignal) return;
   fCurlyArc->SetPhimin((Double_t)fPhiminEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}

//______________________________________________________________________________
void TCurlyArcEditor::DoPhimax()
{
   // Slot connected to the phimax setting.

   if (fAvoidSignal) return;
   fCurlyArc->SetPhimax((Double_t)fPhimaxEntry->GetNumber());
   fCurlyArc->Paint(fCurlyArc->GetDrawOption());
   Update();
}
