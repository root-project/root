// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionManagerEditor.h"
#include "TEveProjectionManager.h"
#include "TEveGValuators.h"

#include "TGNumberEntry.h"
#include "TGComboBox.h"
#include "TGLabel.h"

//______________________________________________________________________________
// TEveProjectionManagerEditor
//
// GUI editor for TEveProjectionManager class.
//

ClassImp(TEveProjectionManagerEditor)

//______________________________________________________________________________
TEveProjectionManagerEditor::TEveProjectionManagerEditor(const TGWindow *p,
                                                         Int_t width, Int_t height,
                                                         UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fType(0),
   fDistortion(0),
   fFixedRadius(0),
   fCurrentDepth(0),

   fCenterX(0),
   fCenterY(0),
   fCenterZ(0)
{
   // Constructor.

   MakeTitle("TEveProjection");
   {
      TGHorizontalFrame* f = new TGHorizontalFrame(this);
      TGLabel* lab = new TGLabel(f, "Type");
      f->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 31, 1, 2));
      fType = new TGComboBox(f);
      fType->AddEntry("CFishEye", TEveProjection::kPT_CFishEye);
      fType->AddEntry("RhoZ",     TEveProjection::kPT_RhoZ);
      TGListBox* lb = fType->GetListBox();
      lb->Resize(lb->GetWidth(), 2*18);
      fType->Resize(80, 20);
      fType->Connect("Selected(Int_t)", "TEveProjectionManagerEditor",
                     this, "DoType(Int_t)");
      f->AddFrame(fType, new TGLayoutHints(kLHintsTop, 1, 1, 2, 4));
      AddFrame(f);
   }

   Int_t nel = 6;
   Int_t labelW = 60;
   fDistortion = new TEveGValuator(this, "Distortion:", 90, 0);
   fDistortion->SetNELength(nel);
   fDistortion->SetLabelWidth(labelW);
   fDistortion->Build();
   fDistortion->SetLimits(0, 50, 101, TGNumberFormat::kNESRealTwo);
   fDistortion->Connect("ValueSet(Double_t)", "TEveProjectionManagerEditor",
                        this, "DoDistortion()");
   AddFrame(fDistortion, new TGLayoutHints(kLHintsTop, 1, 1, 1, 0));


   fFixedRadius = new TEveGValuator(this, "FixedR:", 90, 0);
   fFixedRadius->SetNELength(nel);
   fFixedRadius->SetLabelWidth(labelW);
   fFixedRadius->Build();
   fFixedRadius->SetLimits(0, 1000, 101, TGNumberFormat::kNESRealOne);
   fFixedRadius->SetToolTip("Radius not scaled by distotion.");
   fFixedRadius->Connect("ValueSet(Double_t)", "TEveProjectionManagerEditor",
                         this, "DoFixedRadius()");
   AddFrame(fFixedRadius, new TGLayoutHints(kLHintsTop, 1, 1, 1, 0));


   fCurrentDepth = new TEveGValuator(this, "CurrentZ:", 90, 0);
   fCurrentDepth->SetNELength(nel);
   fCurrentDepth->SetLabelWidth(labelW);
   fCurrentDepth->Build();
   fCurrentDepth->SetLimits(-300, 300, 601, TGNumberFormat::kNESRealTwo);
   fCurrentDepth->SetToolTip("Z coordinate of incoming projected object.");
   fCurrentDepth->Connect("ValueSet(Double_t)", "TEveProjectionManagerEditor",
                          this, "DoCurrentDepth()");
   AddFrame(fCurrentDepth, new TGLayoutHints(kLHintsTop, 1, 1, 1, 3));

   //_________
   MakeTitle("Distortion centre");
   fCenterFrame = new TGVerticalFrame(this);

   fCenterX = new TEveGValuator(fCenterFrame, "CenterX:", 90, 0);
   fCenterX->SetNELength(nel);
   fCenterX->SetLabelWidth(labelW);
   fCenterX->Build();
   fCenterX->SetLimits(-5, 5, 501, TGNumberFormat::kNESRealThree);
   fCenterX->SetToolTip("Origin of the projection.");
   fCenterX->Connect("ValueSet(Double_t)", "TEveProjectionManagerEditor",
                     this, "DoCenter()");
   fCenterFrame->AddFrame(fCenterX, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fCenterY = new TEveGValuator(fCenterFrame, "CenterY:", 90, 0);
   fCenterY->SetNELength(nel);
   fCenterY->SetLabelWidth(labelW);
   fCenterY->Build();
   fCenterY->SetLimits(-5, 5, 501, TGNumberFormat::kNESRealThree);
   fCenterY->SetToolTip("Origin of the projection.");
   fCenterY->Connect("ValueSet(Double_t)", "TEveProjectionManagerEditor",
                     this, "DoCenter()");
   fCenterFrame->AddFrame(fCenterY, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fCenterZ = new TEveGValuator(fCenterFrame, "CenterZ:", 90, 0);
   fCenterZ->SetNELength(nel);
   fCenterZ->SetLabelWidth(labelW);
   fCenterZ->Build();
   fCenterZ->SetLimits(-25, 25, 501, TGNumberFormat::kNESRealThree);
   fCenterZ->SetToolTip("Origin of the projection.");
   fCenterZ->Connect("ValueSet(Double_t)", "TEveProjectionManagerEditor",
                     this, "DoCenter()");
   fCenterFrame->AddFrame(fCenterZ, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
   
   AddFrame(fCenterFrame, new TGLayoutHints(kLHintsTop, 1, 1, 1, 0));
}

//______________________________________________________________________________
void TEveProjectionManagerEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveProjectionManager*>(obj);
   
   fType->Select(fM->GetProjection()->GetType(), kFALSE);
   fDistortion->SetValue(1000.0f * fM->GetProjection()->GetDistortion());
   fFixedRadius->SetValue(fM->GetProjection()->GetFixedRadius());
   fCurrentDepth->SetValue(fM->GetCurrentDepth());

   fCenterX->SetValue(fM->GetCenter().fX);
   fCenterY->SetValue(fM->GetCenter().fY);
   fCenterZ->SetValue(fM->GetCenter().fZ);
}

//______________________________________________________________________________
void TEveProjectionManagerEditor::DoType(Int_t type)
{
   // Slot for setting of projection type.

   fM->SetProjection((TEveProjection::EPType_e)type, 0.001f * fDistortion->GetValue());
   fM->ProjectChildren();
   Update();
}

//______________________________________________________________________________
void TEveProjectionManagerEditor::DoDistortion()
{
   // Slot for setting distortion.

   fM->GetProjection()->SetDistortion(0.001f * fDistortion->GetValue());
   fM->UpdateName();
   fM->ProjectChildren();
   Update();
}

//______________________________________________________________________________
void TEveProjectionManagerEditor::DoFixedRadius()
{
   // Slot for setting fixed radius.

   fM->GetProjection()->SetFixedRadius(fFixedRadius->GetValue());
   fM->ProjectChildren();
   Update();
}

//______________________________________________________________________________
void TEveProjectionManagerEditor::DoCurrentDepth()
{
   // Slot for setting current depth.

   fM->SetCurrentDepth(fCurrentDepth->GetValue());
   fM->ProjectChildren();
   Update();
}

//______________________________________________________________________________
void TEveProjectionManagerEditor::DoCenter()
{
   // Slot for setting center of distortion.

   fM->SetCenter(fCenterX->GetValue(), fCenterY->GetValue(), fCenterZ->GetValue());
   Update();
}

