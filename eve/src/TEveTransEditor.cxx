// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTransEditor.h"
#include "TEveTrans.h"
#include "TEveGValuators.h"

#include "TVirtualPad.h"
#include "TMath.h"

#include "TGButton.h"
#include "TGLabel.h"

//______________________________________________________________________________
// TEveTransSubEditor
//
// Sub-editor for TEveTrans class.

ClassImp(TEveTransSubEditor);

//______________________________________________________________________________
TEveTransSubEditor::TEveTransSubEditor(TGWindow* p) :
   TGVerticalFrame (p),
   fTrans          (0),
   fTopHorFrame    (0),
   fUseTrans       (0),
   fEditTrans      (0),
   fEditTransFrame (0),
   fPos            (0),
   fRot            (0),
   fScale          (0),
   fAutoUpdate     (0),
   fUpdate         (0)
{
   // Constructor.

   // --- Top controls

   fTopHorFrame = new TGHorizontalFrame(this);

   fUseTrans  = new TGCheckButton(fTopHorFrame, "UseTrans");
   fTopHorFrame->AddFrame(fUseTrans, new TGLayoutHints(kLHintsLeft, 1,2,0,0));
   fUseTrans->Connect("Toggled(Bool_t)", "TEveTransSubEditor", this, "DoUseTrans()");
   fEditTrans = new TGCheckButton(fTopHorFrame, "EditTrans");
   fTopHorFrame->AddFrame(fEditTrans, new TGLayoutHints(kLHintsLeft, 2,1,0,0));
   fEditTrans->Connect("Toggled(Bool_t)"," TEveTransSubEditor", this, "DoEditTrans()");

   AddFrame(fTopHorFrame, new TGLayoutHints(kLHintsTop, 0,0,2,1));

   // --- Trans edit part

   fEditTransFrame = new TGVerticalFrame(this);

   TGFont *font = gClient->GetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");

   TGHorizontalFrame* hfp = new TGHorizontalFrame(fEditTransFrame);
   TGLabel* labp = new TGLabel(hfp, "Location");
   labp->SetTextFont(font);
   hfp->AddFrame(labp);
   fEditTransFrame->AddFrame(hfp,  new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,0));
   fPos = new TEveGTriVecValuator(fEditTransFrame, "Pos", 160, 20);
   fPos->SetNELength(6);
   fPos->Build(kFALSE, "", "", "");
   fPos->SetLimits(-1e5, 1e5, TGNumberFormat::kNESRealThree);
   fPos->GetValuator(0)->SetToolTip("X coordinate");
   fPos->GetValuator(1)->SetToolTip("Y coordinate");
   fPos->GetValuator(2)->SetToolTip("Z coordinate");
   fEditTransFrame->AddFrame(fPos, new TGLayoutHints(kLHintsTop , 0,0,0,0));

   TGHorizontalFrame* hfr = new TGHorizontalFrame(fEditTransFrame);
   TGLabel* labr = new TGLabel(hfr, "Rotation");
   labr->SetTextFont(font);
   hfr->AddFrame(labr);
   fEditTransFrame->AddFrame(hfr, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,0));
   fRot = new TEveGTriVecValuator(fEditTransFrame, "Rot", 160, 20);
   fRot->SetNELength(6);
   fRot->Build(kFALSE, "", "", "");
   fRot->SetLimits(-360, 360, TGNumberFormat::kNESRealOne);
   fRot->GetValuator(0)->SetToolTip("X coordinate");
   fRot->GetValuator(1)->SetToolTip("Y coordinate");
   fRot->GetValuator(2)->SetToolTip("Z coordinate");
   fEditTransFrame->AddFrame(fRot, new TGLayoutHints(kLHintsTop , 0,0, 0, 0));

   TGHorizontalFrame* hfs = new TGHorizontalFrame(fEditTransFrame);
   TGLabel* labs = new TGLabel(hfs, "Scale");
   labs->SetTextFont(font);
   hfs->AddFrame(labs);
   fEditTransFrame->AddFrame(hfs, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,0));
   fScale = new TEveGTriVecValuator(fEditTransFrame, "Scale", 160, 20);
   fScale->SetNELength(6);
   fScale->Build(kFALSE, "", "", "");
   fScale->SetLimits(1e-2, 1e2, TGNumberFormat::kNESRealTwo);
   fScale->GetValuator(0)->SetToolTip("X coordinate");
   fScale->GetValuator(1)->SetToolTip("Y coordinate");
   fScale->GetValuator(2)->SetToolTip("Z coordinate");
   fEditTransFrame->AddFrame(fScale, new TGLayoutHints(kLHintsTop , 0,0,0, 2));

   fPos  ->Connect("ValueSet()", "TEveTransSubEditor", this, "DoTransChanged()");
   fRot  ->Connect("ValueSet()", "TEveTransSubEditor", this, "DoTransChanged()");
   fScale->Connect("ValueSet()", "TEveTransSubEditor", this, "DoTransChanged()");

   {
      TGHorizontalFrame* hframe = new TGHorizontalFrame(fEditTransFrame);
      fAutoUpdate = new TGCheckButton(hframe, "AutoUpdate");
      hframe->AddFrame(fAutoUpdate, new TGLayoutHints(kLHintsLeft, 1,1,1,1));
      fUpdate = new TGTextButton(hframe, "Update");
      hframe->AddFrame(fUpdate, new TGLayoutHints(kLHintsLeft, 0,0,1,1));
      fUpdate->Connect("Clicked()", "TEveTransSubEditor", this, "TransChanged()");

      fEditTransFrame->AddFrame(hframe, new TGLayoutHints(kLHintsTop , 0,0,4,0));
   }

   AddFrame(fEditTransFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,1,2));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTransSubEditor::SetModel(TEveTrans* t)
{
   // Set model object.

   fTrans = t;

   fUseTrans ->SetState(fTrans->fUseTrans  ? kButtonDown : kButtonUp);
   fEditTrans->SetState(fTrans->fEditTrans ? kButtonDown : kButtonUp);
   if (fTrans->fEditTrans)
   {
      for (Int_t i=0; i<3; ++i)
      {
         fRot  ->GetValuator(i)->GetEntry()->SetState(fTrans->GetEditRotation());
         fScale->GetValuator(i)->GetEntry()->SetState(fTrans->GetEditScale());
      }
      fEditTransFrame->MapWindow();
   }
   else
   {
      fEditTransFrame->UnmapWindow();
   }

   ((TGMainFrame*)fEditTransFrame->GetMainFrame())->Layout();

   fPos->SetValues(fTrans->ArrT());
   Float_t a[3];
   fTrans->GetRotAngles(a);
   a[0] *= TMath::RadToDeg();
   a[1] *= TMath::RadToDeg();
   a[2] *= TMath::RadToDeg();
   fRot->SetValues(a);
   Double_t x, y, z;
   fTrans->GetScale(x, y, z);
   fScale->SetValues(x, y, z);
}

//______________________________________________________________________________
void TEveTransSubEditor::SetTransFromData()
{
   // Set model object from widget data.

   Double_t v[3];
   fTrans->UnitTrans();
   fRot->GetValues(v);
   fTrans->SetRotByAngles(v[0]*TMath::DegToRad(), v[1]*TMath::DegToRad(), v[2]*TMath::DegToRad());
   fPos->GetValues(v);
   fTrans->SetPos(v);
   fScale->GetValues(v);
   fTrans->Scale(v[0], v[1], v[2]);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTransSubEditor::UseTrans()
{
   // Emit "UseTrans()" signal.

   Emit("UseTrans()");
}

//______________________________________________________________________________
void TEveTransSubEditor::TransChanged()
{
   // Set transformation values from widget and emit "TransChanged()" signal.

   SetTransFromData();
   Emit("TransChanged()");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTransSubEditor::DoUseTrans()
{
   // Slot for UseTrans.

   fTrans->SetUseTrans(fUseTrans->IsOn());
   UseTrans();
}

//______________________________________________________________________________
void TEveTransSubEditor::DoEditTrans()
{
   // Slot for EditTrans.

   fTrans->SetEditTrans(fEditTrans->IsOn());
   TransChanged();
}

//______________________________________________________________________________
void TEveTransSubEditor::DoTransChanged()
{
   // Slot for TransChanged.

   if (fAutoUpdate->IsOn())
      TransChanged();
}


//______________________________________________________________________________
// TEveTransEditor
//
// Editor for TEveTrans class.

ClassImp(TEveTransEditor);

//______________________________________________________________________________
TEveTransEditor::TEveTransEditor(const TGWindow *p, Int_t width, Int_t height,
                                 UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM (0),
   fSE(0)
{
   // Constructor.

   MakeTitle("TEveTrans");

   fSE = new TEveTransSubEditor(this);
   AddFrame(fSE, new TGLayoutHints(kLHintsTop, 2, 0, 2, 2));
   fSE->Connect("UseTrans()",     "TEveTransEditor", this, "Update()");
   fSE->Connect("TransChanged()", "TEveTransEditor", this, "Update()");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTransEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveTrans*>(obj);
   fSE->SetModel(fM);
}
