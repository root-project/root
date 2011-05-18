// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveArrowEditor.h"
#include "TEveArrow.h"
#include "TVirtualPad.h"
#include "TGLabel.h"
#include "TEveGValuators.h"

//______________________________________________________________________________
// GUI editor for TEveArrow.
//

ClassImp(TEveArrowEditor);

//______________________________________________________________________________
TEveArrowEditor::TEveArrowEditor(const TGWindow *p, Int_t width, Int_t height,
                                 UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fTubeR(0),
   fConeR(0),
   fConeL(0),

   fOrigin(0),
   fVector(0)
{
   // Constructor.

   MakeTitle("TEveArrow");
   Int_t labelW = 51;

   fTubeR = new TEveGValuator(this, "TubeR:", 90, 0);
   fTubeR->SetLabelWidth(labelW);
   fTubeR->SetNELength(6);
   fTubeR->Build();
   fTubeR->SetLimits(0.001, 1, 101, TGNumberFormat::kNESRealThree);
   fTubeR->Connect("ValueSet(Double_t)", "TEveArrowEditor", this, "DoTubeR()");
   AddFrame(fTubeR, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fConeR = new TEveGValuator(this, "ConeR:", 90, 0);
   fConeR->SetLabelWidth(labelW);
   fConeR->SetNELength(6);
   fConeR->Build();
   fConeR->SetLimits(0.001, 1, 101, TGNumberFormat::kNESRealThree);
   fConeR->Connect("ValueSet(Double_t)", "TEveArrowEditor", this, "DoConeR()");
   AddFrame(fConeR, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fConeL = new TEveGValuator(this, "ConeL:", 90, 0);
   fConeL->SetLabelWidth(labelW);
   fConeL->SetNELength(6);
   fConeL->Build();
   fConeL->SetLimits(0.001, 1, 101, TGNumberFormat::kNESRealThree);
   fConeL->Connect("ValueSet(Double_t)", "TEveArrowEditor", this, "DoConeL()");
   AddFrame(fConeL, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   TGFont *font = gClient->GetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   {
      TGHorizontalFrame* hfp = new TGHorizontalFrame(this);
      TGLabel* labp = new TGLabel(hfp, "Origin");
      labp->SetTextFont(font);
      hfp->AddFrame(labp);
      AddFrame(hfp,  new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,0));
      fOrigin = new TEveGTriVecValuator(this, "Origin", 160, 20);
      fOrigin->SetNELength(6);
      fOrigin->Build(kFALSE, "", "", "");
      fOrigin->SetLimits(-1e5, 1e5, TGNumberFormat::kNESRealThree);
      AddFrame(fOrigin, new TGLayoutHints(kLHintsTop , 0,0,0,0));
   }

   {
      TGHorizontalFrame* hfp = new TGHorizontalFrame(this);
      TGLabel* labp = new TGLabel(hfp, "Vector");
      labp->SetTextFont(font);
      hfp->AddFrame(labp);
      AddFrame(hfp,  new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,0));
      fVector = new TEveGTriVecValuator(this, "Vector", 160, 20);
      fVector->SetNELength(6);
      fVector->Build(kFALSE, "", "", "");
      fVector->SetLimits(-1e5, 1e5, TGNumberFormat::kNESRealThree);
      AddFrame(fVector, new TGLayoutHints(kLHintsTop , 0,0,0,0));
   }

   fVector->Connect("ValueSet()", "TEveArrowEditor", this, "DoVertex()");
   fOrigin->Connect("ValueSet()", "TEveArrowEditor", this, "DoVertex()");
}

//______________________________________________________________________________
void TEveArrowEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveArrow*>(obj);

   fTubeR->SetValue(fM->fTubeR);
   fTubeR->SetValue(fM->fTubeR);
   fConeL->SetValue(fM->fConeL);

   fOrigin->SetValues(&fM->fOrigin[0]);
   fVector->SetValues(&fM->fVector[0]);
}

//______________________________________________________________________________
void TEveArrowEditor::DoTubeR()
{
   // Slot for TubeR.

   fM->SetTubeR(fTubeR->GetValue());
   Update();
}

//______________________________________________________________________________
void TEveArrowEditor::DoConeR()
{
   // Slot for ConeR.

   fM->SetConeR(fConeR->GetValue());
   Update();
}

//______________________________________________________________________________
void TEveArrowEditor::DoConeL()
{
   // Slot for ConeL.

   fM->SetConeL(fConeL->GetValue());
   Update();
}

//______________________________________________________________________________
void TEveArrowEditor::DoVertex()
{
   // Slot for origin and vector.

   Double_t v[3];
   fOrigin->GetValues(v);
   fM->fOrigin.Set(v);
   fVector->GetValues(v);
   fM->fVector.Set(v);
   Update();
}


