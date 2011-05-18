// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePointSetArrayEditor.h"
#include "TEvePointSet.h"
#include "TEveGValuators.h"

#include "TVirtualPad.h"
#include "TColor.h"

#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

//______________________________________________________________________________
//
// Editor for TEvePointSetArray class.

ClassImp(TEvePointSetArrayEditor);

//______________________________________________________________________________
TEvePointSetArrayEditor::TEvePointSetArrayEditor(const TGWindow *p,
                                                 Int_t width, Int_t height,
                                                 UInt_t options, Pixel_t back) :
   TGedFrame(p,width, height, options | kVerticalFrame, back),
   fM(0),
   fRange(0)
{
   // Constructor.

   fM = 0;
   MakeTitle("TEvePointSetArray");

   fRange = new TEveGDoubleValuator(this,"Range", 200, 0);
   fRange->SetNELength(6);
   fRange->Build();
   fRange->GetSlider()->SetWidth(224);
   fRange->Connect("ValueSet()",
                   "TEvePointSetArrayEditor", this, "DoRange()");
   AddFrame(fRange, new TGLayoutHints(kLHintsTop, 1, 1, 2, 1));
}

//______________________________________________________________________________
TEvePointSetArrayEditor::~TEvePointSetArrayEditor()
{
   // Destructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEvePointSetArrayEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEvePointSetArray*>(obj);

   // printf("FullRange(%f, %f) Selected(%f,%f)\n",
   //        fM->GetMin(), fM->GetMax(), fM->GetCurMin(), fM->GetCurMax());

   fRange->SetLimits(fM->fMin, fM->fMax, TGNumberFormat::kNESRealTwo);
   fRange->SetValues(fM->fCurMin, fM->fCurMax);
}

/******************************************************************************/

//______________________________________________________________________________
void TEvePointSetArrayEditor::DoRange()
{
   // Slot for setting the range of the separating quantity.

   fM->SetRange(fRange->GetMin(), fRange->GetMax());
   Update();
}
