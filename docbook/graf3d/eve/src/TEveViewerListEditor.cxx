// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveViewerListEditor.h"
#include "TEveViewer.h"
#include "TEveGValuators.h"

//______________________________________________________________________________
// GUI editor for TEveViewerList.
//

ClassImp(TEveViewerListEditor);

//______________________________________________________________________________
TEveViewerListEditor::TEveViewerListEditor(const TGWindow *p, Int_t width, Int_t height,
             UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),

   fBrightness(0),
   fColorSet(0)
{
   // Constructor.

   MakeTitle("TEveViewerList");

   Int_t labelW = 63;
   fBrightness = new TEveGValuator(this, "Brightness:", 90, 0);
   fBrightness->SetLabelWidth(labelW);
   fBrightness->SetNELength(4);
   fBrightness->Build();
   fBrightness->SetLimits(-2, 2 ,  41 , TGNumberFormat::kNESRealTwo);
   fBrightness->Connect("ValueSet(Double_t)", "TEveViewerListEditor", this, "DoBrightness()");
   AddFrame(fBrightness, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fColorSet =  new TGTextButton(this , "Switch ColorSet");
   fColorSet->Connect("Clicked()", "TEveViewerListEditor", this, "SwitchColorSet()");
   AddFrame(fColorSet, new TGLayoutHints(kLHintsLeft, 2, 1, 4, 4));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewerListEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveViewerList*>(obj);

   fBrightness->SetValue(fM->GetColorBrightness());
}

/******************************************************************************/

// Implements callback/slot methods

//______________________________________________________________________________
void TEveViewerListEditor::DoBrightness()
{
   // Slot for brightness.

   fColorSet->SetText(fM->UseLightColorSet()?"DarkColorSet": "Light ColorSet");
   fM->SetColorBrightness(fBrightness->GetValue());
}

//______________________________________________________________________________
void TEveViewerListEditor::SwitchColorSet()
{
   // Slot for color set.

   fColorSet->SetText(fM->UseLightColorSet()? "Light ColorSet":"Dark ColorSet");
   fM->SwitchColorSet();
}
