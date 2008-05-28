// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePolygonSetProjectedEditor.h"
#include "TEvePolygonSetProjected.h"

#include "TColor.h"

#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"

//==============================================================================
//==============================================================================
// TEvePolygonSetProjectedEditor
//==============================================================================

//______________________________________________________________________________
//
// GUI editor for class TEvePolygonSetProjected.

ClassImp(TEvePolygonSetProjectedEditor);

//______________________________________________________________________________
TEvePolygonSetProjectedEditor::TEvePolygonSetProjectedEditor(const TGWindow *p,
                                                             Int_t width, Int_t height,
                                                             UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fPS(0),
   fLineWidth(0),
   fLineColor(0)
{
   // Constructor.

   MakeTitle("TEvePolygonSetProjected");

   {
      TGCompositeFrame *f = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);

      TGLabel *l = new TGLabel(f, "LineColor:");
      f->AddFrame(l, new TGLayoutHints(kLHintsLeft, 2, 2, 1, 1));
      fLineColor = new TGColorSelect(f, 0, -1);
      fLineColor->Connect("ColorSelected(Pixel_t)", "TEvePolygonSetProjectedEditor", this, "DoLineColor(Pixel_t)");
      f->AddFrame(fLineColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

      fLineWidth = new TGNumberEntry(f, 0., 4, -1,
                                     TGNumberFormat::kNESRealOne, TGNumberFormat::kNEAPositive,
                                     TGNumberFormat::kNELLimitMinMax, 0.1, 20.0);
      fLineWidth->GetNumberEntry()->SetToolTipText("Line width of outline.");
      fLineWidth->Connect("ValueSet(Long_t)", "TEvePolygonSetProjectedEditor", this, "DoLineWidth()");
      f->AddFrame(fLineWidth, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

      AddFrame(f, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEvePolygonSetProjectedEditor::SetModel(TObject* obj)
{
   // Set model object.

   fPS = dynamic_cast<TEvePolygonSetProjected*>(obj);
   fLineWidth->SetNumber(fPS->fLineWidth);
   fLineColor->SetColor(TColor::Number2Pixel(fPS->GetLineColor()), kFALSE);
}

/******************************************************************************/

//______________________________________________________________________________
void TEvePolygonSetProjectedEditor::DoLineWidth()
{
   // Slot for setting line with of polygon outline.

   fPS->fLineWidth = fLineWidth->GetNumber();
   Update();
}

//______________________________________________________________________________
void TEvePolygonSetProjectedEditor::DoLineColor(Pixel_t pixel)
{
   // Slot for setting line color of polygon outline.

   fPS->SetLineColor(TColor::GetColor(pixel));
   Update();
}
