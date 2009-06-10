// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TGedEditor.h"
#include "TGDoubleSlider.h"
#include "TGNumberEntry.h"
#include "TVirtualPad.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TClass.h"
#include "TGTab.h"

#include "TGL5DDataSetEditor.h"
#include "TGLUtil.h"
#include "TGL5D.h"

//______________________________________________________________________________
//
// GUI editor for OpenGL 5D Painter.

ClassImp(TGL5DDataSetEditor);

//______________________________________________________________________________
TGL5DDataSetEditor::TGL5DDataSetEditor(const TGWindow *p,  Int_t width,
   Int_t height, UInt_t options, Pixel_t back) : TGedFrame(p,  width, height,
   options | kVerticalFrame, back), fShowBoxCut(), fNumberOfPlanes(0), 
   fAlpha(0), fLogScale(0), fSlideRange(0), fApplyAlpha(0), fApplyPlanes(0)
{
  //  Constructor.

   CreateStyleTab();
}

//______________________________________________________________________________

TGL5DDataSetEditor::~TGL5DDataSetEditor()
{
   // Destructor.

}

//______________________________________________________________________________
void TGL5DDataSetEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fShowBoxCut->Connect("Toggled(Bool_t)", "TGL5DDataSetEditor", this, "DoShowBoxCut()");
   fSlideRange->Connect("PositionChanged()","TGL5DDataSetEditor", this,"DoSliderRangeMoved()");
   fLogScale->Connect("Toggled(Bool_t)", "TGL5DDataSetEditor", this, "DoLogScale()");
   fApplyPlanes->Connect("Clicked()", "TGL5DDataSetEditor", this, "DoPlanes()");
   fApplyAlpha->Connect("Clicked()", "TGL5DDataSetEditor", this, "DoAlpha()");
   
   fAlpha->Connect("ValueChanged(Long_t)", "TGL5DDataSetEditor", this, "DoAlphaChanged()");
   fAlpha->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "DoAlphaChanged()");
   fNumberOfPlanes->Connect("ValueChanged(Long_t)", "TGL5DDataSetEditor", this, "DoNContoursChanged()");
   fNumberOfPlanes->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "DoNContoursChanged()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TGL5DDataSetEditor::CreateStyleTab()
{
   // Creates "Style" tab.

   TGHorizontalFrame *f;
   //MakeTitle("Update behaviour");
   fShowBoxCut  = new TGCheckButton(this, "Show Box Cut");
   fShowBoxCut->SetToolTipText("Box cut. When attached to a plot, cuts away a part of it");
   AddFrame(fShowBoxCut, new TGLayoutHints(kLHintsLeft, 5, 2, 2, 2));

   MakeTitle("isosurfaces");
   f = new TGHorizontalFrame(this, 200, 50);
   f->AddFrame(new TGLabel(f, "Number:"), new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));
   fNumberOfPlanes = new TGNumberEntry(f, 0, 3, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
                                       TGNumberFormat::kNELLimitMinMax, 1, 200);
   fNumberOfPlanes->GetNumberEntry()->SetToolTipText("Set number of isosurfaces");
   f->AddFrame(fNumberOfPlanes, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));
   fApplyPlanes = new TGTextButton(f, "   Apply   ");
   f->AddFrame(fApplyPlanes, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));
   AddFrame(f, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
   fApplyPlanes->SetState(kButtonDisabled);

   MakeTitle("Alpha");
   f = new TGHorizontalFrame(this, 200, 50);
   f->AddFrame(new TGLabel(f, "Value:"), new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));
   fAlpha = new TGNumberEntry(f, 0, 1, -1, TGNumberFormat::kNESRealThree, TGNumberFormat::kNEANonNegative, 
                              TGNumberFormat::kNELLimitMinMax, 0.1, 0.5);
   fAlpha->GetNumberEntry()->SetToolTipText("Value of alpha parameter");
   f->AddFrame(fAlpha, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));
   fApplyAlpha = new TGTextButton(f, "   Apply   ");
   f->AddFrame(fApplyAlpha, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));
   AddFrame(f, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));
   fApplyAlpha->SetState(kButtonDisabled);

   fLogScale  = new TGCheckButton(this, "Log Scale");
   AddFrame(fLogScale, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 2, 2, 2));

   AddFrame(new TGLabel(this, "Slide Range:"), new TGLayoutHints(kLHintsLeft, 5, 2, 2, 2));
   fSlideRange = new TGDoubleHSlider(this, 200, kDoubleScaleDownRight);
   AddFrame(fSlideRange, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 2, 2, 2));
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoAlpha()
{
   // Slot connected to the Log Scale check button.

   if (fPainter) {
      fApplyAlpha->SetState(kButtonDisabled);
      fPainter->SetAlpha(fAlpha->GetNumber());
      fAlpha->SetNumber(fPainter->GetAlpha());
   }
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoLogScale()
{
   // Slot connected to the Log Scale check button.

   if (fPainter) {
   }
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoPlanes()
{
   // Slot connected to the Apply Planes button.

   if (fPainter) {
      fApplyPlanes->SetState(kButtonDisabled);
      fPainter->SetNContours((Int_t)fNumberOfPlanes->GetIntNumber());
      fNumberOfPlanes->SetIntNumber(fPainter->GetNContours());
   }
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoShowBoxCut()
{
   // Slot connected to the Show BoxCut check button.

   if (fPainter) fPainter->ShowBoxCut(fShowBoxCut->IsOn());
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoSliderRangeMoved()
{
   // Slot connected to the Range Slider.

   if (fPainter) {
      fPainter->SetV5SliderMin(fSlideRange->GetMinPosition());
      fPainter->SetV5SliderMax(fSlideRange->GetMaxPosition());
   }
   if (gPad) gPad->Update();
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoAlphaChanged()
{
   // Slot connected to the Alpha button.

   fApplyAlpha->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGL5DDataSetEditor::DoNContoursChanged()
{
   // Slot connected to the Number of Planes value-entry.

   fApplyPlanes->SetState(kButtonUp);
}

//______________________________________________________________________________
void TGL5DDataSetEditor::SetModel(TObject* obj)
{
   // Sets model or disables/hides viewer.

   fPainter = 0;

   TGL5DDataSet *dset = static_cast<TGL5DDataSet *>(obj);
   fPainter = dset->GetRealPainter();

   fShowBoxCut->SetState(fPainter->IsBoxCutShown() ? kButtonDown : kButtonUp);

   // Replace with real variable values...

   fNumberOfPlanes->SetNumber(fPainter->GetNContours());
   fAlpha->SetNumber(fPainter->GetAlpha());

   fSlideRange->SetRange(fPainter->GetV5PredictedMin(), fPainter->GetV5PredictedMax());
   fSlideRange->SetPosition(fPainter->GetV5SliderMin(), fPainter->GetV5SliderMax());

   //fLogScale->SetState(fPainter->IsLogScale() ? kButtonDown : kButtonUp);

   if (fInit)
      ConnectSignals2Slots();

}
