// @(#)root/gl:$Id:$
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
   fLogScale(0), fSlideRange(0)
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
   fNumberOfPlanes->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "DoNumberOfPlanes()");

   fInit = kFALSE;
}

//______________________________________________________________________________
void TGL5DDataSetEditor::CreateStyleTab()
{
   // Creates "Style" tab.

   //MakeTitle("Update behaviour");
   fShowBoxCut  = new TGCheckButton(this, "Show Box Cut");
   fShowBoxCut->SetToolTipText("Box cut. When attached to a plot, cuts away a part of it");
   AddFrame(fShowBoxCut, new TGLayoutHints(kLHintsLeft, 5, 2, 2, 2));

   TGHorizontalFrame *f = new TGHorizontalFrame(this, 100, 30);
   f->AddFrame(new TGLabel(f, "Nb of isosurfaces:"), new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 5, 2, 2, 2));
   fNumberOfPlanes = new TGNumberEntry(f, 0, 3, -1, TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative);
   fNumberOfPlanes->GetNumberEntry()->SetToolTipText("Set number of isosurfaces");
   f->AddFrame(fNumberOfPlanes, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));
   AddFrame(f, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0));

   fLogScale  = new TGCheckButton(this, "Log Scale");
   AddFrame(fLogScale, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 2, 2, 2));

   AddFrame(new TGLabel(this, "Slide Range:"), new TGLayoutHints(kLHintsLeft, 5, 2, 2, 2));
   fSlideRange = new TGDoubleHSlider(this, 200, kDoubleScaleDownRight);
   AddFrame(fSlideRange, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5, 2, 2, 2));
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
void TGL5DDataSetEditor::DoNumberOfPlanes()
{
   // Slot connected to the Number of Planes entry.

   if (fPainter) {
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
   }
   if (gPad) gPad->Update();
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

   //fNumberOfPlanes->SetNumber(fPainter->GetNbIso());

   fSlideRange->SetRange(0, 100);
   fSlideRange->SetPosition(10, 90);

   //fLogScale->SetState(fPainter->IsLogScale() ? kButtonDown : kButtonUp);

   if (fInit)
      ConnectSignals2Slots();

}

