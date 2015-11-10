// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <string>
#include <map>

#include "TGDoubleSlider.h"
#include "TGColorSelect.h"
#include "TGNumberEntry.h"
#include "TVirtualPad.h"
#include "TGListBox.h"
#include "TGSlider.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TString.h"
#include "TColor.h"
#include "TAxis.h"

#include "TGL5DDataSetEditor.h"
#include "TGL5DPainter.h"
#include "TGLUtil.h"
#include "TGL5D.h"

/** \class TGL5DDataSetEditor
\ingroup opengl
GUI editor for OpenGL 5D Painter.
Exception safety and ROOT's GUI are two
mutually exclusive things. So, only ROOT's GUI here.
*/

namespace {

typedef TGL5DPainter::SurfIter_t    SurfIter_t;
typedef std::map<Int_t, SurfIter_t> IterMap_t;
typedef IterMap_t::iterator         IterMapIter_t;

}

//
//Pimpl.
//
class TGL5DDataSetEditor::TGL5DEditorPrivate {
public:
   IterMap_t fIterators;
   Bool_t IsValid(Int_t index)const
   {
      return fIterators.find(index) != fIterators.end();
   }
};

ClassImp(TGL5DDataSetEditor);

////////////////////////////////////////////////////////////////////////////////

TGL5DDataSetEditor::TGL5DDataSetEditor(const TGWindow *p,  Int_t width, Int_t height,
                                       UInt_t options, Pixel_t back) :
   TGedFrame(p,  width, height, options | kVerticalFrame, back),
   //"Grid" tab.
   fNCellsXEntry(0),
   fNCellsYEntry(0),
   fNCellsZEntry(0),
   fXRangeSlider(0),
   fXRangeSliderMin(0),
   fXRangeSliderMax(0),
   fYRangeSlider(0),
   fYRangeSliderMin(0),
   fYRangeSliderMax(0),
   fZRangeSlider(0),
   fZRangeSliderMin(0),
   fZRangeSliderMax(0),
   fCancelGridBtn(0),
   fOkGridBtn(0),
   //"Surfaces" tab.
   fV4MinEntry(0),
   fV4MaxEntry(0),
   fHighlightCheck(0),
   fIsoList(0),
   fVisibleCheck(0),
   fShowCloud(0),
   fSurfColorSelect(0),
   fSurfAlphaSlider(0),
   fSurfRemoveBtn(0),
   fNewIsoEntry(0),
   fAddNewIsoBtn(0),
   //"Style" tab's widgets.
   fShowBoxCut(),
   fNumberOfPlanes(0),
   fAlpha(0),
   fLogScale(0),
   fSlideRange(0),
   fApplyAlpha(0),
   fApplyPlanes(0),
   //Model.
   fDataSet(0),
   fPainter(0),
   fHidden(0),
   fSelectedSurface(-1)
{
   //Constructor.
   CreateStyleTab();
   CreateGridTab();
   CreateIsoTab();

   fHidden = new TGL5DEditorPrivate;
}

//______________________________________________________________________________

TGL5DDataSetEditor::~TGL5DDataSetEditor()
{
   //Destructor.
   delete fHidden;
}

////////////////////////////////////////////////////////////////////////////////
///Connect signals to slots.

void TGL5DDataSetEditor::ConnectSignals2Slots()
{
   //Controls from "Style" tab.
   fShowBoxCut->Connect("Toggled(Bool_t)", "TGL5DDataSetEditor", this, "BoxCutToggled()");
   fAlpha->Connect("ValueChanged(Long_t)", "TGL5DDataSetEditor", this, "AlphaChanged()");
   fAlpha->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "AlphaChanged()");
   fNumberOfPlanes->Connect("ValueChanged(Long_t)", "TGL5DDataSetEditor", this, "NContoursChanged()");
   fNumberOfPlanes->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "NContoursChanged()");
   fApplyPlanes->Connect("Clicked()", "TGL5DDataSetEditor", this, "ApplyPlanes()");
   fApplyAlpha->Connect("Clicked()", "TGL5DDataSetEditor", this, "ApplyAlpha()");

   //Controls from "Grid" tab.
   fNCellsXEntry->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "GridParametersChanged()");
   fNCellsXEntry->Connect("ValueChanged(Long_t)", "TGL5DDataSetEditor", this, "GridParametersChanged()");

   fNCellsYEntry->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "GridParametersChanged()");
   fNCellsZEntry->Connect("ValueSet(Long_t)", "TGL5DDataSetEditor", this, "GridParametersChanged()");

   fXRangeSlider->Connect("PositionChanged()", "TGL5DDataSetEditor", this, "XSliderChanged()");
   fXRangeSliderMin->Connect("ReturnPressed()", "TGL5DDataSetEditor", this, "XSliderSetMin()");
   fXRangeSliderMax->Connect("ReturnPressed()", "TGL5DDataSetEditor", this, "XSliderSetMax()");

   fYRangeSlider->Connect("PositionChanged()", "TGL5DDataSetEditor", this, "YSliderChanged()");
   fYRangeSliderMin->Connect("ReturnPressed()", "TGL5DDataSetEditor", this, "YSliderSetMin()");
   fYRangeSliderMax->Connect("ReturnPressed()", "TGL5DDataSetEditor", this, "YSliderSetMax()");

   fZRangeSlider->Connect("PositionChanged()", "TGL5DDataSetEditor", this, "ZSliderChanged()");
   fZRangeSliderMin->Connect("ReturnPressed()", "TGL5DDataSetEditor", this, "ZSliderSetMin()");
   fZRangeSliderMax->Connect("ReturnPressed()", "TGL5DDataSetEditor", this, "ZSliderSetMax()");

   fCancelGridBtn->Connect("Pressed()", "TGL5DDataSetEditor", this, "RollbackGridParameters()");
   fOkGridBtn->Connect("Pressed()", "TGL5DDataSetEditor", this, "ApplyGridParameters()");

   //Controls from "Surfaces" tab.
   fIsoList->Connect("Selected(Int_t)", "TGL5DDataSetEditor", this, "SurfaceSelected(Int_t)");
   fIsoList->GetContainer()->RemoveInput(kKeyPressMask);

   fHighlightCheck->Connect("Clicked()", "TGL5DDataSetEditor", this, "HighlightClicked()");
   fVisibleCheck->Connect("Clicked()", "TGL5DDataSetEditor", this, "VisibleClicked()");
   fSurfColorSelect->Connect("ColorSelected(Pixel_t)", "TGL5DDataSetEditor", this, "ColorChanged(Pixel_t)");
   fSurfAlphaSlider->Connect("PositionChanged(Int_t)", "TGL5DDataSetEditor", this, "AlphaChanged(Int_t)");
   fSurfRemoveBtn->Connect("Pressed()", "TGL5DDataSetEditor", this, "RemoveSurface()");

   fAddNewIsoBtn->Connect("Pressed()", "TGL5DDataSetEditor", this, "AddNewSurface()");

   fInit = kFALSE;
}


namespace
{

// Auxiliary functions.

////////////////////////////////////////////////////////////////////////////////

void make_slider_range_entries(TGCompositeFrame *parent, TGNumberEntryField *&minEntry,
                               const TString &minToolTip, TGNumberEntryField *&maxEntry,
                               const TString &maxToolTip)
{
   TGCompositeFrame *frame = new TGCompositeFrame(parent, 80, 20, kHorizontalFrame);

   minEntry = new TGNumberEntryField(frame, -1, 0., TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEAAnyNumber);
   minEntry->SetToolTipText(minToolTip.Data());
   minEntry->Resize(57, 20);
   frame->AddFrame(minEntry, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));

   maxEntry = new TGNumberEntryField(frame, -1, 0., TGNumberFormat::kNESRealThree,
                                     TGNumberFormat::kNEAAnyNumber);
   maxEntry->SetToolTipText(maxToolTip.Data());
   maxEntry->Resize(57, 20);
   frame->AddFrame(maxEntry, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   parent->AddFrame(frame, new TGLayoutHints(kLHintsTop, 5, 0, 0, 0));
}

////////////////////////////////////////////////////////////////////////////////

TGHorizontalFrame *make_labeled_hframe(TGCompositeFrame *p, const char *text)
{
   TGHorizontalFrame *frame = new TGHorizontalFrame(p);
   TGLabel *label = new TGLabel(frame, text);
   frame->AddFrame(label, new TGLayoutHints(kLHintsLeft | kLHintsBottom, 0, 0, 0));
   p->AddFrame(frame, new TGLayoutHints(kLHintsLeft, 0, 0, 1, 0));

   return frame;
}

////////////////////////////////////////////////////////////////////////////////

TGDoubleHSlider *make_double_hslider(TGCompositeFrame *parent, const char *labelName)
{
   TGCompositeFrame *sliderFrame = new TGCompositeFrame(parent, 80, 20, kHorizontalFrame);
   TGLabel *sliderLabel = new TGLabel(sliderFrame, labelName);
   sliderFrame->AddFrame(sliderLabel,
                         new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));
   TGDoubleHSlider *slider = new TGDoubleHSlider(sliderFrame, 1, 2);
   slider->Resize(110, 20);

   sliderFrame->AddFrame(slider, new TGLayoutHints(kLHintsLeft));
   parent->AddFrame(sliderFrame, new TGLayoutHints(kLHintsTop, 2, 2, 2, 2));

   return slider;
}

}

////////////////////////////////////////////////////////////////////////////////
/// Creates "Style" tab.

void TGL5DDataSetEditor::CreateStyleTab()
{
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
   //fApplyPlanes->SetState(kButtonDisabled);

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

////////////////////////////////////////////////////////////////////////////////
///Tab, containing controls to set
///the ranges and number of cells in a grid.

void TGL5DDataSetEditor::CreateGridTab()
{
   TGCompositeFrame *tabFrame = CreateEditorTabSubFrame("Grid");
   //1. The first part of the tab - "Grid parameters" group.
   TGGroupFrame *gridGroup = new TGGroupFrame(tabFrame, "Grid parameters", kVerticalFrame);
   //2. Numeric entries.
   const UInt_t min = 10, max = 300;
   const UInt_t nDigits = 4;

   TGHorizontalFrame *frame = make_labeled_hframe(gridGroup, "Cells along X:");
   fNCellsXEntry = new TGNumberEntry(frame, 0., nDigits, -1, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEAPositive, TGNumberFormat::kNELLimitMinMax,
                                     min, max);
   frame->AddFrame(fNCellsXEntry,
                   new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsBottom, 2, 0, 0));
   //
   frame = make_labeled_hframe(gridGroup, "Cells along Y:");
   fNCellsYEntry = new TGNumberEntry(frame, 0., nDigits, -1, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEAPositive, TGNumberFormat::kNELLimitMinMax,
                                     min, max);
   frame->AddFrame(fNCellsYEntry,
                   new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsBottom, 2, 0, 0));
   //
   frame = make_labeled_hframe(gridGroup, "Cells along Z:");
   fNCellsZEntry = new TGNumberEntry(frame, 0., nDigits, -1, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEAPositive, TGNumberFormat::kNELLimitMinMax,
                                     min, max);
   frame->AddFrame(fNCellsZEntry,
                   new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsBottom, 2, 0, 0));
   tabFrame->AddFrame(gridGroup, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2, 3, 3, 0));

   //3. The second part - "Ranges" group.
   TGGroupFrame *rangeGroup = new TGGroupFrame(tabFrame, "Ranges", kVerticalFrame);
   //4. Sliders and number entry fields.
   fXRangeSlider = make_double_hslider(rangeGroup, "X:");
   make_slider_range_entries(rangeGroup, fXRangeSliderMin, "Set the minimum value of the x-axis",
                          fXRangeSliderMax, "Set the maximum value of the x-axis");
   fYRangeSlider = make_double_hslider(rangeGroup, "Y:");
   make_slider_range_entries(rangeGroup, fYRangeSliderMin, "Set the minimum value of the y-axis",
                          fYRangeSliderMax, "Set the maximum value of the y-axis");
   fZRangeSlider = make_double_hslider(rangeGroup, "Z:");
   make_slider_range_entries(rangeGroup, fZRangeSliderMin, "Set the minimum value of the z-axis",
                          fZRangeSliderMax, "Set the maximum value of the z-axis");

   tabFrame->AddFrame(rangeGroup, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));

   //5. Buttons.
   TGHorizontalFrame *horizontalFrame = new TGHorizontalFrame(tabFrame, 200, 50);
   fCancelGridBtn = new TGTextButton(horizontalFrame, "  Cancel  ");
   horizontalFrame->AddFrame(fCancelGridBtn, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));
   fOkGridBtn = new TGTextButton(horizontalFrame, "  Apply  ");
   horizontalFrame->AddFrame(fOkGridBtn, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));
   tabFrame->AddFrame(horizontalFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX, 2, 3, 0, 0));
}

////////////////////////////////////////////////////////////////////////////////
///Tab, containing controls to work with iso-surfaces.

void TGL5DDataSetEditor::CreateIsoTab()
{
   TGCompositeFrame *tabFrame = CreateEditorTabSubFrame("Surfaces");

   //1. The first group - contains V4 range (read only number entries with min and max).
   TGGroupFrame *v4Group = new TGGroupFrame(tabFrame, "V4 Range", kVerticalFrame);

   make_slider_range_entries(v4Group, fV4MinEntry, "Minimum value of V4",
                             fV4MaxEntry, "Maximum value of V4");

   tabFrame->AddFrame(v4Group, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));
   //
   fV4MinEntry->SetState(kFALSE);
   fV4MaxEntry->SetState(kFALSE);

   //2. The second group - contains controls to select surface and
   //manipulate its parameters.
   TGGroupFrame *isoGroup = new TGGroupFrame(tabFrame, "Iso-surfaces", kVerticalFrame);

   fHighlightCheck = new TGCheckButton(isoGroup, "Highlight selected");
   fHighlightCheck->SetToolTipText("Highlight selected surface");
   fHighlightCheck->SetState(kButtonDown);
   isoGroup->AddFrame(fHighlightCheck, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));

   TGHorizontalFrame *hf = new TGHorizontalFrame(isoGroup);
   fIsoList = new TGListBox(hf);
   fIsoList->Resize(120, 120);
   hf->AddFrame(fIsoList, new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 5));
   isoGroup->AddFrame(hf, new TGLayoutHints(kLHintsLeft, 2, 1, 1, 1));

   fVisibleCheck = new TGCheckButton(isoGroup, "Visible");
   fVisibleCheck->SetToolTipText("Show/hide surface");
   isoGroup->AddFrame(fVisibleCheck, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));

   fShowCloud = new TGCheckButton(isoGroup, "Show cloud");
   fShowCloud->SetToolTipText("Show/hide cloud for surface");
   isoGroup->AddFrame(fShowCloud, new TGLayoutHints(kLHintsLeft, 4, 1, 1, 1));

   //Sorry, Matevz :) I stole this from TGLViewerEditor :))
   hf = new TGHorizontalFrame(isoGroup);
   TGLabel* lab = new TGLabel(hf, "Color");
   hf->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 4, 8, 3));
   fSurfColorSelect = new TGColorSelect(hf, 0, -1);
   hf->AddFrame(fSurfColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 8, 1));
   isoGroup->AddFrame(hf, new TGLayoutHints(kLHintsLeft, 2, 1, 1, 1));

   TGHorizontalFrame *frame = make_labeled_hframe(isoGroup, "Opacity: ");
   fSurfAlphaSlider = new TGHSlider(frame, 80);
   fSurfAlphaSlider->SetRange(0, 100);
   frame->AddFrame(fSurfAlphaSlider, new TGLayoutHints(kLHintsLeft));

   fSurfRemoveBtn = new TGTextButton(isoGroup, "  Remove surface  ");
   isoGroup->AddFrame(fSurfRemoveBtn, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 2));
   tabFrame->AddFrame(isoGroup, new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX, 2, 3, 0, 0));

   //3. Group with controls to add new iso-surface.
   TGGroupFrame *newGroup = new TGGroupFrame(tabFrame, "New iso-surface", kVerticalFrame);
   hf = new TGHorizontalFrame(newGroup);
   fNewIsoEntry = new TGNumberEntry(hf, 0., 12, -1, TGNumberFormat::kNESReal);
   hf->AddFrame(fNewIsoEntry, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));
   fNewIsoEntry->Resize(60, 20);
   fAddNewIsoBtn = new TGTextButton(hf, "    Add    ");
   hf->AddFrame(fAddNewIsoBtn, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX, 2, 2, 2, 2));
   newGroup->AddFrame(hf, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));

   tabFrame->AddFrame(newGroup, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 3, 0, 0));
}

////////////////////////////////////////////////////////////////////////////////
///Set model or disables/hides viewer.

void TGL5DDataSetEditor::SetModel(TObject* obj)
{
   fPainter = 0;
   Bool_t needUpdate = fSelectedSurface != -1;

   if ((fDataSet = dynamic_cast<TGL5DDataSet *>(obj))) {
      fPainter = fDataSet->GetRealPainter();

      SetStyleTabWidgets();
      SetGridTabWidgets();
      SetIsoTabWidgets();

      DisableGridTabButtons();
      DisableSurfaceControls();

      if (fInit)
         ConnectSignals2Slots();
   }

   if (needUpdate && gPad)
      gPad->Update();
}

namespace {

void set_grid_range_widgets(const TAxis *a, const Rgl::Range_t r, TGDoubleHSlider *slider,
                            TGNumberEntryField *eMin, TGNumberEntryField *eMax)
{
   slider->SetRange(r.first, r.second);
   slider->SetPosition(a->GetBinLowEdge(1), a->GetBinUpEdge(a->GetLast()));

   eMin->SetNumber(a->GetBinLowEdge(1));
   eMin->SetLimits(TGNumberFormat::kNELLimitMinMax, r.first, r.second);
   eMax->SetNumber(a->GetBinUpEdge(a->GetLast()));
   eMax->SetLimits(TGNumberFormat::kNELLimitMinMax, r.first, r.second);
}

}

////////////////////////////////////////////////////////////////////////////////
///Set "Style" tab's controls from model.

void TGL5DDataSetEditor::SetStyleTabWidgets()
{
   fShowBoxCut->SetState(fPainter->IsBoxCutShown() ? kButtonDown : kButtonUp);
   fNumberOfPlanes->SetNumber(fPainter->GetNContours());
   fAlpha->SetNumber(fPainter->GetAlpha());
}

////////////////////////////////////////////////////////////////////////////////
///Set "Grid" tab's controls from model.

void TGL5DDataSetEditor::SetGridTabWidgets()
{
   const TAxis *xA = fDataSet->GetXAxis();
   const TAxis *yA = fDataSet->GetYAxis();
   const TAxis *zA = fDataSet->GetZAxis();
   const Rgl::Range_t &xR = fDataSet->GetXRange();
   const Rgl::Range_t &yR = fDataSet->GetYRange();
   const Rgl::Range_t &zR = fDataSet->GetZRange();
   //Number of cells.
   fNCellsXEntry->SetIntNumber(xA->GetNbins());
   fNCellsYEntry->SetIntNumber(yA->GetNbins());
   fNCellsZEntry->SetIntNumber(zA->GetNbins());
   //X-range.
   set_grid_range_widgets(xA, xR, fXRangeSlider, fXRangeSliderMin, fXRangeSliderMax);
   //Y-range.
   set_grid_range_widgets(yA, yR, fYRangeSlider, fYRangeSliderMin, fYRangeSliderMax);
   //Z-range.
   set_grid_range_widgets(zA, zR, fZRangeSlider, fZRangeSliderMin, fZRangeSliderMax);
}

////////////////////////////////////////////////////////////////////////////////
///Set "Surfaces" tab's controls from model.

void TGL5DDataSetEditor::SetIsoTabWidgets()
{
   const Rgl::Range_t &v4R = fDataSet->GetV4Range();
   //V4 range.
   fV4MinEntry->SetNumber(v4R.first);
   fV4MaxEntry->SetNumber(v4R.second);

   fIsoList->RemoveAll();
   fHidden->fIterators.clear();

   SurfIter_t curr = fPainter->SurfacesBegin();

   for (Int_t ind = 0; curr != fPainter->SurfacesEnd(); ++curr, ++ind) {
      TString entry(TString::Format("Level: %f", curr->f4D));
      fIsoList->AddEntry(entry.Data(), ind);
      fIsoList->Layout();
      curr->fHighlight = kFALSE;
      //I'm saving list's iterators here.
      //If list modified (surface removed)
      //- corresponding iterator must be removed,
      //all other iterators are still valid (thanks to std::list).
      //If surface added, new iterator must be added at the end.
      fHidden->fIterators[ind] = curr;
   }

   fNewIsoEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, v4R.first, v4R.second);
   fNewIsoEntry->SetNumber(v4R.first);

   fSelectedSurface = -1;
}

////////////////////////////////////////////////////////////////////////////////
///Some of controls in a "Grid" tab was modified.

void TGL5DDataSetEditor::GridParametersChanged()
{
   EnableGridTabButtons();
}

////////////////////////////////////////////////////////////////////////////////
///Grid parameters were changed, enable "Cancel" and "Apply" buttons.

void TGL5DDataSetEditor::EnableGridTabButtons()
{
   fCancelGridBtn->SetState(kButtonUp);
   fOkGridBtn->SetState(kButtonUp);
}

////////////////////////////////////////////////////////////////////////////////
///Disable "Cancel" and "Apply" buttons.

void TGL5DDataSetEditor::DisableGridTabButtons()
{
   fCancelGridBtn->SetState(kButtonDisabled);
   fOkGridBtn->SetState(kButtonDisabled);
}

////////////////////////////////////////////////////////////////////////////////
///Surface was selected in a list box, enable some controls.

void TGL5DDataSetEditor::EnableSurfaceControls()
{
   fVisibleCheck->SetState(kButtonUp);
//   fShowCloud->SetState(kButtonUp);
//   fSurfColorBtn->SetState(kButtonUp);
   fSurfRemoveBtn->SetState(kButtonUp);
}

////////////////////////////////////////////////////////////////////////////////
///Disable surface controls.

void TGL5DDataSetEditor::DisableSurfaceControls()
{
   fVisibleCheck->SetState(kButtonDisabled);
   fShowCloud->SetState(kButtonDisabled);
//   fSurfColorBtn->SetState(kButtonDisabled);
   fSurfRemoveBtn->SetState(kButtonDisabled);
}

////////////////////////////////////////////////////////////////////////////////
///X slider in a "Grid" tab.

void TGL5DDataSetEditor::XSliderChanged()
{
   fXRangeSliderMin->SetNumber(fXRangeSlider->GetMinPosition());
   fXRangeSliderMax->SetNumber(fXRangeSlider->GetMaxPosition());

   EnableGridTabButtons();
}

////////////////////////////////////////////////////////////////////////////////
///Y slider in a "Grid" tab.

void TGL5DDataSetEditor::YSliderChanged()
{
   fYRangeSliderMin->SetNumber(fYRangeSlider->GetMinPosition());
   fYRangeSliderMax->SetNumber(fYRangeSlider->GetMaxPosition());

   EnableGridTabButtons();
}

////////////////////////////////////////////////////////////////////////////////
///Z slider in a "Grid" tab.

void TGL5DDataSetEditor::ZSliderChanged()
{
   fZRangeSliderMin->SetNumber(fZRangeSlider->GetMinPosition());
   fZRangeSliderMax->SetNumber(fZRangeSlider->GetMaxPosition());

   EnableGridTabButtons();
}

////////////////////////////////////////////////////////////////////////////////
///Value in a number entry was modified.

void TGL5DDataSetEditor::XSliderSetMin()
{
   if (fXRangeSliderMin->GetNumber() < fXRangeSliderMax->GetNumber()) {
      fXRangeSlider->SetPosition(fXRangeSliderMin->GetNumber(),
                                 fXRangeSliderMax->GetNumber());
      EnableGridTabButtons();
   } else
      fXRangeSliderMin->SetNumber(fXRangeSlider->GetMinPosition());
}

////////////////////////////////////////////////////////////////////////////////
///Value in a number entry was modified.

void TGL5DDataSetEditor::XSliderSetMax()
{
   if (fXRangeSliderMin->GetNumber() < fXRangeSliderMax->GetNumber()) {
      fXRangeSlider->SetPosition(fXRangeSliderMin->GetNumber(),
                                 fXRangeSliderMax->GetNumber());
      EnableGridTabButtons();
   } else
      fXRangeSliderMax->SetNumber(fXRangeSlider->GetMaxPosition());
}


////////////////////////////////////////////////////////////////////////////////
///Value in a number entry was modified.

void TGL5DDataSetEditor::YSliderSetMin()
{
   if (fYRangeSliderMin->GetNumber() < fYRangeSliderMax->GetNumber()) {
      fYRangeSlider->SetPosition(fYRangeSliderMin->GetNumber(),
                                 fYRangeSliderMax->GetNumber());
      EnableGridTabButtons();
   } else
      fYRangeSliderMin->SetNumber(fYRangeSlider->GetMinPosition());
}

////////////////////////////////////////////////////////////////////////////////
///Value in a number entry was modified.

void TGL5DDataSetEditor::YSliderSetMax()
{
   if (fYRangeSliderMin->GetNumber() < fYRangeSliderMax->GetNumber()) {
      fYRangeSlider->SetPosition(fYRangeSliderMin->GetNumber(),
                                 fYRangeSliderMax->GetNumber());
      EnableGridTabButtons();
   } else
      fYRangeSliderMax->SetNumber(fYRangeSlider->GetMaxPosition());
}

////////////////////////////////////////////////////////////////////////////////
///Value in a number entry was modified.

void TGL5DDataSetEditor::ZSliderSetMin()
{
   if (fZRangeSliderMin->GetNumber() < fZRangeSliderMax->GetNumber()) {
      fZRangeSlider->SetPosition(fZRangeSliderMin->GetNumber(),
                                 fZRangeSliderMax->GetNumber());
      EnableGridTabButtons();
   } else
      fZRangeSliderMin->SetNumber(fZRangeSlider->GetMinPosition());

}

////////////////////////////////////////////////////////////////////////////////
///Value in a number entry was modified.

void TGL5DDataSetEditor::ZSliderSetMax()
{
   if (fZRangeSliderMin->GetNumber() < fZRangeSliderMax->GetNumber()) {
      fZRangeSlider->SetPosition(fZRangeSliderMin->GetNumber(),
                                 fZRangeSliderMax->GetNumber());
      EnableGridTabButtons();
   } else
      fYRangeSliderMax->SetNumber(fZRangeSlider->GetMaxPosition());
}

////////////////////////////////////////////////////////////////////////////////
///"Cancel" button was pressed in a "Grid" tab.
///Return old values.

void TGL5DDataSetEditor::RollbackGridParameters()
{
   SetGridTabWidgets();
   DisableGridTabButtons();
}

////////////////////////////////////////////////////////////////////////////////
///"Apply" button was pressed in a "Grid" tab.
///Modify all meshes.

void TGL5DDataSetEditor::ApplyGridParameters()
{
   DisableGridTabButtons();
   //
   fDataSet->GetXAxis()->Set(fNCellsXEntry->GetIntNumber(),
                             fXRangeSlider->GetMinPosition(),
                             fXRangeSlider->GetMaxPosition());

   fDataSet->GetYAxis()->Set(fNCellsYEntry->GetIntNumber(),
                             fYRangeSlider->GetMinPosition(),
                             fYRangeSlider->GetMaxPosition());

   fDataSet->GetZAxis()->Set(fNCellsZEntry->GetIntNumber(),
                             fZRangeSlider->GetMinPosition(),
                             fZRangeSlider->GetMaxPosition());

   fPainter->ResetGeometryRanges();
   if (gPad)
      gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
///Check, if selected surface must be highlighted.

void TGL5DDataSetEditor::HighlightClicked()
{
   if (fSelectedSurface == -1)
      return;

   fHidden->fIterators[fSelectedSurface]->fHighlight = fHighlightCheck->IsOn();

   if (gPad)
      gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
///Surface was selected in a list box.
///Enable surface controls and set them into
///correct state.

void TGL5DDataSetEditor::SurfaceSelected(Int_t id)
{
   if (id >= 0) {
      //Check, if the index is valid.
      if (!fHidden->IsValid(id)) {
         Error("SurfaceSelected", "Got wrong index %d", id);
         return;
      }

      if (fSelectedSurface != -1) {
         //Previously selected surface IS ALWAYS
         //valid index, so no index check here.
         fHidden->fIterators[fSelectedSurface]->fHighlight = kFALSE;
      }

      EnableSurfaceControls();


      SurfIter_t surf = fHidden->fIterators[fSelectedSurface = id];
      surf->fHighlight = fHighlightCheck->IsOn();
      //Surface is visible/invisible - check/uncheck.
      fVisibleCheck->SetOn(!surf->fHide);
      fSurfColorSelect->SetColor(TColor::Number2Pixel(surf->fColor), kFALSE);
      fSurfAlphaSlider->SetPosition(surf->fAlpha);

      if (gPad)
         gPad->Update();
   } else if (fSelectedSurface != -1) {
      //Deselect.
      fHidden->fIterators[fSelectedSurface]->fHighlight = kFALSE;
      fSelectedSurface = -1;
      DisableSurfaceControls();//No surface is selected, no working controls.
      if (gPad)
         gPad->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
///Hide/show selected surface.

void TGL5DDataSetEditor::VisibleClicked()
{
   //In principle, this control can be enabled,
   //only if some surface was selected and
   //fSelectedSurface != -1. But I do not trust to
   //ROOT's GUI so I have a check.
   if (fSelectedSurface != -1) {
      fHidden->fIterators[fSelectedSurface]->fHide = !(fVisibleCheck->IsOn());
      if (gPad)
         gPad->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
///Change the color of the selected surface.

void TGL5DDataSetEditor::ColorChanged(Pixel_t pixel)
{
   if (fSelectedSurface != -1) {
      fHidden->fIterators[fSelectedSurface]->fColor = Color_t(TColor::GetColor(pixel));
      if (gPad)
         gPad->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
///Change transparency of selected surface.

void TGL5DDataSetEditor::AlphaChanged(Int_t alpha)
{
   if (fSelectedSurface != -1) {
      fHidden->fIterators[fSelectedSurface]->fAlpha = alpha;
      if (gPad)
         gPad->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
///Remove selected surface.

void TGL5DDataSetEditor::RemoveSurface()
{
   if (fSelectedSurface != -1) {

      SurfIter_t it = fHidden->fIterators[fSelectedSurface];
      fHidden->fIterators.erase(fSelectedSurface);
      fIsoList->RemoveEntry(fSelectedSurface);
      fIsoList->Layout();
      fPainter->RemoveSurface(it);
      DisableSurfaceControls();
      fSelectedSurface = -1;

      if (gPad)
         gPad->Update();
   }
}

////////////////////////////////////////////////////////////////////////////////
///Add new iso-surface.

void TGL5DDataSetEditor::AddNewSurface()
{
   fPainter->AddSurface(fNewIsoEntry->GetNumber());
   SetModel(fDataSet);

   if (gPad)
      gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the "Apply" button for alpha value.

void TGL5DDataSetEditor::ApplyAlpha()
{
   if (fPainter) {
      fApplyAlpha->SetState(kButtonDisabled);
      fPainter->SetAlpha(fAlpha->GetNumber());
      fAlpha->SetNumber(fPainter->GetAlpha());

      //Update other tabs and change controls' states.
      SetModel(fDataSet);
   }

   if (gPad)
      gPad->Update();
}


////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the Apply Planes button.

void TGL5DDataSetEditor::ApplyPlanes()
{
   if (fPainter) {
      //fApplyPlanes->SetState(kButtonDisabled);
      fPainter->SetNContours((Int_t)fNumberOfPlanes->GetIntNumber());
      fNumberOfPlanes->SetIntNumber(fPainter->GetNContours());

      //Update other tabs and change controls' states.
      SetModel(fDataSet);
   }

   if (gPad)
      gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the Show BoxCut check button.

void TGL5DDataSetEditor::BoxCutToggled()
{
   if (fPainter)
      fPainter->ShowBoxCut(fShowBoxCut->IsOn());
   if (gPad)
      gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the Alpha entry.

void TGL5DDataSetEditor::AlphaChanged()
{
   fApplyAlpha->SetState(kButtonUp);
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the Number of Planes value-entry.

void TGL5DDataSetEditor::NContoursChanged()
{
//   fApplyPlanes->SetState(kButtonUp);
}
