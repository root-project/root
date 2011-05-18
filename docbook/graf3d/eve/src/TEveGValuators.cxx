// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGValuators.h"

#include "TMath.h"
#include "TGLabel.h"
#include "TGSlider.h"
#include "TGDoubleSlider.h"


/******************************************************************************/
// TEveGValuatorBase
/******************************************************************************/

//______________________________________________________________________________
//
// Base class for composite GUI elements for setting of numeric
// values.

ClassImp(TEveGValuatorBase);

//______________________________________________________________________________
TEveGValuatorBase::TEveGValuatorBase(const TGWindow *p, const char* name,
                                     UInt_t w, UInt_t h, Int_t widgetId) :
   TGCompositeFrame(p, w, h), TGWidget(widgetId),

   fLabelWidth (0),
   fAlignRight (kFALSE),
   fShowSlider (kTRUE),

   fNELength (5),
   fNEHeight (20),

   fLabel (0)
{
   // Constructor.

   SetName(name);
}


/******************************************************************************/
// TEveGValuator
/******************************************************************************/

//______________________________________________________________________________
//
// Composite GUI element for single value selection (supports label,
// number-entry and slider).

ClassImp(TEveGValuator);

//______________________________________________________________________________
TEveGValuator::TEveGValuator(const TGWindow *p, const char* title,
                             UInt_t w, UInt_t h, Int_t widgetId) :
   TEveGValuatorBase(p, title, w, h, widgetId),

   fValue (0),
   fMin   (0),
   fMax   (0),

   fSliderNewLine (kFALSE),
   fSliderDivs    (-1),
   fEntry  (0),
   fSlider (0)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveGValuator::Build(Bool_t connect)
{
   // Create sub-components (label, number entry, slider).

   TGCompositeFrame *hf1, *hfs;
   if(fShowSlider && fSliderNewLine) {
      SetLayoutManager(new TGVerticalLayout(this));
      hf1 = new TGHorizontalFrame(this);
      hf1->SetLayoutManager(new TGHorizontalLayout(hf1));
      AddFrame(hf1, new TGLayoutHints(kLHintsTop, 0,0,0,0));
      hfs = new TGHorizontalFrame(this);
      hfs->SetLayoutManager(new TGHorizontalLayout(hfs));
      AddFrame(hfs, new TGLayoutHints(kLHintsTop, 0,0,0,0));
   } else {
      hf1 = this;
      hfs = this;
      SetLayoutManager(new TGHorizontalLayout(this));
   }

   // label
   {
      TGLayoutHints *labh, *labfrh;
      if(fAlignRight) {
         labh   = new TGLayoutHints(kLHintsRight | kLHintsBottom, 0,0,0,0);
         labfrh = new TGLayoutHints(kLHintsRight);
      } else {
         labh   = new TGLayoutHints(kLHintsLeft  | kLHintsBottom, 0,0,0,0);
         labfrh = new TGLayoutHints(kLHintsLeft);
      }
      TGCompositeFrame *labfr =
         new TGHorizontalFrame(hf1, fLabelWidth, fNEHeight,
                               fLabelWidth != 0 ? kFixedSize : kFixedHeight);
      fLabel = new TGLabel(labfr, fName);
      labfr->AddFrame(fLabel, labh);
      hf1->AddFrame(labfr, labfrh);
   }

   // number-entry
   TGLayoutHints*  elh =  new TGLayoutHints(kLHintsLeft, 0,0,0,0);
   fEntry = new TGNumberEntry(hf1, 0, fNELength);
   fEntry->SetHeight(fNEHeight);
   fEntry->GetNumberEntry()->SetToolTipText("Enter Slider Value");
   hf1->AddFrame(fEntry, elh);

   if (connect)
      fEntry->Connect("ValueSet(Long_t)",
                      "TEveGValuator", this, "EntryCallback()");

   // slider
   if(fShowSlider) {
      fSlider = new TGHSlider(hfs, GetWidth(), kSlider1 | kScaleBoth);
      hfs->AddFrame(fSlider, new TGLayoutHints(kLHintsLeft|kLHintsTop, 1,1,0,0));

      if (connect)
         fSlider->Connect("PositionChanged(Int_t)",
                          "TEveGValuator", this, "SliderCallback()");
   }
}

//______________________________________________________________________________
void TEveGValuator::SetLimits(Float_t min, Float_t max, Int_t npos,
                              TGNumberFormat::EStyle nef)
{
   // Set limits of the represented value.

   fMin = Float_t(min);
   fMax = Float_t(max);
   fEntry->SetFormat(nef);
   fEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, min, max);

   if(fSlider) {
      fSliderDivs = npos - 1;
      fSlider->SetRange(0, fSliderDivs);
   }
}

//______________________________________________________________________________
void TEveGValuator::SetLimits(Int_t min, Int_t max)
{
   // Set limits of the represented value for integer values.

   fMin = Float_t(min);
   fMax = Float_t(max);
   fEntry->SetFormat(TGNumberFormat::kNESInteger);
   fEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, min, max);

   if(fSlider) {
      fSliderDivs = max - min;
      fSlider->SetRange(0, fSliderDivs);
   }
}

//______________________________________________________________________________
Int_t TEveGValuator::CalcSliderPos(Float_t v)
{
   // Return slider position for given value.

   return (Int_t) TMath::Nint((v - fMin)*fSliderDivs/(fMax - fMin));
}

//______________________________________________________________________________
void TEveGValuator::EntryCallback()
{
   // Callback for change in number-entry.

   fValue = fEntry->GetNumber();
   if(fSlider) {
      fSlider->SetPosition(CalcSliderPos(fValue));
   }
   ValueSet(fValue);
}

//______________________________________________________________________________
void TEveGValuator::SliderCallback()
{
   // Callback for change in slider position.

   fValue = fMin + fSlider->GetPosition()*(fMax-fMin)/fSliderDivs;
   fEntry->SetNumber(fValue);
   ValueSet(fValue);
}


//______________________________________________________________________________
void TEveGValuator::ValueSet(Double_t val)
{
   // Emit "ValueSet(Double_t)" signal.

   Emit("ValueSet(Double_t)", val);
}

//______________________________________________________________________________
void TEveGValuator::SetValue(Float_t val, Bool_t emit)
{
   // Set value, optionally emit signal.

   fValue = val;
   fEntry->SetNumber(fValue);

   if(fSlider){
      fSlider->SetPosition(CalcSliderPos(fValue));
   }
   if(emit)
      ValueSet(val);
}

//______________________________________________________________________________
void TEveGValuator::SetToolTip(const char* tip)
{
   // Set the tooltip of the number-entry.

   fEntry->GetNumberEntry()->SetToolTipText(tip);
}

//______________________________________________________________________________
void TEveGValuator::SetEnabled(Bool_t state)
{
   // Set enabled state of the whole widget.

   fEntry->GetNumberEntry()->SetEnabled(state);
   fEntry->GetButtonUp()->SetEnabled(state);
   fEntry->GetButtonDown()->SetEnabled(state);
   if(fSlider) {
      if(state) fSlider->MapWindow();
      else      fSlider->UnmapWindow();
   }
}


/******************************************************************************/
// TEveGDoubleValuator
/******************************************************************************/

//______________________________________________________________________________
//
// Composite GUI element for selection of range (label, two
// number-entries and double-slider).

ClassImp(TEveGDoubleValuator);

//______________________________________________________________________________
TEveGDoubleValuator::TEveGDoubleValuator(const TGWindow *p, const char* title,
                                         UInt_t w, UInt_t h, Int_t widgetId) :
   TEveGValuatorBase(p, title, w, h, widgetId),

   fMinEntry(0),
   fMaxEntry(0),
   fSlider(0)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveGDoubleValuator::Build(Bool_t connect)
{
   // Create sub-components (label, number entries, double-slider).

   TGCompositeFrame *hf1, *hfs;
   if(fShowSlider) {
      SetLayoutManager(new TGVerticalLayout(this));
      hf1 = new TGHorizontalFrame(this);
      hf1->SetLayoutManager(new TGHorizontalLayout(hf1));
      AddFrame(hf1, new TGLayoutHints(kLHintsTop));
      hfs = new TGHorizontalFrame(this);
      hfs->SetLayoutManager(new TGHorizontalLayout(hfs));
      AddFrame(hfs, new TGLayoutHints(kLHintsTop));
   } else {
      hf1 = this;
      hfs = this;
      SetLayoutManager(new TGHorizontalLayout(this));
   }

   // label
   TGLayoutHints* lh;
   if(fAlignRight)
      lh = new TGLayoutHints(kLHintsRight | kLHintsBottom, 4,0,0,0);
   else
      lh = new TGLayoutHints(kLHintsLeft  | kLHintsBottom, 0,4,0,0);

   if(fLabelWidth > 0) {
      TGCompositeFrame *lf = new TGHorizontalFrame(hf1, fLabelWidth, fNEHeight, kFixedSize);
      fLabel = new TGLabel(lf, fName);
      lf->AddFrame(fLabel, lh);
      // add label frame to top horizontal frame
      TGLayoutHints* lfh = new TGLayoutHints(kLHintsLeft, 0,0,0,0);
      hf1->AddFrame(lf, lfh);
   } else {
      fLabel = new TGLabel(hf1, fName);
      hf1->AddFrame(fLabel, lh);
   }

   // entries
   fMinEntry = new TGNumberEntry(hf1, 0, fNELength);
   fMinEntry->SetHeight(fNEHeight);
   fMinEntry->GetNumberEntry()->SetToolTipText("Enter Slider Min Value");
   hf1->AddFrame(fMinEntry, new TGLayoutHints(kLHintsLeft, 0,0,0,0));
   if (connect)
      fMinEntry->Connect("ValueSet(Long_t)",
                         "TEveGDoubleValuator", this, "MinEntryCallback()");

   fMaxEntry = new TGNumberEntry(hf1, 0, fNELength);
   fMaxEntry->SetHeight(fNEHeight);
   fMaxEntry->GetNumberEntry()->SetToolTipText("Enter Slider Max Value");
   hf1->AddFrame(fMaxEntry,  new TGLayoutHints(kLHintsLeft, 2,0,0,0));
   if (connect)
      fMaxEntry->Connect("ValueSet(Long_t)",
                         "TEveGDoubleValuator", this, "MaxEntryCallback()");

   // slider
   if(fShowSlider) {
      fSlider = new TGDoubleHSlider(hfs, GetWidth(), kDoubleScaleBoth);
      hfs->AddFrame(fSlider, new TGLayoutHints(kLHintsTop|kLHintsLeft, 0,0,1,0));
      if (connect)
         fSlider->Connect("PositionChanged()",
                          "TEveGDoubleValuator", this, "SliderCallback()");
   }
}

//______________________________________________________________________________
void TEveGDoubleValuator::SetLimits(Int_t min, Int_t max)
{
   // Set limits of the represented range for integer values.

   fMinEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, min, max);
   fMinEntry->SetFormat(TGNumberFormat::kNESInteger);
   fMaxEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, min, max);
   fMaxEntry->SetFormat(TGNumberFormat::kNESInteger);

   if(fSlider) {
      fSlider->SetRange(min, max);
   }
}

//______________________________________________________________________________
void TEveGDoubleValuator::SetLimits(Float_t min, Float_t max,
                                    TGNumberFormat::EStyle nef)
{
   // Set limits of the represented range.

   //  printf("TEveGDoubleValuator::SetLimits(Float_t min, Float_t max, Int_ \n");
   fMinEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, min, max);
   fMinEntry->SetFormat(nef);
   fMaxEntry->SetLimits(TGNumberFormat::kNELLimitMinMax, min, max);
   fMaxEntry->SetFormat(nef);

   if(fSlider) fSlider->SetRange(min, max);
}

//______________________________________________________________________________
void TEveGDoubleValuator::MinEntryCallback()
{
   // Callback for change in low number-entry.

   if(GetMin() > GetMax())
      fMaxEntry->SetNumber(GetMin());
   if(fSlider) fSlider->SetPosition(GetMin(), GetMax());
   ValueSet();
}

//______________________________________________________________________________
void TEveGDoubleValuator::MaxEntryCallback()
{
   // Callback for change in high number-entry.

   if(GetMax() < GetMin())
      fMinEntry->SetNumber(GetMax());
   if(fSlider) fSlider->SetPosition(GetMin(), GetMax());
   ValueSet();
}

//______________________________________________________________________________
void TEveGDoubleValuator::SliderCallback()
{
   // Callback for change in slider position / width.

   Float_t minp, maxp;
   fSlider->GetPosition(minp, maxp);
   fMinEntry->SetNumber(minp);
   fMaxEntry->SetNumber(maxp);
   ValueSet();
}

//______________________________________________________________________________
void TEveGDoubleValuator::SetValues(Float_t min, Float_t max, Bool_t emit)
{
   // Set min/max values, optionally emit signal.

   fMinEntry->SetNumber(min);
   fMaxEntry->SetNumber(max);

   if(fSlider) fSlider->SetPosition(min, max);
   if(emit)    ValueSet();
}

//______________________________________________________________________________
void TEveGDoubleValuator::ValueSet()
{
   // Emit "ValueSet()" signal.

   Emit("ValueSet()");
}


/******************************************************************************/
// TEveGTriVecValuator
/******************************************************************************/

//______________________________________________________________________________
//
// Composite GUI element for setting three numerical values (label,
// three number-entries). All three values have the same number-format
// and value-range.

ClassImp(TEveGTriVecValuator);

//______________________________________________________________________________
TEveGTriVecValuator::TEveGTriVecValuator(const TGWindow *p, const char* name,
                                         UInt_t w, UInt_t h, Int_t widgetId) :
   TGCompositeFrame(p, w, h), TGWidget(widgetId),

   fLabelWidth (0),
   fNELength   (5),
   fNEHeight   (20)
{
   // Constructor.

   SetName(name);
}

//______________________________________________________________________________
void TEveGTriVecValuator::Build(Bool_t vertical, const char* lab0, const char* lab1, const char* lab2)
{
   // Create sub-components (label, number entries).

   if (vertical) SetLayoutManager(new TGVerticalLayout(this));
   else          SetLayoutManager(new TGHorizontalLayout(this));

   const char *labs[3] = { lab0, lab1, lab2 };
   TGLayoutHints* lh;
   for (Int_t i=0; i<3; ++i) {
      fVal[i] = new TEveGValuator(this, labs[i], 10, 0);
      fVal[i]->SetLabelWidth(fLabelWidth);
      fVal[i]->SetShowSlider(kFALSE);
      fVal[i]->SetNELength(fNELength);
      fVal[i]->SetNEHeight(fNEHeight);
      fVal[i]->Build();
      fVal[i]->Connect
         ("ValueSet(Double_t)", "TEveGTriVecValuator", this, "ValueSet()");
      if (vertical) lh = new TGLayoutHints(kLHintsTop,  1, 1, 1, 1);
      else          lh = new TGLayoutHints(kLHintsLeft|kLHintsExpandX, 1, 1, 1, 1);
      AddFrame(fVal[i], lh);
   }
}

//______________________________________________________________________________
void TEveGTriVecValuator::ValueSet()
{
   // Emit "ValueSet()" signal.

   Emit("ValueSet()");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGTriVecValuator::SetLimits(Int_t min, Int_t max)
{
   // Set limits for all three number-entries, integer values.

   for (Int_t i=0; i<3; ++i)
      fVal[i]->SetLimits(min, max);
}

//______________________________________________________________________________
void TEveGTriVecValuator::SetLimits(Float_t min, Float_t max,
                                    TGNumberFormat::EStyle nef)
{
   // Set limits for all three number-entries.

   for (Int_t i=0; i<3; ++i)
      fVal[i]->SetLimits(min, max, 0, nef);
}


