// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCaloLegoEditor.h"
#include "TEveCalo.h"
#include "TEveGValuators.h"

#include "TColor.h"
#include "TGColorSelect.h"
#include "TGLabel.h"

//______________________________________________________________________________
// GUI editor for TEveCaloLego.
//

ClassImp(TEveCaloLegoEditor);

//______________________________________________________________________________
TEveCaloLegoEditor::TEveCaloLegoEditor(const TGWindow *p, Int_t width, Int_t height,
                                       UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM(0),
   fGridColor(0),
   fFontColor(0),

   fFontSize(0),
   fNZStep(0)
{
   // Constructor.

   MakeTitle("TEveCaloLego");

   {
      // grid color
      TGHorizontalFrame* f = new TGHorizontalFrame(this);
      TGLabel* lab = new TGLabel(f, "GridColor:");   
      f->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 10, 1, 2));
    
      fGridColor = new TGColorSelect(f, 0, -1);
      f->AddFrame(fGridColor, new TGLayoutHints(kLHintsLeft|kLHintsTop, 3, 1, 0, 2));
      fGridColor->Connect("ColorSelected(Pixel_t)", "TEveCaloLegoEditor", this, "DoGridColor(Pixel_t)");

      AddFrame(f, new TGLayoutHints(kLHintsTop, 1, 1, 1, 0));
   }
   // axis 
   {
      // font color
      TGHorizontalFrame* f = new TGHorizontalFrame(this);
      TGLabel* lab = new TGLabel(f, "FontColor:");   
      f->AddFrame(lab, new TGLayoutHints(kLHintsLeft|kLHintsBottom, 1, 8, 1, 2));
      
      fFontColor = new TGColorSelect(f, 0, -1);
      f->AddFrame(fFontColor, new TGLayoutHints(kLHintsLeft|kLHintsTop, 3, 1, 0, 2));
      fFontColor->Connect("ColorSelected(Pixel_t)", "TEveCaloLegoEditor", this, "DoFontColor(Pixel_t)");

      AddFrame(f, new TGLayoutHints(kLHintsTop, 1, 1, 1, 0));
   }

   fFontSize = new TEveGValuator(this, "FontSize:", 90, 0);
   fFontSize->SetLabelWidth(50);
   fFontSize->SetNELength(5);
   fFontSize->Build();
   fFontSize->SetLimits(0.01, 10, 100, TGNumberFormat::kNESRealTwo);
   fFontSize->SetToolTip("Font size relative to size of projected Z axis in %");
   fFontSize->Connect("ValueSet(Double_t)", "TEveCaloLegoEditor", this, "DoFontSize()");
   AddFrame(fFontSize, new TGLayoutHints(kLHintsTop, 4, 1, 1, 1));

   fNZStep = new TEveGValuator(this, "NZStep:", 90, 0);
   fNZStep->SetLabelWidth(50);
   fNZStep->SetNELength(5);
   fNZStep->SetShowSlider(kFALSE);
   fNZStep->Build();
   fNZStep->SetLimits(1, 20);
   fNZStep->SetToolTip("Number of labels along the Z axis.");
   fNZStep->Connect("ValueSet(Double_t)", "TEveCaloLegoEditor", this, "DoNZStep()");
   AddFrame(fNZStep, new TGLayoutHints(kLHintsTop, 4, 1, 1, 1));
}

//______________________________________________________________________________
void TEveCaloLegoEditor::SetModel(TObject* obj)
{
   // Set model object.

   fM = dynamic_cast<TEveCaloLego*>(obj); 
   fFontColor->SetColor(TColor::Number2Pixel(fM->GetFontColor()), kFALSE);
   fGridColor->SetColor(TColor::Number2Pixel(fM->GetGridColor()), kFALSE);

   fFontSize->SetValue(fM->GetFontSize());  
   fNZStep->SetValue(fM->GetNZStep());

}

//______________________________________________________________________________
void TEveCaloLegoEditor::DoFontColor(Pixel_t pixel)
{
   // Slot for FontColor.

   fM->SetFontColor(Color_t(TColor::GetColor(pixel)));
   Update();
}

//______________________________________________________________________________
void TEveCaloLegoEditor::DoGridColor(Pixel_t pixel)
{
   // Slot for GridColor.

   fM->SetGridColor(Color_t(TColor::GetColor(pixel)));
   Update();
}

//______________________________________________________________________________
void TEveCaloLegoEditor::DoFontSize()
{
   // Slot for FontSize.

   fM->SetFontSize((Int_t)fFontSize->GetValue());
   Update();
}

//______________________________________________________________________________
void TEveCaloLegoEditor::DoNZStep()
{
   // Slot for NZStep.

   fM->SetNZStep((Int_t)fNZStep->GetValue());
   Update();
}
