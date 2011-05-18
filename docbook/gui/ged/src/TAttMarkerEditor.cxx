// @(#)root/ged:$Id$
// Author: Ilka Antcheva   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttMarkerEditor                                                    //
//                                                                      //
//  Implements GUI for editing marker attributes.                       //
//            color, style and size                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TAttMarkerEditor.gif">
*/
//End_Html


#include "TAttMarkerEditor.h"
#include "TGedMarkerSelect.h"
#include "TGColorSelect.h"
#include "TGNumberEntry.h"
#include "TColor.h"

ClassImp(TAttMarkerEditor)

enum EMarkerWid {
   kCOLOR,
   kMARKER,
   kMARKER_SIZE
};

//______________________________________________________________________________
TAttMarkerEditor::TAttMarkerEditor(const TGWindow *p, Int_t width,
                                   Int_t height,UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of marker attributes GUI.

   fAttMarker = 0;
   fSizeForText = kFALSE;
   
   MakeTitle("Marker");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);

   fMarkerType = new TGedMarkerSelect(f2, 1, kMARKER);
   f2->AddFrame(fMarkerType, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fMarkerType->Associate(this);

   fMarkerSize = new TGNumberEntry(f2, 0., 4, kMARKER_SIZE, 
                                   TGNumberFormat::kNESRealOne,
                                   TGNumberFormat::kNEANonNegative, 
                                   TGNumberFormat::kNELLimitMinMax, 0.2, 5.0);
   fMarkerSize->GetNumberEntry()->SetToolTipText("Set marker size");
   f2->AddFrame(fMarkerSize, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fMarkerSize->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}

//______________________________________________________________________________
TAttMarkerEditor::~TAttMarkerEditor()
{
   // Destructor of marker editor.
}

//______________________________________________________________________________
void TAttMarkerEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fColorSelect->Connect("ColorSelected(Pixel_t)", "TAttMarkerEditor", this, "DoMarkerColor(Pixel_t)");
   fMarkerType->Connect("MarkerSelected(Style_t)", "TAttMarkerEditor", this, "DoMarkerStyle(Style_t)");
   fMarkerSize->Connect("ValueSet(Long_t)", "TAttMarkerEditor", this, "DoMarkerSize()");
   (fMarkerSize->GetNumberEntry())->Connect("ReturnPressed()", "TAttMarkerEditor", this, "DoMarkerSize()");
   fInit = kFALSE;
}

//______________________________________________________________________________
void TAttMarkerEditor::SetModel(TObject* obj)
{
   // Pick up the values of used marker attributes.
   fAvoidSignal = kTRUE;

   fAttMarker = dynamic_cast<TAttMarker *>(obj);
   if (!fAttMarker) return;

   TString str = GetDrawOption();
   str.ToUpper();
   if (obj->InheritsFrom("TH2") && str.Contains("TEXT")) {
      fSizeForText = kTRUE;
   } else {
      fSizeForText = kFALSE;
   }   
   Style_t marker = fAttMarker->GetMarkerStyle();
   if ((marker==1 || marker==6 || marker==7) && !fSizeForText) {
      fMarkerSize->SetNumber(1.);
      fMarkerSize->SetState(kFALSE);
   } else {
      Float_t s = fAttMarker->GetMarkerSize();
      fMarkerSize->SetState(kTRUE);
      fMarkerSize->SetNumber(s);
   }
   fMarkerType->SetMarkerStyle(marker);

   Color_t c = fAttMarker->GetMarkerColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p);

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}


//______________________________________________________________________________
void TAttMarkerEditor::DoMarkerColor(Pixel_t color)
{
   // Slot connected to the marker color.

   if (fAvoidSignal) return;
   fAttMarker->SetMarkerColor(TColor::GetColor(color));
   Update();
}

//______________________________________________________________________________
void TAttMarkerEditor::DoMarkerStyle(Style_t marker)
{
   // Slot connected to the marker type.

   if (fAvoidSignal) return;
   if ((marker==1 || marker==6 || marker==7) && !fSizeForText) {
      fMarkerSize->SetNumber(1.);
      fMarkerSize->SetState(kFALSE);
   } else
      fMarkerSize->SetState(kTRUE);

   fAttMarker->SetMarkerStyle(marker);
   Update();
}

//______________________________________________________________________________
void TAttMarkerEditor::DoMarkerSize()
{
   // Slot connected to the marker size.

   if (fAvoidSignal) return;
   Style_t marker = fAttMarker->GetMarkerStyle();
   if ((marker==1 || marker==6 || marker==7) && !fSizeForText) {
      fMarkerSize->SetNumber(1.);
      fMarkerSize->SetState(kFALSE);
   } else
      fMarkerSize->SetState(kTRUE);
   Float_t size = fMarkerSize->GetNumber();
   fAttMarker->SetMarkerSize(size);
   Update();
}
