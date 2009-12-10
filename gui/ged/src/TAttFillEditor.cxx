// @(#)root/ged:$Id$
// Author: Ilka Antcheva   10/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttFillEditor                                                      //
//                                                                      //
//  Implements GUI for editing fill attributes.                         //                                             //
//             color and fill style                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TAttFillEditor.gif">
*/
//End_Html

#include "TAttFillEditor.h"
#include "TGedPatternSelect.h"
#include "TGColorSelect.h"
#include "TColor.h"

ClassImp(TAttFillEditor)

enum EFillWid {
   kCOLOR,
   kPATTERN
};


//______________________________________________________________________________
TAttFillEditor::TAttFillEditor(const TGWindow *p, Int_t width,
                               Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of fill attributes GUI.

   fPriority = 2;

   fAttFill = 0;
   
   MakeTitle("Fill");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);
   fPatternSelect = new TGedPatternSelect(f2, 1, kPATTERN);
   f2->AddFrame(fPatternSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fPatternSelect->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}

//______________________________________________________________________________
TAttFillEditor::~TAttFillEditor()
{ 
  // Destructor of fill editor.
}

//______________________________________________________________________________
void TAttFillEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fColorSelect->Connect("ColorSelected(Pixel_t)", "TAttFillEditor", this, "DoFillColor(Pixel_t)");
   fPatternSelect->Connect("PatternSelected(Style_t)", "TAttFillEditor", this, "DoFillPattern(Style_t)");
   fInit = kFALSE;
}

//______________________________________________________________________________
void TAttFillEditor::SetModel(TObject* obj)
{
   // Pick up the used fill attributes.

   TAttFill *attfill = dynamic_cast<TAttFill *>(obj);
   if (!attfill) return;
   
   fAttFill = attfill;
   fAvoidSignal = kTRUE;
   
   Color_t c = fAttFill->GetFillColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p, kFALSE);

   Style_t s = fAttFill->GetFillStyle();
   fPatternSelect->SetPattern(s, kFALSE);

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
void TAttFillEditor::DoFillColor(Pixel_t color)
{
   // Slot connected to the fill area color.

   if (fAvoidSignal) return;
   fAttFill->SetFillColor(TColor::GetColor(color));
   Update();
}

//______________________________________________________________________________
void TAttFillEditor::DoFillPattern(Style_t pattern)
{
   // Slot connected to the fill area pattern.

   if (fAvoidSignal) return;
   fAttFill->SetFillStyle(pattern);
   Update();
}

