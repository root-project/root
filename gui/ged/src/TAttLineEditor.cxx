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
//  TAttLineEditor                                                      //
//                                                                      //
//  Implements GUI for editing line attributes.                         //                                             //
//           color, line width, line style                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TAttLineEditor.gif">
*/
//End_Html


#include "TAttLineEditor.h"
#include "TGColorSelect.h"
#include "TGComboBox.h"
#include "TColor.h"
#include "TGraph.h"

ClassImp(TAttLineEditor)

enum ELineWid {
   kCOLOR,
   kLINE_WIDTH,
   kLINE_STYLE
};


//______________________________________________________________________________
TAttLineEditor::TAttLineEditor(const TGWindow *p, Int_t width,
                               Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor of line attributes GUI.

   fPriority = 1;
   fAttLine = 0;

   MakeTitle("Line");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);

   fStyleCombo = new TGLineStyleComboBox(this, kLINE_STYLE);
   fStyleCombo->Resize(137, 20);
   AddFrame(fStyleCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fStyleCombo->Associate(this);

   fWidthCombo = new TGLineWidthComboBox(f2, kLINE_WIDTH);
   fWidthCombo->Resize(91, 20);
   f2->AddFrame(fWidthCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fWidthCombo->Associate(this);
}

//______________________________________________________________________________
TAttLineEditor::~TAttLineEditor()
{
   // Destructor of line editor.
}

//______________________________________________________________________________
void TAttLineEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fColorSelect->Connect("ColorSelected(Pixel_t)", "TAttLineEditor", this, "DoLineColor(Pixel_t)");
   fStyleCombo->Connect("Selected(Int_t)", "TAttLineEditor", this, "DoLineStyle(Int_t)"); 
   fWidthCombo->Connect("Selected(Int_t)", "TAttLineEditor", this, "DoLineWidth(Int_t)"); 

   fInit = kFALSE;
}

//______________________________________________________________________________
void TAttLineEditor::SetModel(TObject* obj)
{
   // Pick up the used line attributes.

   TAttLine *attline = dynamic_cast<TAttLine*>(obj);
   if (!attline) return;
   
   fAttLine = attline;
   fAvoidSignal = kTRUE;

   fStyleCombo->Select(fAttLine->GetLineStyle());

   if (obj->InheritsFrom(TGraph::Class())) {
      fWidthCombo->Select(TMath::Abs(fAttLine->GetLineWidth()%100));
   } else {
      fWidthCombo->Select(fAttLine->GetLineWidth());
   }

   Color_t c = fAttLine->GetLineColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p);

   if (fInit) ConnectSignals2Slots();

   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
void TAttLineEditor::DoLineColor(Pixel_t color)
{
   // Slot connected to the line color.

   if (fAvoidSignal) return;
   fAttLine->SetLineColor(TColor::GetColor(color));
   Update();
}


//______________________________________________________________________________
void TAttLineEditor::DoLineStyle(Int_t style)
{
   // Slot connected to the line style.

   if (fAvoidSignal) return;
   fAttLine->SetLineStyle(style);
   Update();
}


//______________________________________________________________________________
void TAttLineEditor::DoLineWidth(Int_t width)
{
   // Slot connected to the line width.

   if (fAvoidSignal) return;
   if (dynamic_cast<TGraph*>(fAttLine)) {
      Int_t graphLineWidth = 100*Int_t(fAttLine->GetLineWidth()/100);
      if (graphLineWidth >= 0) {
         fAttLine->SetLineWidth(graphLineWidth+width);
      } else {
         fAttLine->SetLineWidth(-(TMath::Abs(graphLineWidth)+width));
      }
   } else {
      fAttLine->SetLineWidth(width);
   }
   Update();
}
