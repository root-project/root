// @(#)root/ged:$Name:  $:$Id: TAttLineEditor.cxx,v 1.9 2006/04/07 15:13:14 antcheva Exp $
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
#include "TGColorDialog.h"
#include "TGClient.h"
#include "TGComboBox.h"
#include "TColor.h"
#include "TAttLine.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TGraph.h"

ClassImp(TAttLineEditor)

enum ELineWid {
   kCOLOR,
   kLINE_WIDTH,
   kLINE_STYLE
};


//______________________________________________________________________________
TAttLineEditor::TAttLineEditor(const TGWindow *p, Int_t id, Int_t width,
                               Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of line attributes GUI.

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

   TClass *cl = TAttLine::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TAttLineEditor::~TAttLineEditor()
{
   // Destructor of line editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
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
void TAttLineEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used line attributes.

   fModel = 0;
   fPad = 0;

   if (!obj || !obj->InheritsFrom(TAttLine::Class()) || obj->InheritsFrom(TVirtualPad::Class())) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;
   
   fAttLine = dynamic_cast<TAttLine *>(fModel);

   fStyleCombo->Select(fAttLine->GetLineStyle(), kFALSE);

   if (fModel->InheritsFrom(TGraph::Class())) {
      fWidthCombo->Select(TMath::Abs(fAttLine->GetLineWidth()%100), kFALSE);
   } else {
      fWidthCombo->Select(fAttLine->GetLineWidth(), kFALSE);
   }

   Color_t c = fAttLine->GetLineColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p, kFALSE);

   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TAttLineEditor::DoLineColor(Pixel_t color)
{
   // Slot connected to the line color.

   fAttLine->SetLineColor(TColor::GetColor(color));
   Update();
}


//______________________________________________________________________________
void TAttLineEditor::DoLineStyle(Int_t style)
{
   // Slot connected to the line style.

   fAttLine->SetLineStyle(style);
   Update();
}


//______________________________________________________________________________
void TAttLineEditor::DoLineWidth(Int_t width)
{
   // Slot connected to the line width.

   if (fModel->InheritsFrom(TGraph::Class())) {
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
