// @(#)root/ged:$Name:  $:$Id: TAttMarkerEditor.cxx,v 1.3 2004/06/25 17:13:23 brun Exp $
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
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/mark_e.gif">
*/
//End_Html


#include "TAttMarkerEditor.h"
#include "TGedMarkerSelect.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"
#include "TGComboBox.h"
#include "TGClient.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "snprintf.h"


ClassImp(TGedFrame)
ClassImp(TAttMarkerEditor)

enum {
   kCOLOR,
   kMARKER,
   kMARKER_SIZE
};

//______________________________________________________________________________
TAttMarkerEditor::TAttMarkerEditor(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height,UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of marker attributes GUI.

   fAttMarker = 0;
   
   MakeTitle("Marker");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);

   fMarkerSelect = new TGedMarkerSelect(f2, 1, kMARKER);
   f2->AddFrame(fMarkerSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fMarkerSelect->Associate(this);

   fSizeCombo = BuildMarkerSizeComboBox(f2, kMARKER_SIZE);
   f2->AddFrame(fSizeCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fSizeCombo->Resize(50, 20);
   fSizeCombo->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TClass *cl = TAttMarker::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________
TAttMarkerEditor::~TAttMarkerEditor()
{
   // Destructor of marker editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________
void TAttMarkerEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fColorSelect->Connect("ColorSelected(Pixel_t)", "TAttMarkerEditor", this, "DoMarkerColor(Pixel_t)");
   fMarkerSelect->Connect("MarkerSelected(Style_t)", "TAttMarkerEditor", this, "DoMarkerStyle(Style_t)");
   fSizeCombo->Connect("Selected(Int_t)", "TAttMarkerEditor", this, "DoMarkerSize(Int_t)");
   fInit = kFALSE;
}

//______________________________________________________________________________
void TAttMarkerEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the values of used marker attributes.

   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAttMarker"))
   {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   fAttMarker = dynamic_cast<TAttMarker *>(fModel);

   Float_t s = fAttMarker->GetMarkerSize();
   s = TMath::Nint(s * 5);

   if (s > 15) s = 15;

   if (s < 1)  s = 1;

   fSizeCombo->Select((Int_t) s);

   fMarkerSelect->SetMarkerStyle(fAttMarker->GetMarkerStyle());

   Color_t c = fAttMarker->GetMarkerColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p);

   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
TGComboBox* TAttMarkerEditor::BuildMarkerSizeComboBox(TGFrame* parent, Int_t id)
{
   // Marker size combobox.

   char a[100];
   TGComboBox *c = new TGComboBox(parent, id);

   for (int i = 1; i <= 15; i++) {
      snprintf(a, 100, "%.1f", 0.2*i);
      c->AddEntry(a, i);
   }

   return c;
}

//______________________________________________________________________________
void TAttMarkerEditor::DoMarkerColor(Pixel_t color)
{
   // Slot connected to the fill area color.

   fAttMarker->SetMarkerColor(TColor::GetColor(color));
   Update();
}

//______________________________________________________________________________
void TAttMarkerEditor::DoMarkerStyle(Style_t marker)
{
   // Slot connected to the fill area pattern.

   fAttMarker->SetMarkerStyle(marker);
   Update();
}

//______________________________________________________________________________
void TAttMarkerEditor::DoMarkerSize(Int_t size)
{
   // Slot connected to the fill area pattern.

   fAttMarker->SetMarkerSize(0.2 * size);
   Update();
}
